#!/usr/bin/env python3
"""Validate paired CUDA runs and report memory and latency."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess

from llama_runtime import ROOT, atomic_json, digest


def read(root, relative):
    return json.loads((root / relative).read_text())


def comparison(root, relative):
    report = read(root, relative)
    if report["status"] != "completed":
        raise RuntimeError(f"Incomplete comparison: {relative}")
    return {key: report[key] for key in ("scope", "config", "model_sha256", "binary_sha256",
                                         "workload_sha256", "engine", "comparison")}


def profile(root, warps):
    directory = root / f"profile-{warps}"
    summary = read(directory, "summary.json")
    kernels = [row for row in summary["mmvq_launches"] if row["rows"] == 17408 and
               "(ggml_type)12, (int)1, (bool)1" in row["name"]]
    if len(kernels) != 1 or kernels[0]["warps"] != warps:
        raise RuntimeError(f"Unexpected fused Q4_K launch: {warps} warp(s)")
    return {"target_kernel": kernels[0], "all_kernel_ms": summary["all_kernel_ms"],
            "device_to_host": summary["device_to_host"],
            "token_sequences": [row["details"]["token_ids"] for row in summary["requests"]],
            "artifacts_sha256": summary["artifacts_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    native = comparison(root, "native/report.json")
    http = {name: comparison(root, path) for name, path in
            (("one-row", "http-memory/report.json"), ("two-warps", "http-warps/report.json"))}
    profiles = {str(w): profile(root, w) for w in (4, 2)}
    baseline, variant = profiles["4"], profiles["2"]
    if baseline.pop("token_sequences") != variant.pop("token_sequences"):
        raise RuntimeError("Profile token sequences do not match")
    if baseline["target_kernel"]["calls"] != variant["target_kernel"]["calls"]:
        raise RuntimeError("Profile launch counts do not match")

    numeric = {}
    for warps in (4, 2):
        with (root / f"mmvq-numeric-{warps}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2 or any(row["supported"] != "1" or row["error_message"] for row in rows):
            raise RuntimeError("Missing or failed Q4_K numeric validation")
        numeric[str(warps)] = rows

    result = native["comparison"]
    baseline_compute = statistics.median(result["reference"]["cuda_compute_mib"])
    candidate_compute = statistics.median(result["one-row"]["cuda_compute_mib"])
    sources = [*ROOT.glob("serving/src/*.cc"), ROOT / "serving/include/kvharbor/serving/runtime.h",
               ROOT / "serving/CMakeLists.txt", ROOT / "scripts/profile_runtime.py",
               ROOT / "scripts/build_mmvq_experiment.py", ROOT / "scripts/perf003_bench.py",
               ROOT / "scripts/summarize_perf003.py", *ROOT.glob("patches/llama.cpp/perf-003-*.patch"),
               ROOT / "configs/runtime/llama.json", ROOT / "benchmarks/kvharbor_sampling.jsonl"]
    raw = ["native/report.json", "http-memory/report.json", "http-warps/report.json",
           "mmvq-numeric-4.csv", "mmvq-numeric-2.csv", "gpu-before.csv", "service-final.json",
           "regression.json", "profile-4/summary.json", "profile-2/summary.json",
           "profile-4/manifest.json", "profile-2/manifest.json"]
    summary = {
        "schema_version": 1, "experiment": "PERF-003", "scope": "single-GPU local engineering comparison",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "hardware_before": (root / "gpu-before.csv").read_text().strip(),
        "native": native, "http": http, "cuda_profile": profiles, "numeric": numeric,
        "cuda_compute_mib_saved": round(baseline_compute - candidate_compute, 2),
        "fused_mmvq_profile_change_pct": 100 * (variant["target_kernel"]["total_ms"] /
                                              baseline["target_kernel"]["total_ms"] - 1),
        "regression": read(root, "regression.json"), "service": read(root, "service-final.json"),
        "raw_directory": root.name,
        "raw_sha256": {path: digest(root / path) for path in raw},
    }
    atomic_json(args.output, summary)
    print(args.output)


if __name__ == "__main__":
    main()
