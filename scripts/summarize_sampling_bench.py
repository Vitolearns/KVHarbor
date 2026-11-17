#!/usr/bin/env python3
"""Summarize paired sampler timings from captured runs."""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

from llama_runtime import ROOT, atomic_json, digest
from profile_runtime import trace_summary


def median(values):
    return statistics.median(values)


def metrics(rows):
    return {
        "requests": len(rows),
        "sampling_us_per_token_p50": median(row["timings"]["sampling_ms"] * 1000 / row["details"]["sampling_calls"] for row in rows),
        "native_wall_p50_ms": median(row["wall_ms"] for row in rows),
        "native_ttft_p50_ms": median(row["ttft_ms"] for row in rows),
        "worker_output_tokens_per_s": sum(row["usage"]["completion_tokens"] for row in rows) * 1000 / sum(row["wall_ms"] for row in rows),
        "sampling_implementations": sorted({row["runtime"].get("sampling_implementation", "legacy") for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    data = lambda name: json.loads((root / name).read_text())
    paired = data("paired/report.json")
    if paired["status"] != "completed":
        raise RuntimeError("comparison has not completed")
    modes = ("legacy", "chain", "direct", "backend")
    rows = {mode: [row for block in paired["blocks"] if block["mode"] == mode for row in block["requests"]] for mode in modes}
    for mode in modes:
        assert all(row["runtime"]["sampling_mode"] == mode for row in rows[mode])
        if mode in ("direct", "backend"):
            assert all(row["details"]["cpu_candidate_rows_constructed"] == 0 for row in rows[mode])
        if mode == "backend":
            assert all(row["details"]["backend_sampled_tokens"] == row["details"]["sampling_calls"] for row in rows[mode])
    per_round = {}
    for block in paired["blocks"]:
        per_round.setdefault(str(block["round"]), {})[block["mode"]] = metrics(block["requests"])
    paired_wall = {}
    grouped = {}
    for block in paired["blocks"]:
        grouped.setdefault(block["round"], {})[block["mode"]] = {row["id"]: row for row in block["requests"]}
    for mode in modes[1:]:
        changes = []
        round_changes = []
        for group in grouped.values():
            current = []
            for key, row in group[mode].items():
                ref = group["legacy"][key]
                assert row["details"]["token_ids"] == ref["details"]["token_ids"]
                current.append((row["wall_ms"] / ref["wall_ms"] - 1) * 100)
            changes.extend(current)
            round_changes.append(median(current))
        paired_wall[mode] = {"median_change_percent": median(changes), "round_medians_percent": round_changes,
                             "round_median_range_percent": [min(round_changes), max(round_changes)]}
    baseline = trace_summary(root / "baseline.sqlite")
    profiles = {"legacy": baseline}
    original = [json.loads(line) for line in (root / "baseline-profile.jsonl").read_text().splitlines()]
    for mode in ("direct", "backend"):
        profile = data(f"profile-{mode}/summary.json")
        assert [r["details"]["token_ids"] for r in profile["requests"]] == [r["details"]["token_ids"] for r in original]
        profiles[mode] = {key: value for key, value in profile.items() if key not in ("requests", "artifacts_sha256")}
    for mode in ("legacy", "direct", "backend"):
        path = root / ("baseline.sqlite" if mode == "legacy" else f"profile-{mode}/trace.sqlite")
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        profiles[mode]["argmax_kernels"] = [dict(row) for row in connection.execute(
            "select s.value name,count(*) calls,sum(k.end-k.start)/1e6 total_ms from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where lower(s.value) like '%argmax%' group by k.shortName")]
        connection.close()
    snapshot = {
        "schema_version": 1, "scope": "local greedy execution experiment; repeated balanced blocks; not release-scale capacity",
        "analysis_source_sha256": {"scripts/summarize_sampling_bench.py": digest(Path(__file__).resolve())},
        "manifest": data("manifest.json"), "model": paired["model"], "system": paired["system"],
        "runtime_lock": paired["runtime_lock"], "config": paired["config"],
        "binary_sha256": paired["binary_sha256"], "workload_sha256": paired["workload_sha256"],
        "source_sha256": paired["source_sha256"],
        "metrics": {mode: metrics(values) for mode, values in rows.items()},
        "per_round": per_round, "paired_wall_vs_legacy": paired_wall,
        "verified_matching_token_sequences": paired["verified_matching_token_sequences"],
        "profiles": profiles,
        "method_limits": ["Sampling stage includes the native logits/token accessor and synchronization, not only argmax arithmetic.",
                          "Profile aggregates include warmup; profiled wall time is not used as the performance comparison.",
                          "Worker throughput excludes model load, warmups, HTTP and client admission queues.",
                          "Clocks are observed, not locked. Four balanced rounds do not establish a general capacity limit."],
    }
    for key, filename in (("http", "http/report.json"), ("regression", "regression.json"),
                          ("documentation", "documentation-validation.json"), ("service_final", "service-final.json")):
        if (root / filename).is_file():
            snapshot[key] = data(filename)
    if "http" in snapshot:
        assert snapshot["http"]["status"] == "completed"
        snapshot["http_metrics"] = {}
        for mode in ("legacy", "direct"):
            requests = [row for block in snapshot["http"]["blocks"] if block["mode"] == mode for row in block["requests"]]
            snapshot["http_metrics"][mode] = {"requests": len(requests),
                "ttft_p50_ms": median(row["ttft_ms"] for row in requests),
                "e2e_p50_ms": median(row["e2e_ms"] for row in requests)}
    snapshot["raw_sha256"] = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
                              if path.is_file() and not set(path.relative_to(root).parts).intersection(("home", "tmp"))}
    atomic_json(args.output, snapshot)
    print(args.output)
    print(json.dumps({"metrics": snapshot["metrics"], "paired_wall_vs_legacy": paired_wall,
                      "matching_sequences": snapshot["verified_matching_token_sequences"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
