#!/usr/bin/env python3
"""Compare CUDA output reservations and Q4_K kernel shapes with paired requests."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import uuid

from llama_runtime import ROOT, LOCK, atomic_json, digest
from sampling_bench import Hardware
from stream_bench import get_json, post_stream

MODES = {"reference": (0, 4), "one-row": (1, 4), "two-warps": (0, 2), "shape-dispatch": (0, "auto")}
DEFAULT_MODES = ("reference", "one-row", "two-warps")


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resource_summary(log, hardware):
    text = log.read_text(errors="replace")
    matches = re.findall(r"CUDA0\s+compute buffer size\s*=\s*([0-9.]+) MiB", text)
    if not matches:
        raise RuntimeError(f"No CUDA compute-buffer record in {log.name}")
    memory = [int(row["csv"].split(",")[0]) for row in hardware
              if row.get("exit_code") == 0 and row.get("csv", "").split(",")[0].strip().isdigit()]
    return {"cuda_compute_mib": float(matches[-1]), "sampled_gpu_peak_mib": max(memory, default=None)}


def wait_health(process, url):
    end = time.monotonic() + 90
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("Experiment server exited before readiness")
        try:
            if get_json(url, "/health", timeout=1).get("status") == "ok":
                return
        except OSError:
            time.sleep(.5)
    raise RuntimeError("Experiment server readiness timed out")


def run_http(command, fixtures, args, log, container):
    url = f"http://127.0.0.1:{args.port}"
    with log.open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            wait_health(proc, url)
            rows = []
            for i in range(args.warmup + len(fixtures) * args.repeat):
                measured = i >= args.warmup
                index = i - args.warmup if measured else i
                fixture = fixtures[index % len(fixtures)]
                payload = {**fixture["body"], "stream": True, "temperature": 0,
                           "stream_options": {"include_usage": True}}
                row = post_stream(url, "/v1/chat/completions", payload, time.perf_counter_ns())
                row.update(id=str(index) if measured else f"warmup-{index}", measured=measured)
                rows.append(row)
            return rows
        finally:
            subprocess.run(["docker", "stop", "--timeout", "20", container],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=35)
            proc.wait(timeout=10)


def metric(row, name, http):
    if name == "wall_ms":
        return row["e2e_ms"] if http else row["wall_ms"]
    if name == "ttft_ms":
        return row["ttft_ms"]
    if http:
        return row["tpot_ms"]
    tokens = row["usage"]["completion_tokens"]
    return row["timings"]["decode_ms"] / max(1, tokens - 1)


def compare(report, http):
    by_round = {}
    for block in report["blocks"]:
        by_round.setdefault(block["round"], {})[block["mode"]] = block
    results = {}
    for mode in report["config"]["modes"]:
        blocks = [block for block in report["blocks"] if block["mode"] == mode]
        rows = [row for block in blocks for row in block["requests"]]
        result = {
            "requests": len(rows), "verified_matching_outputs": 0,
            "cuda_compute_mib": [block["resources"]["cuda_compute_mib"] for block in blocks],
            "sampled_gpu_peak_mib": [block["resources"]["sampled_gpu_peak_mib"] for block in blocks],
            "p50_ms": {name: statistics.median(metric(row, name, http) for row in rows)
                       for name in ("wall_ms", "ttft_ms", "decode_per_token_ms")},
            "output_tokens_per_s": sum(row["usage"]["completion_tokens"] for row in rows) * 1000 /
                sum(metric(row, "wall_ms", http) for row in rows),
            "paired_wall_change_pct_by_round": [],
        }
        for cases in by_round.values():
            pairs = zip(cases["reference"]["requests"], cases[mode]["requests"], strict=True)
            changes = []
            for baseline, candidate in pairs:
                if baseline["id"] != candidate["id"]:
                    raise RuntimeError("Request order differs across modes")
                equal = (baseline["output_sha256"] == candidate["output_sha256"] if http else
                         baseline["details"]["token_ids"] == candidate["details"]["token_ids"])
                if not equal or baseline["usage"] != candidate["usage"]:
                    raise RuntimeError(f"Output mismatch for {mode}/{candidate['id']}")
                result["verified_matching_outputs"] += 1
                changes.append(100 * (metric(candidate, "wall_ms", http) /
                                      metric(baseline, "wall_ms", http) - 1))
            result["paired_wall_change_pct_by_round"].append(statistics.median(changes))
        results[mode] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=ROOT / "benchmarks/kvharbor_sampling.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()
    if ("reference" not in args.modes or len(set(args.modes)) != len(args.modes)
            or min(args.rounds, args.repeat, args.warmup) < 1):
        parser.error("Select distinct modes including reference and positive run counts")

    model = args.model.resolve(strict=True)
    lock = json.loads(LOCK.read_text())
    model_sha = digest(model)
    if model_sha != lock["model"]["sha256"]:
        raise RuntimeError("The model does not match the locked benchmark weights")
    model_stat = (model.stat().st_size, model.stat().st_mtime_ns)
    build = ROOT / "build-serving-cuda"
    engine = ROOT / ("build-llama-q4k-shapes" if "shape-dispatch" in args.modes else "build-llama-perf003")
    engine_manifest = json.loads((engine / "kvharbor-perf003-build.json").read_text())
    exe = "kvharbor-server" if args.http else "kvharbor-runtime-bench"
    binary_sha = digest(build / exe)
    fixtures = load_rows(args.workload)
    if not fixtures:
        parser.error("The workload is empty")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "input.jsonl").write_text("".join(json.dumps(row["body"], ensure_ascii=False) + "\n" for row in fixtures))
    order = list(itertools.permutations(args.modes))
    report = {
        "schema_version": 1, "status": "running",
        "scope": "unprofiled HTTP request" if args.http else "unprofiled native worker",
        "config": {"modes": args.modes, "rounds": args.rounds, "repeat": args.repeat, "warmup": args.warmup,
                   "context": 2048, "threads": 4, "sampling": "direct", "reuse_context": True, "order": order},
        "model_sha256": model_sha, "binary_sha256": binary_sha, "workload_sha256": digest(args.workload),
        "engine": engine_manifest, "blocks": [],
    }
    try:
        for round_id in range(args.rounds):
            for position, mode in enumerate(order[round_id % len(order)]):
                if (digest(build / exe) != binary_sha or
                    any(digest(engine / "bin" / lib) != engine_manifest["binaries"][lib]
                        for lib in ("libllama.so", "libggml-cuda.so")) or
                    (model.stat().st_size, model.stat().st_mtime_ns) != model_stat):
                    raise RuntimeError("An experiment input changed during the comparison")
                tag = f"r{round_id}-{position}-{mode}"
                log = out / (tag + ".log")
                output_rows, warps = MODES[mode]
                container = "kvharbor-perf003-" + uuid.uuid4().hex[:8]
                command = [
                    "docker", "run", "--rm", "--pull=never", "--gpus", "all",
                    "--name", container, "--user", f"{os.getuid()}:{os.getgid()}",
                    "--network", "host" if args.http else "none",
                    "--mount", f"type=bind,src={build},dst=/native,readonly",
                    "--mount", f"type=bind,src={engine},dst=/build,readonly",
                    "--mount", f"type=bind,src={model},dst=/models/model.gguf,readonly",
                    "--mount", f"type=bind,src={out},dst=/run-output",
                    "--env", "LD_LIBRARY_PATH=/build/bin", "--env", f"KVHARBOR_CUDA_Q4K_WARPS={warps}",
                    "--entrypoint", f"/native/{exe}", lock["build_image"],
                    "--model", "/models/model.gguf", "--context", "2048", "--threads", "4",
                    "--sampling", "direct", "--reuse-context", "--max-output-rows", str(output_rows),
                ]
                raw_file = out / (tag + ".jsonl")
                with Hardware() as hardware:
                    if args.http:
                        command += ["--port", str(args.port), "--workers", "1"]
                        rows = run_http(command, fixtures, args, log, container)
                        raw_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                    else:
                        command += ["--input", "/run-output/input.jsonl", "--output", "/run-output/" + raw_file.name,
                                    "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
                        with log.open("w") as stream:
                            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                        rows = load_rows(raw_file)
                hardware_file = out / (tag + "-hardware.json")
                atomic_json(hardware_file, hardware.rows)
                measured = [row for row in rows if row["measured"]]
                if (len(measured) != len(fixtures) * args.repeat or
                    any(row["outcome"] != "completed" or not row["usage"]["completion_tokens"] for row in measured)):
                    raise RuntimeError(f"Incomplete requests in {tag}")
                resources = resource_summary(log, hardware.rows)
                report["blocks"].append({"round": round_id, "mode": mode, "position": position,
                    "command": command, "raw_file": raw_file.name, "raw_sha256": digest(raw_file),
                    "hardware_file": hardware_file.name, "resources": resources, "requests": measured})
                atomic_json(out / "report.json", report)
                print(json.dumps({"round": round_id, "mode": mode, **resources,
                                  "wall_p50_ms": statistics.median(metric(row, "wall_ms", args.http) for row in measured)}), flush=True)
        report["comparison"] = compare(report, args.http)
        report["status"] = "completed"
        print(json.dumps(report["comparison"]), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(out / "report.json", report)


if __name__ == "__main__":
    main()
