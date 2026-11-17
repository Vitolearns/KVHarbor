#!/usr/bin/env python3
"""Replay greedy samplers in balanced blocks and save per-request timings."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import threading
import time

from llama_runtime import ROOT, atomic_json, digest

MODES = ("legacy", "chain", "direct", "backend")
# Williams design: each treatment appears in every position; ordered neighbours balance.
ORDER = (("legacy", "chain", "backend", "direct"),
         ("chain", "direct", "legacy", "backend"),
         ("direct", "backend", "chain", "legacy"),
         ("backend", "legacy", "direct", "chain"))


class Hardware:
    def __init__(self):
        self.done = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        start = time.monotonic()
        while not self.done.is_set():
            query = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,pstate,clocks.sm,clocks.mem",
                     "--format=csv,noheader,nounits"]
            try:
                result = subprocess.run(query, text=True, capture_output=True, timeout=5)
                self.rows.append({"elapsed_s": time.monotonic() - start, "exit_code": result.returncode,
                                  "csv": result.stdout.strip(), "error": result.stderr.strip()})
            except (OSError, subprocess.TimeoutExpired) as error:
                self.rows.append({"elapsed_s": time.monotonic() - start, "error": str(error)})
            self.done.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=ROOT / "benchmarks/kvharbor_sampling.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat < 1 or args.warmup < 0 or not 128 <= args.context <= 32768 or not 1 <= args.threads <= 128:
        parser.error("invalid rounds, repeat, warmup, context or threads")
    model = args.model.resolve(strict=True)
    model_stat = (model.stat().st_size, model.stat().st_mtime_ns)
    binary = ROOT / "build-serving-cuda/kvharbor-runtime-bench"
    binary_sha = digest(binary)
    fixtures = [json.loads(line) for line in args.workload.read_text().splitlines() if line.strip()]
    if not fixtures:
        parser.error("empty workload")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "workload.jsonl").write_text("".join(json.dumps(row["body"], ensure_ascii=False) + "\n" for row in fixtures))
    sources = [*ROOT.glob("serving/src/*.cc"), ROOT / "serving/include/kvharbor/serving/runtime.h",
               ROOT / "serving/CMakeLists.txt", Path(__file__).resolve(), ROOT / "scripts/native_serving.sh",
               ROOT / "configs/runtime/llama.json"]
    report = {"schema_version": 1, "scope": "unprofiled native worker wall time; model load and warmups excluded; not HTTP capacity",
              "config": {"rounds": args.rounds, "repeat": args.repeat, "warmup": args.warmup,
                         "context": args.context, "threads": args.threads, "balanced_positions": args.rounds % 4 == 0,
                         "order": ORDER, "gpu_clock_policy": "observed, not locked"},
              "model": {"file": model.name, "bytes": model.stat().st_size, "sha256": digest(model)},
              "binary_sha256": binary_sha, "workload_sha256": digest(args.workload),
              "runtime_lock": json.loads((ROOT / "configs/runtime/llama.json").read_text()),
              "system": {"kernel": platform.release(), "machine": platform.machine(),
                         "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                            if line.startswith("model name")), "unknown")},
              "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
              "fixture_ids": [row["id"] for row in fixtures], "blocks": [], "status": "running"}
    env = {**os.environ, "GGUF_MODEL": str(model), "KVHARBOR_OUTPUT_DIR": str(out), "KVHARBOR_AUDIO": "0"}
    try:
        for round_id in range(args.rounds):
            for position, mode in enumerate(ORDER[round_id % 4]):
                assert digest(binary) == binary_sha, "binary changed during comparison"
                assert (model.stat().st_size, model.stat().st_mtime_ns) == model_stat, "model changed during comparison"
                tag = f"r{round_id}-{position}-{mode}"
                command = ["bash", "scripts/native_serving.sh", "bench", "--context", str(args.context),
                           "--threads", str(args.threads), "--reuse-context", "--sampling", mode,
                           "--input", "/run-output/workload.jsonl", "--output", f"/run-output/{tag}.jsonl",
                           "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
                with Hardware() as hardware, (out / (tag + ".log")).open("w") as log:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                atomic_json(out / (tag + "-hardware.json"), hardware.rows)
                if code:
                    raise RuntimeError(f"benchmark exited {code}: {tag}.log")
                rows = [json.loads(line) for line in (out / (tag + ".jsonl")).read_text().splitlines()]
                measured = [row for row in rows if row["measured"]]
                if len(measured) != len(fixtures) * args.repeat or any(row["outcome"] != "completed" for row in measured):
                    raise RuntimeError("incomplete measured workload: " + tag)
                report["blocks"].append({"round": round_id, "position": position, "mode": mode,
                                         "raw_file": tag + ".jsonl", "hardware_file": tag + "-hardware.json",
                                         "requests": measured})
                atomic_json(out / "report.json", report)
                print(json.dumps({"round": round_id, "mode": mode, "requests": len(measured),
                                  "sampling_us_per_token": statistics.median(row["timings"]["sampling_ms"] * 1000 / row["details"]["sampling_calls"] for row in measured),
                                  "wall_p50_ms": statistics.median(row["wall_ms"] for row in measured)}), flush=True)
        by_round = {}
        for block in report["blocks"]:
            by_round.setdefault(block["round"], {})[block["mode"]] = block["requests"]
        matched = {mode: 0 for mode in MODES if mode != "legacy"}
        for modes in by_round.values():
            reference = {row["id"]: row for row in modes["legacy"]}
            for mode, rows in modes.items():
                for row in rows:
                    if row["details"]["token_ids"] != reference[row["id"]]["details"]["token_ids"]:
                        raise RuntimeError("token mismatch: " + mode + "/" + row["id"])
                    if mode != "legacy":
                        matched[mode] += 1
        report["verified_matching_token_sequences"] = matched
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        atomic_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "matched_sequences": matched}), flush=True)


if __name__ == "__main__":
    main()
