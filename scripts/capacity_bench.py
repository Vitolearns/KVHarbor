#!/usr/bin/env python3
"""Measure KVHarbor capacity, client queueing, terminal outcomes and recovery."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_BENCH = Path(__file__).with_name("stream_bench.py")
WORKLOAD = ROOT / "benchmarks/kvharbor_text_mixed.jsonl"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def percentiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    values = sorted(values)
    return {"n": len(values), **{name: values[math.ceil(len(values) * q) - 1]
            for name, q in (("p50", .5), ("p95", .95), ("p99", .99))}}


def request_outcome(row):
    status = row.get("http_status")
    if status == 429:
        return "rejected"
    if status in (408, 504) or row.get("outcome") in ("timed_out", "timeout") or any(
        word in str(row.get(field, "")).lower()
        for field in ("finish_reason", "error")
        for word in ("timeout", "timed out", "deadline")
    ):
        return "timeout"
    if status == 200 and row.get("done") and row.get("outcome") == "completed":
        return "completed"
    if status == 200 and not row.get("done"):
        return "incomplete_sse"
    return "failed"


def project(raw, args):
    outcomes = dict.fromkeys(("completed", "rejected", "timeout", "incomplete_sse", "failed"), 0)
    clean = []
    for row in raw["requests"]:
        label = request_outcome(row)
        outcomes[label] += 1
        usage = row.get("usage") or {}
        tokens = usage.get("completion_tokens")
        ttft, tpot = row.get("ttft_ms"), row.get("tpot_ms")
        queue_ms = row["client_queue_ms"]
        arrival_ttft = None if ttft is None else queue_ms + ttft
        meets_slo = (label == "completed" and arrival_ttft is not None
                     and arrival_ttft <= args.ttft_ms
                     and ((tpot is not None and tpot <= args.tpot_ms)
                          or (tpot is None and tokens in (0, 1))))
        clean.append({
            "index": row["index"], "case": row.get("case"), "group": row.get("group"),
            "outcome": label, "http_status": row.get("http_status"),
            "done": row.get("done"), "client_queue_ms": queue_ms,
            "ttft_ms": ttft, "arrival_ttft_ms": arrival_ttft, "tpot_ms": tpot,
            "e2e_ms": row.get("e2e_ms"), "completed_tokens": tokens,
            "prompt_tokens": usage.get("prompt_tokens"), "meets_slo": meets_slo,
            "output_sha256": row.get("output_sha256"),
        })
    passed = sum(row["meets_slo"] for row in clean)
    elapsed = raw["summary"]["measurement_s"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("measurement duration must be finite and positive")
    completed = [row for row in clean if row["outcome"] == "completed"]
    sent = [r["dispatch_ns"] for r in raw["requests"]]
    span_s = (max(sent) - min(sent)) / 1e9 if len(sent) > 1 else 0
    return {
        "offered": len(clean), "seconds": elapsed, "outcomes": outcomes,
        "realized_arrival_rps": (len(sent) - 1) / span_s if span_s > 0 else None,
        "slo_compliant": passed, "slo_fraction_of_offered": passed / len(clean),
        "goodput_rps": passed / elapsed,
        "output_tokens_per_s": sum(row["completed_tokens"] or 0 for row in completed) / elapsed,
        "client_queue_ms": percentiles([row["client_queue_ms"] for row in clean]),
        **{metric: percentiles([r[metric] for r in completed if r[metric] is not None])
           for metric in ("ttft_ms", "arrival_ttft_ms", "tpot_ms", "e2e_ms")},
        "requests": clean,
    }


def endpoint(args, name):
    with OPENER.open(f"{args.url.rstrip('/')}/{name}", timeout=3) as response:
        return json.load(response)


def ledger_state(raw):
    state = {key: raw[key] for key in ("active", "accepted", "started", "rejected", "slot_releases")}
    state["terminal"] = sum(raw["outcomes"].values())
    state["queued"] = sum(task["queued"] for task in raw["tasks"].values())
    state["outcomes"] = raw["outcomes"]
    if state["accepted"] != state["terminal"] + state["active"] + state["queued"]:
        raise RuntimeError("accepted requests do not balance with terminal, active and queued")
    if state["started"] != state["slot_releases"] + state["active"]:
        raise RuntimeError("started workers do not balance with releases and active workers")
    return state


def ledger_delta(before, after):
    delta = {key: after[key] - before[key] for key in before if key != "outcomes"}
    delta["outcomes"] = {key: after["outcomes"].get(key, 0) - before["outcomes"].get(key, 0)
                         for key in before["outcomes"].keys() | after["outcomes"].keys()}
    return delta


def idle_state(args):
    start = time.monotonic()
    while True:
        state = ledger_state(endpoint(args, "state"))
        if state["active"] == state["queued"] == 0:
            return state, (time.monotonic() - start) * 1000
        if time.monotonic() - start >= args.drain_seconds:
            raise RuntimeError("Requests did not drain")
        time.sleep(.05)


def replay(args, name, count, rate, arrival, seed):
    before, _ = idle_state(args)
    out = args.output_dir / f"{name}.raw.json"
    command = [
        sys.executable, str(STREAM_BENCH), "--base-url", args.url, "--model", args.model,
        "--workload", str(args.workload), "--output", str(out), "--scope", args.scope,
        "--requests", str(count), "--concurrency", str(args.concurrency),
        "--rate", str(rate), "--arrival", arrival, "--seed", str(seed),
        "--timeout", str(args.client_timeout),
    ]
    with (args.output_dir / f"{name}.client.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    after, drain_ms = idle_state(args)
    delta = ledger_delta(before, after)
    if delta["accepted"] + delta["rejected"] != count:
        raise RuntimeError("Client requests do not match server admission counters")
    return {"seed": seed, "offered_rps": rate, "arrival": arrival,
            "ledger": delta, "drain_poll_ms_after_client_complete": drain_ms,
            **project(json.loads(out.read_bytes()), args)}


def fingerprints(args, health):
    paths = [STREAM_BENCH, Path(__file__).resolve(), ROOT / "serving/CMakeLists.txt"]
    paths += sorted((ROOT / "serving").rglob("*.cc"))
    paths += sorted((ROOT / "serving").rglob("*.h"))
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    runtime = health.get("runtimes", {}).get("llm", {})
    lock = None
    engines = {}
    if args.scope == "real-http-sse":
        if runtime.get("backend") != args.expected_backend or runtime.get("model") != args.model:
            raise RuntimeError("Requested backend or model differs from the running service")
        filename = "edge-llm.json" if args.expected_backend == "tensorrt-edge-llm" else "llama.json"
        lock = json.loads((ROOT / "configs/runtime" / filename).read_text())
        if "model" in lock and lock["model"]["id"] != args.model:
            del lock["model"]
        if args.expected_backend == "tensorrt-edge-llm":
            if runtime.get("sdk_version") != lock["version"]:
                raise RuntimeError("Edge-LLM version differs from the pinned SDK")
            if not args.engine_dir or not args.engine_dir.is_dir():
                raise RuntimeError("Edge-LLM measurements require --engine-dir for artifact hashes")
            files = [p for p in sorted(args.engine_dir.rglob("*")) if p.is_file()]
            if not any(p.suffix.lower() in (".engine", ".plan", ".trt") for p in files):
                raise RuntimeError("No serialized engine in --engine-dir")
            engines = {str(p.relative_to(args.engine_dir)): sha(p) for p in files}
    return {
        "health": health,
        "measured_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "server_binary_sha256": sha(args.server_binary) if args.server_binary else None,
        "model_sha256": sha(args.model_file) if args.model_file else None,
        "runtime_lock": lock, "engine_sha256": engines,
    }


def gpu_samples(stop, samples, interval):
    command = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
               "temperature.gpu,clocks.current.sm,power.draw,driver_version", "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3)
            samples.append({"elapsed_monotonic_s": time.monotonic(), "exit_code": result.returncode,
                            "csv": result.stdout.strip()})
        except (OSError, subprocess.TimeoutExpired) as error:
            samples.append({"error": type(error).__name__})
            return
        stop.wait(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rates", type=float, nargs="+", default=[.4, .8, 1.6])
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--client-timeout", type=float, default=20)
    parser.add_argument("--drain-seconds", type=float, default=30)
    parser.add_argument("--ttft-ms", type=float, default=2000)
    parser.add_argument("--tpot-ms", type=float, default=75)
    parser.add_argument("--burst-count", type=int, default=24)
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--expected-backend", choices=("llama.cpp", "tensorrt-edge-llm"),
                        default="llama.cpp")
    parser.add_argument("--gpu-sample-interval", type=float, default=5)
    parser.add_argument("--scope", choices=("real-http-sse", "test-double"), default="real-http-sse")
    args = parser.parse_args()
    positive = [*args.rates, args.requests, args.repeats, args.concurrency,
                args.client_timeout, args.drain_seconds, args.ttft_ms, args.tpot_ms,
                args.gpu_sample_interval]
    if any(not math.isfinite(value) or value <= 0 for value in positive) or args.burst_count < 0:
        parser.error("Run parameters must be finite and positive. Zero burst count skips bursts.")
    health = endpoint(args, "health")
    if health.get("status") != "ok":
        raise RuntimeError("KVHarbor health check failed")
    evidence = fingerprints(args, health)
    args.output_dir.mkdir(parents=True)
    report = {
        "schema_version": 2, "scope": args.scope, "model_id": args.model,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "slo_ms": {"arrival_to_first_token": args.ttft_ms, "tpot": args.tpot_ms},
        "slo_ttft_includes_client_queue": True,
        "client_concurrency": args.concurrency, "client_timeout_seconds": args.client_timeout,
        **evidence, "runs": [], "overload_recovery": [], "gpu_samples": [],
        "gpu_sample_interval_s": args.gpu_sample_interval,
    }
    initial, _ = idle_state(args)
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_samples,
                               args=(stop, report["gpu_samples"], args.gpu_sample_interval), daemon=True)
    sampler.start()
    try:
        report["warmup"] = replay(args, "warmup", 3, 0, "fixed", 40)
        if report["warmup"]["outcomes"]["completed"] != 3:
            raise RuntimeError("Expected three successful warm-up requests")
        for repeat in range(args.repeats):
            indices = range(len(args.rates)) if repeat % 2 == 0 else reversed(range(len(args.rates)))
            for index in indices:
                name = f"repeat-{repeat + 1}-rate-{index + 1}"
                res = replay(args, name, args.requests, args.rates[index], "poisson", 41 + repeat)
                report["runs"].append({"repeat": repeat + 1, **res})
                print(f"{name}: {res['outcomes']}, SLO {res['slo_compliant']}/{res['offered']}", flush=True)
        if args.burst_count:
            for repeat in range(args.repeats):
                burst = replay(args, f"burst-{repeat + 1}", args.burst_count, 100, "fixed", 41 + repeat)
                recovery = replay(args, f"recovery-{repeat + 1}", 3, 0, "fixed", 40)
                report["overload_recovery"].append({"burst": burst, "recovery": recovery})
                if not burst["outcomes"]["rejected"] or recovery["outcomes"]["completed"] != 3:
                    raise RuntimeError("Expected overload rejection and three successful recovery requests")
        final, _ = idle_state(args)
        report["ledger"] = ledger_delta(initial, final)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        stop.set()
        sampler.join(timeout=4)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
