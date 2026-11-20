#!/usr/bin/env python3
"""Compare direct KVHarbor runtime calls with HTTP/SSE for identical request bodies."""
import argparse
import hashlib
import json
from pathlib import Path

from stream_bench import percentiles


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("workload", "direct-input", "direct", "sse", "health", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    cases = jsonl(args.workload)
    if not cases or [c["body"] for c in cases] != jsonl(args.direct_input):
        parser.error("direct input and HTTP workload must contain identical request bodies in order")
    with args.sse.open() as stream:
        http = json.load(stream)
    health = json.loads(args.health.read_text())["runtimes"]["llm"]
    backend = health["backend"]
    if backend not in ("llama.cpp", "tensorrt-edge-llm"):
        parser.error("expected health from a real text backend")
    if http.get("workload_sha256") != digest(args.workload) or http.get("scope") != "real-http-sse":
        parser.error("HTTP report workload digest or scope differs")
    direct = [row for row in jsonl(args.direct) if row["measured"]]
    http_rows = sorted(http["requests"], key=lambda x: x["index"])
    if not direct or len(direct) != len(http_rows) or len(direct) % len(cases):
        parser.error("supply equal numbers of complete workload repetitions")
    metrics = {p: {k: [] for k in ("ttft_ms", "tpot_ms", "e2e_ms")} for p in ("direct", "sse")}
    mismatches = []
    for index, (native, wire) in enumerate(zip(direct, http_rows)):
        rep, ci = divmod(index, len(cases))
        if (native["id"] != f"{rep}-{ci}" or wire["index"] != index or wire["case"] != cases[ci]["id"]
                or native["runtime"]["backend"] != backend
                or native["runtime"]["model"] != http["config"]["model"]):
            parser.error(f"input, model or backend identity differs for request {index}")
        if native["outcome"] != "completed" or wire["outcome"] != "completed" or not wire["done"]:
            parser.error(f"request {index} did not complete on both paths")
        if "text" not in wire:
            parser.error("capture SSE text with stream_bench.py --capture-text")
        fields = []
        if native["text"] != wire["text"]:
            fields.append("text")
        if any(native["usage"].get(k) != wire["usage"].get(k)
               for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            fields.append("usage")
        if native["details"]["finish_reason"] != wire["finish_reason"]:
            fields.append("finish_reason")
        if fields:
            mismatches.append({"index": index, "case": wire["case"], "fields": fields})
        metrics["direct"]["ttft_ms"].append(native["ttft_ms"])
        metrics["direct"]["e2e_ms"].append(native["wall_ms"])
        tokens = native["usage"]["completion_tokens"]
        last = native.get("last_output_ms")
        if tokens > 1 and last is not None and last >= native["ttft_ms"] >= 0:
            metrics["direct"]["tpot_ms"].append((last-native["ttft_ms"])/(tokens-1))
        for k in metrics["sse"]:
            metrics["sse"][k].append(wire[k])
    summary = {p: {k: percentiles(samples) for k, samples in series.items()}
               for p, series in metrics.items()}
    report = {"schema_version": 1, "backend": backend, "model": http["config"]["model"],
              "requests_per_path": len(direct), "exact_matches": len(direct)-len(mismatches),
              "mismatches": mismatches, "metrics": summary,
              "sha256": {k: digest(getattr(args,k)) for k in ("workload","direct_input","direct","sse","health")}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("backend", "requests_per_path", "exact_matches", "metrics")}))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
