"""Check real multi-context dispatch and compare greedy HTTP outputs across modes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capacity_bench import endpoint, idle_state, ledger_delta, request_outcome
from scripts.stream_bench import post_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--model", default="Qwen3.8-27B-Q4_K_M")
    parser.add_argument("--engine-order", choices=("fifo", "decode-first"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    args.drain_seconds = 30
    health = endpoint(args, "health")
    runtime = health["runtimes"]["llm"]
    assert runtime["backend"] == "llama.cpp", health
    assert runtime["engine_schedule"] == args.engine_order, health
    assert runtime["sequences_per_context"] == 1, health
    state = endpoint(args, "state")
    assert state["tasks"]["llm"]["slots"] == 3, state
    workload = ROOT / "benchmarks/kvharbor_text_mixed.jsonl"
    fixtures = [json.loads(line) for line in workload.read_text().splitlines() if line.strip()]
    before, _ = idle_state(args)

    def generate(index):
        fixture = fixtures[index % len(fixtures)]
        body = {"model": args.model, "stream": True, "temperature": 0,
                "stream_options": {"include_usage": True}, **fixture["body"]}
        result = post_stream(args.url, "/v1/chat/completions", body,
                             time.perf_counter_ns(), timeout=20)
        assert result["done"] and result["outcome"] == "completed", result
        return {"index": index, "case": fixture["id"], "usage": result["usage"],
                "output_sha256": result["output_sha256"]}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(generate, range(6)))
    after, _ = idle_state(args)
    delta = ledger_delta(before, after)
    assert delta["accepted"] == delta["started"] == delta["terminal"] == delta["slot_releases"] == 6, delta
    assert delta["outcomes"].get("completed") == 6, delta
    report = {"scope": "real-runtime-regression", "health": health,
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
              "requests": rows, "ledger": delta}
    if args.compare:
        reference = json.loads(args.compare.read_text())
        assert reference["workload_sha256"] == report["workload_sha256"]
        assert reference["requests"] == rows, "Greedy outputs or token usage changed across dispatch modes"
        report["matching_output_pairs"] = len(rows)
    before, _ = idle_state(args)
    partial = post_stream(args.url, "/v1/chat/completions",
                          {"model": args.model, "stream": True, "temperature": 0, "max_tokens": 512,
                           "prompt": "Write a numbered list of 100 detailed facts about water. Continue the list."},
                          time.perf_counter_ns(), timeout=20, deadline_ms=1000)
    after, _ = idle_state(args)
    fault = ledger_delta(before, after)
    assert partial["http_status"] == 200 and partial["first_output_ns"] is not None, partial
    assert request_outcome(partial) == "incomplete_sse", partial
    assert fault["accepted"] == fault["started"] == fault["terminal"] == fault["slot_releases"] == 1, fault
    assert fault["outcomes"].get("deadline") == 1, fault
    report["streamed_deadline"] = {"client_outcome": "incomplete_sse", "http_status": 200,
                                   "first_content_observed": True, "done": partial["done"],
                                   "ledger": fault}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"engine_order": args.engine_order, "completed": len(rows),
                      "matching_output_pairs": report.get("matching_output_pairs")}))


if __name__ == "__main__":
    main()
