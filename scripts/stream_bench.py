#!/usr/bin/env python3
"""Measure SSE latency and throughput under fixed or Poisson arrivals."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import socket
import time
import urllib.error
import urllib.request


class SSEDecoder:
    """Incremental SSE framing with CR, LF, CRLF and bounded events."""
    def __init__(self, limit=1024 * 1024):
        self.line = bytearray()
        self.data = []
        self.size = 0
        self.limit = limit
        self.skip_lf = False
        self.first_line = True

    def feed(self, block):
        for byte in block:
            if self.skip_lf:
                self.skip_lf = False
                if byte == 10:
                    continue
            if byte not in (10, 13):
                self.line.append(byte)
                if len(self.line) > self.limit:
                    raise ValueError("SSE line exceeds size limit")
                continue
            self.skip_lf = byte == 13
            raw = bytes(self.line)
            self.line.clear()
            if self.first_line:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
                self.first_line = False
            if not raw:
                if self.data:
                    yield "\n".join(self.data)
                self.data.clear()
                self.size = 0
                continue
            key, _, value = raw.partition(b":")
            if key == b"data":
                value = value.removeprefix(b" ")
                self.size += len(value) + 1
                if self.size > self.limit:
                    raise ValueError("SSE event exceeds size limit")
                self.data.append(value.decode("utf-8"))


def get_json(base, path, timeout=5):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read(1024 * 1024))


def post_stream(base, endpoint, body, planned_ns, timeout=120, capture_text=False,
                cancel_after=0, on_content=None, deadline_ms=None):
    data = json.dumps(body, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if deadline_ms is not None:
        headers["X-Request-Timeout-Ms"] = str(deadline_ms)
    request = urllib.request.Request(base.rstrip("/") + endpoint, data=data, headers=headers)
    started = time.perf_counter_ns()
    row = {"planned_ns": planned_ns, "dispatch_ns": started, "http_status": None,
           "first_output_ns": None, "chunk_ns": [], "outcome": "failed", "error": None,
           "finish_reason": None, "usage": None, "done": False}
    text = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row["http_status"] = response.status
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("expected text/event-stream")
            decoder = SSEDecoder()
            while not row["done"] and row["outcome"] != "cancelled":
                line = response.readline(65536)
                if not line:
                    break
                for event in decoder.feed(line):
                    if event == "[DONE]":
                        row["done"] = True
                        break
                    if not event:
                        continue
                    payload = json.loads(event)
                    if payload.get("error"):
                        raise ValueError("backend reported a stream error")
                    if isinstance(payload.get("usage"), dict):
                        row["usage"] = payload["usage"]
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason") is not None:
                        row["finish_reason"] = choice["finish_reason"]
                    content = (choice.get("delta") or {}).get("content") or choice.get("text")
                    if isinstance(content, str) and content:
                        stamp = time.perf_counter_ns()
                        if row["first_output_ns"] is None:
                            row["first_output_ns"] = stamp
                        row["chunk_ns"].append(stamp)
                        text.append(content)
                        if on_content:
                            on_content(len(text))
                        if cancel_after and len(text) >= cancel_after:
                            row["outcome"] = "cancelled"
                            break
            if row["done"]:
                row["outcome"] = "completed"
            elif row["outcome"] != "cancelled":
                row["error"] = "missing_done"
    except urllib.error.HTTPError as e:
        row["http_status"] = e.code
        row["outcome"] = "rejected" if e.code == 429 else "timed_out" if e.code in (408, 504) else "failed"
        row["error"] = f"http_{e.code}"
        e.close()
    except (TimeoutError, socket.timeout):
        row["outcome"] = "timed_out"
        row["error"] = "client_io_timeout"
    except (OSError, http.client.HTTPException, ValueError, AttributeError, TypeError, IndexError) as e:
        row["error"] = type(e).__name__
    row["complete_ns"] = time.perf_counter_ns()
    row["client_queue_ms"] = (started - planned_ns) / 1e6
    row["e2e_ms"] = (row["complete_ns"] - started) / 1e6
    row["ttft_ms"] = ((row["first_output_ns"] - started) / 1e6) if row["first_output_ns"] is not None else None
    times = row["chunk_ns"]
    row["chunk_gap_ms"] = [(b-a)/1e6 for a, b in zip(times, times[1:])]
    tokens = (row["usage"] or {}).get("completion_tokens")
    row["tpot_ms"] = (times[-1]-times[0])/1e6/(tokens-1) if type(tokens) is int and tokens>1 and times else None
    output = "".join(text)
    row["output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    if capture_text:
        row["text"] = output
    return row


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"n":len(values), **{f"p{p}": values[math.ceil(p/100*len(values))-1] for p in (50, 95, 99)}}


def summarize(rows, args):
    successes = [r for r in rows if r["outcome"] == "completed"]
    seconds = (max(r["complete_ns"] for r in rows)-min(r["planned_ns"] for r in rows))/1e9
    result = {"offered":len(rows), "measurement_s":seconds,
              "outcomes":{s:sum(r["outcome"]==s for r in rows) for s in ("completed","failed","rejected","timed_out","cancelled")},
              "requests_per_s":len(successes)/seconds,
              "output_tokens_per_s":sum((r["usage"] or {}).get("completion_tokens",0) or 0 for r in successes)/seconds,
              "ttft_ms":percentiles(r["ttft_ms"] for r in successes), "tpot_ms":percentiles(r["tpot_ms"] for r in successes),
              "e2e_ms":percentiles(r["e2e_ms"] for r in successes),
              "client_queue_ms":percentiles(r["client_queue_ms"] for r in rows),
              "chunk_gap_ms":percentiles(g for r in successes for g in r["chunk_gap_ms"])}
    if args.slo_ttft_ms is not None:
        good = [r for r in successes if r["ttft_ms"] is not None
                and r["ttft_ms"] + r["client_queue_ms"] <= args.slo_ttft_ms
                and (args.slo_tpot_ms is None or r["tpot_ms"] is not None and r["tpot_ms"] <= args.slo_tpot_ms)]
        result.update(goodput_requests_per_s=len(good)/seconds, slo_fraction_of_offered=len(good)/len(rows))
    result["by_group"]={group:{"offered":len([r for r in rows if r["group"]==group]),
        "completed":len([r for r in successes if r["group"]==group]),
        "ttft_ms":percentiles(r["ttft_ms"] for r in successes if r["group"]==group)}
        for group in sorted({r["group"] for r in rows})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.8-27B-Q4_K_M")
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rate", type=float, default=0.0, help="open-loop arrivals/s; zero submits a fixed backlog")
    parser.add_argument("--arrival", choices=("fixed", "poisson"), default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--slo-ttft-ms", type=float)
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--capture-text", action="store_true")
    parser.add_argument("--scope", choices=("real-http-sse", "test-double"), default="real-http-sse")
    args = parser.parse_args()
    if not 1<=args.requests<=100000 or not 1<=args.concurrency<=1024 or not math.isfinite(args.rate) or args.rate<0:
        parser.error("invalid workload size, concurrency or rate")
    cases = [json.loads(line) for line in args.workload.read_text().splitlines() if line.strip()]
    if not cases:
        parser.error("empty workload")
    rng = random.Random(args.seed)
    start = time.perf_counter_ns()
    due = start

    def run(index, case, planned):
        body = {"model":args.model, "stream":True, "stream_options":{"include_usage":True},
                "temperature":0, "cache_prompt":False, **case["body"]}
        row = post_stream(args.base_url,args.endpoint,body,planned,args.timeout,args.capture_text,
                          deadline_ms=case.get("deadline_ms"))
        return {"index":index,"case":case["id"],"group":case.get("group","default"),**row}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i in range(args.requests):
            if i and args.rate:
                due += round((rng.expovariate(args.rate) if args.arrival=="poisson" else 1/args.rate)*1e9)
            remain=(due-time.perf_counter_ns())/1e9
            if remain>0:
                time.sleep(remain)
            futures.append(pool.submit(run,i,cases[i%len(cases)],due))
        rows = [f.result() for f in futures]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    summary = summarize(rows,args)
    report = {"schema_version":1,"scope":args.scope,"created_at_unix":time.time(),"config":config,
              "workload_sha256":hashlib.sha256(args.workload.read_bytes()).hexdigest(),"summary":summary,"requests":rows}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    main()
