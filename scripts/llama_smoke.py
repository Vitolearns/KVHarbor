#!/usr/bin/env python3
"""Verify real SSE responses and slot recovery after client cancellation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from stream_bench import get_json, post_stream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:8081")
    p.add_argument("--model",default="Qwen3.8-27B-Q4_K_M")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--native",action="store_true",help="observe the C++ service's worker slots")
    args=p.parse_args()
    base=args.base_url
    report={"schema_version":1,"scope":"functional-smoke","health":get_json(base,"/health"),
            "models":get_json(base,"/v1/models"),"consecutive":[]}
    endpoint="/v1/chat/completions"
    def active():
        if args.native:
            return get_json(base,"/state")["tasks"]["llm"]["active"]
        return sum(bool(slot.get("is_processing")) for slot in get_json(base,"/slots"))
    def run(prompt,n=16,**kwargs):
        body={"model":args.model,"messages":[{"role":"user","content":prompt}],"temperature":0,
              "max_tokens":n,"stream":True,"stream_options":{"include_usage":True},"cache_prompt":False}
        return post_stream(base,endpoint,body,time.perf_counter_ns(),capture_text=True,**kwargs)
    for prompt,expected in [("Answer in one word: What is the capital of France?","Paris"),
                            ("Answer with just the number: What is 17 plus 25?","42")]:
        row=run(prompt)
        report["consecutive"].append(row)
        if row["outcome"]!="completed" or expected not in row["text"] or not (row["usage"] or {}).get("completion_tokens"):
            raise RuntimeError("unexpected response: " + json.dumps(row))
    before=[]
    row=run("List the integers from 1 to 400 without any explanation.",256,
            cancel_after=1,on_content=lambda _:before.append(active()))
    if row["outcome"]!="cancelled" or not any(before):
        raise RuntimeError("cancellation did not observe an active slot")
    cancel_done=time.monotonic()
    while True:
        if not active():
            break
        if time.monotonic()-cancel_done>10:
            raise RuntimeError("cancelled slot did not return to idle")
        time.sleep(.05)
    report["cancel"]={"request":row,"active_slots_before":max(before),
                      "active_slots_after":0,"observed_recovery_ms":(time.monotonic()-cancel_done)*1000}
    report["after_cancel"]=run("Answer in one word: What is the capital of France?")
    if report["after_cancel"]["outcome"]!="completed" or "Paris" not in report["after_cancel"]["text"]:
        raise RuntimeError("failed to reuse backend after cancellation")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("SSE: consecutive requests, usage, done markers and cancelled slot recovery passed")


if __name__=="__main__":
    main()
