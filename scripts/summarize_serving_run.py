#!/usr/bin/env python3
"""Summarize measured KVHarbor artifacts without including local model paths."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
from llama_runtime import ROOT, atomic_json, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--audio-dir",default="audio-confirmed",help="audio run subdirectory within the raw run")
    args=p.parse_args()
    root=args.run
    def data(name):
        return json.loads((root/name).read_text())
    def bench(name):
        d=data(name)
        return {"config":{k:v for k,v in d["config"].items() if k not in ("output","base_url")},
                "workload_sha256":d["workload_sha256"],"summary":d["summary"]}
    def memory(name):
        with (root/name).open() as f:
            rows=list(csv.reader(f))[1:]
        used=[int(r[1].split()[0]) for r in rows if len(r)>1 and r[1].split()[0].isdigit()]
        return {"samples":len(used),"sampled_peak_mib":max(used) if used else None}
    manifest=data("manifest.json")
    manifest["model"].pop("path",None)
    source_files=[*ROOT.glob("serving/src/*"),*ROOT.glob("serving/include/**/*.h"),
                  ROOT/"serving/CMakeLists.txt",ROOT/"scripts/llama_runtime.py",ROOT/"scripts/native_serving.sh",
                  ROOT/"scripts/stream_bench.py",ROOT/"scripts/audio_bench.py",ROOT/"scripts/asr_bench.py",
                  ROOT/"scripts/prepare_audio_runtime.py",ROOT/"scripts/summarize_serving_run.py",
                  ROOT/"serving/tests/test_http.py",ROOT/"serving/tests/test_audio_http.py",ROOT/"tests/test_stream_bench.py",
                  *ROOT.glob("configs/runtime/*.json"),ROOT/"benchmarks/kvharbor_text_mixed.jsonl",
                  ROOT/"benchmarks/kvharbor_deadline_burst.jsonl"]
    output={"schema_version":2,"scope":"local initial engineering validation, not release-scale capacity",
            "source_snapshot_scope":"current source at summary time; historical runs predate later speech/test additions",
            "environment":manifest,"source_sha256":{str(f.relative_to(ROOT)):digest(f) for f in sorted(source_files)},
            "llama_cpp_baseline":{"slot_count":2,"total_context":4096,
                "capacity":[bench(f"llama-c{c}.json") for c in (1,2,4)],"open_loop":bench("llama-open-loop.json"),
                "gpu":memory("llama-gpu.csv"),"sse_smoke":data("llama-sse-smoke.json")},
            "native_smoke":data("native-sse-smoke.json"),"native_faults":data("serving-fault-test-recheck.log"),"context_reuse":{}}
    prof={}
    http={}
    for mode in ("fresh","reuse"):
        prof[mode]=[r for r in map(json.loads,(root/f"profile-{mode}.jsonl").read_text().splitlines()) if r["measured"]]
        http[mode]=data(f"context-{mode}-http.json")["requests"]
        rows=prof[mode]
        assert len(rows)>0 and all(r["outcome"]=="completed" for r in rows)
        output["context_reuse"][mode]={"native_request_count":len(rows),"runtime":rows[0]["runtime"],
            "native_median_ms":{k:statistics.median(r["timings"][k] for r in rows) for k in
                ("context_create_ms","context_reset_ms","context_free_ms","prefill_ms","decode_ms","sampling_ms")},
            "native_ttft_p50_ms":statistics.median(r["ttft_ms"] for r in rows),
            "native_wall_p50_ms":statistics.median(r["wall_ms"] for r in rows),
            "profile_gpu":memory(f"profile-{mode}-gpu.csv"),"http":bench(f"context-{mode}-http.json")}
    assert [r["id"] for r in prof["fresh"]]==[r["id"] for r in prof["reuse"]]
    assert [r["details"]["token_ids"] for r in prof["fresh"]]==[r["details"]["token_ids"] for r in prof["reuse"]]
    assert all(r["outcome"]=="completed" for side in http.values() for r in side)
    assert [r["output_sha256"] for r in http["fresh"]]==[r["output_sha256"] for r in http["reuse"]]
    output["context_reuse"]["verified_matching_token_sequences"]=len(prof["fresh"])
    output["context_reuse"]["verified_matching_http_outputs"]=len(http["fresh"])
    audio=root/args.audio_dir/"audio-report.json"
    if audio.is_file():
        a=json.loads(audio.read_text())
        output["audio"]={k:v for k,v in a.items() if k!="phases"}
        output["audio"]["phases"]={k:{key:value for key,value in v.items() if key not in ("samples","requests")} for k,v in a["phases"].items()}
        output["audio"]["lock"]=json.loads((ROOT/"configs/runtime/audio.json").read_text())
        matches={}
        for task,key in (("llm","output_sha256"),("asr","text")):
            alone=a["phases"][task+"_alone"]["requests"][task]
            mixed=a["phases"]["mixed"]["requests"][task]
            assert len(alone)==len(mixed) and all(x[key]==y[key] for x,y in zip(alone,mixed)), task
            matches[task]=len(alone)
        output["audio"]["verified_matching_outputs"]=matches
        alone=a["phases"]["tts_alone"]["requests"]["tts"]
        mixed=a["phases"]["mixed"]["requests"]["tts"]
        assert len(alone)==len(mixed) and all(x["sample_rate"]==y["sample_rate"] for x,y in zip(alone,mixed))
        output["audio"]["tts_waveform_comparison"]={"request_pairs":len(alone),
            "bitwise_matches":sum(x["pcm_sha256"]==y["pcm_sha256"] for x,y in zip(alone,mixed)),
            "max_duration_delta_ms":max(abs(x["duration_s"]-y["duration_s"])*1000 for x,y in zip(alone,mixed)),
            "scope":"stochastic model waveform; format and non-silence checked, listening/MOS not evaluated"}
    for key,name in (("asr_demo","asr-demo-report.json"),("audio_validation","audio-negative-tests.json"),
                     ("verification","verification-recheck.json"),("tts_nondeterminism","tts-nondeterminism.json"),
                     ("documentation","documentation-validation.json"),("data_integrity","data-integrity-recheck.json"),
                     ("profiler_availability","profiler-availability.json")):
        if (root/name).is_file():
            output[key]=data(name)
    if "audio_validation" in output:
        records=[]
        for line in (root/"audio-server.log").read_text(errors="replace").splitlines():
            try:
                row=json.loads(line)
            except ValueError:
                continue
            if isinstance(row,dict) and "id" in row and "slot_released" in row:
                records.append(row)
        states=[value["final_state"] for key in ("audio_validation","asr_demo","audio")
                if (value:=output.get(key)) and "final_state" in value]
        state=max(states,key=lambda value:value["accepted"])
        assert len(records)==len({r["id"] for r in records})==state["accepted"]
        assert sum(r["slot_released"] for r in records)==state["slot_releases"]
        output["native_audio_ledger"]={"unique_terminal_records":len(records),"final_state":state}
        timeout=[r for r in records if r["id"]==output["audio"]["asr_deadline"]["id"]]
        assert len(timeout)==1 and timeout[0]["timings"]["decode_ms"]>0
        output["audio"]["asr_deadline"]["native_record"]=timeout[0]
    output["binary_records"]={name:(root/name).read_text().splitlines() for name in
                              ("native-binaries.sha256","native-audio-binary.sha256")}
    for mode in ("fifo","deadline"):
        if (root/f"policy-{mode}.json").is_file():
            output.setdefault("admission_policy",{})[mode]=bench(f"policy-{mode}.json")
    output["raw_sha256"]={str(f.relative_to(root)):digest(f) for f in sorted(root.rglob("*"))
                          if f.is_file() and f.suffix in (".json",".jsonl",".csv",".wav",".log",".sha256")}
    atomic_json(args.output,output)
    print(args.output)


if __name__=="__main__":
    main()
