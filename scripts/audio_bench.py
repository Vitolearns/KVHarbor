#!/usr/bin/env python3
"""Validate native audio delivery and replay LLM/ASR/TTS mixed requests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import wave

from stream_bench import get_json, percentiles, post_stream

FIXTURES = [
    ("en-weather", "The weather is warm today.", 2),
    ("en-chair", "The cat is sitting on the chair.", 2),
    ("zh-weather", "今天天气很好，我们一起去公园散步。", 45),
    ("zh-hello", "你好，欢迎使用语音服务。", 45),
]


def tts_request(base, text, voice):
    data=json.dumps({"input":text,"voice":voice,"speed":1}).encode()
    return urllib.request.Request(base+"/v1/audio/speech",data=data,headers={"Content-Type":"application/json"})


def tts(base, text, voice, wav_path=None):
    start=time.perf_counter()
    with urllib.request.urlopen(tts_request(base,text,voice),timeout=90) as response:
        rate=int(response.headers["X-Audio-Sample-Rate"])
        if response.headers.get_content_type()!="audio/pcm" or response.headers["X-Audio-Encoding"]!="pcm_s16le_mono":
            raise ValueError("unexpected audio content type")
        # First playable measurement uses 20 ms of PCM, not HTTP headers.
        first=response.read(max(2,rate//50*2))
        first_ms=(time.perf_counter()-start)*1000
        payload=first+response.read(rate*120+1)
        rid=response.headers["X-Request-Id"]
    wall=time.perf_counter()-start
    if not payload or len(payload)%2 or len(payload)>rate*120:
        raise ValueError("invalid PCM length")
    frames=len(payload)//2
    rms=math.sqrt(sum(x[0]**2 for x in struct.iter_unpack("<h",payload))/frames)/32768
    if rms<.001:
        raise ValueError("silent TTS output")
    if wav_path is not None:
        with wave.open(str(wav_path),"wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(payload)
    return {"id":rid,"first_playable_ms":first_ms,"e2e_ms":wall*1000,"duration_s":frames/rate,
            "sample_rate":rate,"frames":frames,"rms":rms,"rtf":wall/(frames/rate),"pcm_sha256":hashlib.sha256(payload).hexdigest()}


def asr(base, wav_path, deadline_ms=None):
    headers={"Content-Type":"audio/wav"}
    if deadline_ms is not None:
        headers["X-Request-Timeout-Ms"]=str(deadline_ms)
    request=urllib.request.Request(base+"/v1/audio/transcriptions",data=wav_path.read_bytes(),headers=headers)
    start=time.perf_counter()
    with urllib.request.urlopen(request,timeout=90) as response:
        row=json.load(response)
    row["e2e_ms"]=(time.perf_counter()-start)*1000
    row["rtf"]=row["e2e_ms"]/1000/row["audio_seconds"]
    if not row["text"].strip():
        raise ValueError("empty transcription")
    return row


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1):
            curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]


def quality(reference, observed, english):
    normalize=lambda s: unicodedata.normalize("NFKC",s).casefold()
    chars=lambda s: [x for x in normalize(s) if x.isalnum()]
    ref,got=chars(reference),chars(observed)
    result={"cer":edit_distance(ref,got)/len(ref)}
    if english:
        words=lambda s: re.findall(r"[a-z0-9]+",normalize(s))
        ref,got=words(reference),words(observed)
        result["wer"]=edit_distance(ref,got)/len(ref)
    return result


class Observe:
    def __init__(self,base,pid):
        self.base=base; self.pid=pid; self.rows=[]; self.done=threading.Event()
        self.thread=threading.Thread(target=self.sample,daemon=True)

    def sample(self):
        begin=time.monotonic()
        next_gpu=begin
        while not self.done.is_set():
            now=time.monotonic()
            row={"t_ms":(now-begin)*1000}
            try:
                s=get_json(self.base,"/state",timeout=2)
                row["active"]={t:v["active"] for t,v in s["tasks"].items()}
                row["queued"]={t:v["queued"] for t,v in s["tasks"].items()}
                if self.pid:
                    proc=Path(f"/proc/{self.pid}")
                    stats=(proc/"stat").read_text().rpartition(")")[2].split()
                    row["cpu_s"]=(int(stats[11])+int(stats[12]))/os.sysconf("SC_CLK_TCK")
                    status=(proc/"status").read_text()
                    row["rss_kib"]=int(re.search(r"^VmRSS:\s+(\d+)",status,re.M)[1])
                if now>=next_gpu:
                    gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],
                                       text=True,capture_output=True,timeout=2)
                    if gpu.returncode==0:
                        memory,util=gpu.stdout.splitlines()[0].split(",")
                        row["gpu_mib"]=int(memory); row["gpu_util_percent"]=int(util)
                    next_gpu=now+.25
            except (OSError,ValueError,TypeError,IndexError,subprocess.TimeoutExpired):
                pass
            self.rows.append(row)
            self.done.wait(.025)

    def __enter__(self):
        self.thread.start()
        return self.rows

    def __exit__(self,*_):
        self.done.set(); self.thread.join(timeout=5)


def wait_idle(base,task):
    end=time.monotonic()+15
    while time.monotonic()<end:
        state=get_json(base,"/state")
        if state["tasks"][task]["active"]==0:
            return state
        time.sleep(.03)
    raise RuntimeError(task+" worker did not return to idle")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:8082")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workload",type=Path,default=Path("benchmarks/kvharbor_text_mixed.jsonl"))
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--pid",type=int)
    p.add_argument("--tts-only",action="store_true")
    args=p.parse_args()
    if args.rounds<1:
        p.error("rounds must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    base=args.base_url.rstrip("/")
    report={"schema_version":2,"scope":"functional-synthetic-roundtrip-and-initial-mixed-load","health":get_json(base,"/health"),"fixtures":[],
            "initial_state":get_json(base,"/state")}
    report["config"]={"rounds":args.rounds,"workload_sha256":hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                      "phase_order":["llm_alone","asr_alone","tts_alone","mixed"],
                      "warmup":"fixture round trips and one LLM request before measured phases"}
    report["source_sha256"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("audio_bench.py","stream_bench.py")}
    if report["initial_state"]["active"] or any(v["queued"] for v in report["initial_state"]["tasks"].values()):
        raise RuntimeError("start the benchmark on an idle service")
    for tag,text,voice in FIXTURES:
        wav=args.output/(tag+".wav")
        record={"id":tag,"reference":text,"voice":voice,"tts":tts(base,text,voice,wav)}
        if not args.tts_only:
            record["asr"]=asr(base,wav)
            record["quality"]=quality(text,record["asr"]["text"],tag.startswith("en-"))
        report["fixtures"].append(record)
    if not args.tts_only:
        cases=[json.loads(l)["body"] for l in args.workload.read_text().splitlines() if l.strip()]
        def llm(i):
            body={**cases[i%len(cases)],"stream":True,"temperature":0}
            row=post_stream(base,"/v1/chat/completions",body,time.perf_counter_ns())
            if row["outcome"]!="completed":
                raise RuntimeError("LLM did not complete: "+str(row["error"]))
            return row
        def recognize(i):
            return asr(base,args.output/(FIXTURES[i%len(FIXTURES)][0]+".wav"))
        def synthesize(i):
            _,text,voice=FIXTURES[i%len(FIXTURES)]
            return tts(base,text,voice)

        llm(0)  # context warm-up, outside measured runs
        actions={"llm":llm,"asr":recognize,"tts":synthesize}
        phases={}
        for task,fn in actions.items():
            with Observe(base,args.pid) as samples:
                rows=[fn(i) for i in range(args.rounds)]
            phases[task+"_alone"]={"requests":{task:rows},"samples":samples}
        with ThreadPoolExecutor(max_workers=3) as pool, Observe(base,args.pid) as samples:
            mixed={t:[] for t in actions}
            for i in range(args.rounds):
                futures={t:pool.submit(fn,i) for t,fn in actions.items()}
                for t,future in futures.items():
                    mixed[t].append(future.result())
        phases["mixed"]={"requests":mixed,"samples":samples}
        for phase in phases.values():
            phase["peak_active"]=max((sum(x.get("active",{}).values()) for x in phase["samples"]),default=0)
            phase["peak_gpu_mib"]=max((x.get("gpu_mib",0) for x in phase["samples"]),default=0)
            phase["peak_rss_mib"]=max((x.get("rss_kib",0) for x in phase["samples"]),default=0)/1024
            cpu=[x for x in phase["samples"] if "cpu_s" in x]
            phase["cpu_core_equivalent"]=(cpu[-1]["cpu_s"]-cpu[0]["cpu_s"])*1000/(cpu[-1]["t_ms"]-cpu[0]["t_ms"]) if len(cpu)>1 else None
            phase["latency"]= {t:{"e2e_ms":percentiles(r["e2e_ms"] for r in rows),
                **({"first_ms":percentiles(r["ttft_ms"] for r in rows)} if t=="llm" else
                    {"first_ms":percentiles(r["first_playable_ms"] for r in rows)} if t=="tts" else {})}
                for t,rows in phase["requests"].items()}
        if phases["mixed"]["peak_active"]<2:
            raise RuntimeError("mixed run did not observe overlapping work")
        report["phases"]=phases
        try:
            asr(base,args.output/"en-weather.wav",deadline_ms=100)
            raise RuntimeError("expected a deadline response from ASR")
        except urllib.error.HTTPError as e:
            if e.code!=408:
                raise
            timeout_id=e.headers.get("X-Request-Id")
        still_busy=get_json(base,"/state")["tasks"]["asr"]["active"]
        start=time.monotonic()
        wait_idle(base,"asr")
        report["asr_deadline"]={"http_status":408,"id":timeout_id,"deadline_ms":100,
                                "active_after_http_timeout":still_busy,"observed_recovery_ms":(time.monotonic()-start)*1000}
        if still_busy!=1:
            raise RuntimeError("ASR deadline did not retain an executing worker")
        report["asr_recovery_probe"]=asr(base,args.output/"en-weather.wav")
        with urllib.request.urlopen(tts_request(base,"The weather is warm today. "*10,2),timeout=30) as response:
            response.read(2048)
            active=get_json(base,"/state")["tasks"]["tts"]["active"]
        start=time.monotonic()
        report["tts_disconnect"]={"active_before_disconnect":active,"state":wait_idle(base,"tts"),"observed_recovery_ms":(time.monotonic()-start)*1000}
        report["tts_recovery_probe"]=tts(base,"The weather is warm today.",2)
        report["final_state"]=get_json(base,"/state")
        initial,final=report["initial_state"],report["final_state"]
        accepted=final["accepted"]-initial["accepted"]
        released=final["slot_releases"]-initial["slot_releases"]
        outcomes=sum(final["outcomes"].values())-sum(initial["outcomes"].values())
        report["ledger_delta"]={"accepted":accepted,"slot_releases":released,"terminal_outcomes":outcomes}
        if any(v["active"] or v["queued"] for v in final["tasks"].values()) or not accepted==released==outcomes:
            raise RuntimeError("worker leases or terminal accounting did not reconcile")
    path=args.output/"audio-report.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(path)
    for r in report["fixtures"]:
        print(r["id"],r["tts"]["duration_s"],"seconds",r.get("quality"))
    if "phases" in report:
        print("mixed observed active tasks",report["phases"]["mixed"]["peak_active"])


if __name__=="__main__":
    main()
