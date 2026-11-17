#!/usr/bin/env python3
"""Download and verify the pinned speech runtime and models."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from llama_runtime import ROOT, atomic_json, command, digest

CACHE=ROOT/".cache/kvharbor"


def fetch(entry):
    file=CACHE/entry["file"]
    if file.is_file() and file.stat().st_size==entry["bytes"] and digest(file)==entry["sha256"]:
        return file
    part=file.with_suffix(file.suffix+".part")
    if entry.get("asset_api"):
        # Large GitHub releases can be interrupted by proxies. Advance only
        # after a complete range, then verify the entire upstream artifact.
        pos=part.stat().st_size if part.exists() else 0
        with tempfile.TemporaryDirectory(dir=CACHE,prefix="range-") as tmp:
            chunk=Path(tmp)/"bytes"
            while pos<entry["bytes"]:
                end=min(entry["bytes"],pos+64*1024*1024)-1
                for attempt in range(4):
                    try:
                        response=command(["curl","--fail","--location","--silent","--show-error",
                          "--connect-timeout","15","--max-time","120","--speed-limit","4096","--speed-time","30",
                          "--header","Accept: application/octet-stream","--range",f"{pos}-{end}",
                          "--write-out","%{http_code}","--output",chunk,
                          entry["asset_api"]+f"?offset={pos}&attempt={attempt}&ts={int(time.time())}"],capture_output=True)
                        if response.stdout==b"206" and chunk.stat().st_size==end-pos+1:
                            break
                    except Exception:
                        if attempt==3:
                            raise
                else:
                    raise RuntimeError("incomplete ASR asset range")
                with chunk.open("rb") as src, part.open("ab") as out:
                    shutil.copyfileobj(src,out)
                pos=end+1
                print(entry["id"],pos,"/",entry["bytes"],flush=True)
    else:
        command(["curl","--fail","--location","--silent","--show-error","--retry","2","--retry-all-errors",
                 "--connect-timeout","15","--max-time","1200","--continue-at","-","--output",part,entry["url"]])
    if part.stat().st_size!=entry["bytes"] or digest(part)!=entry["sha256"]:
        raise RuntimeError("archive digest or length mismatch: "+entry["file"])
    os.replace(part,file)
    return file


def install(entry,dest):
    marker=dest/".kvharbor-asset.json"
    if marker.is_file() and json.loads(marker.read_text()).get("sha256")==entry["sha256"]:
        return
    archive=fetch(entry)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unpack-",dir=dest.parent) as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp,filter="data")
        roots=list(Path(tmp).iterdir())
        if len(roots)!=1 or not roots[0].is_dir():
            raise RuntimeError("expected a single archive directory")
        if dest.exists():
            for src in roots[0].rglob("*"):
                if src.is_file():
                    old=dest/src.relative_to(roots[0])
                    if not old.is_file() or digest(old)!=digest(src):
                        raise RuntimeError("existing files differ from pinned archive: "+str(old))
        else:
            os.replace(roots[0],dest)
    atomic_json(marker,entry)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-only",action="store_true")
    args=parser.parse_args()
    lock=json.loads((ROOT/"configs/runtime/audio.json").read_text())
    CACHE.mkdir(parents=True,exist_ok=True)
    sdk=lock["sdk"]
    dest=CACHE/"sherpa"
    install(sdk,dest)
    header=dest/"include/sherpa-onnx/c-api/c-api.h"
    if not header.is_file() or digest(header)!=sdk["header_sha256"]:
        response=command(["curl","--fail","--location","--silent","--show-error","--connect-timeout","15",
                          "--max-time","60",sdk["header_api"]],capture_output=True)
        data=base64.b64decode(json.loads(response.stdout)["content"])
        if hashlib.sha256(data).hexdigest()!=sdk["header_sha256"]:
            raise RuntimeError("sherpa-onnx header digest mismatch")
        header.parent.mkdir(parents=True,exist_ok=True)
        tmp=header.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp,header)
    print("verified C++ SDK",sdk["version"],flush=True)
    if not args.sdk_only:
        def model(task):
            entry=lock[task]
            install(entry,ROOT/entry["destination"])
            print("verified",task,entry["id"],flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(model,("asr","tts")))


if __name__=="__main__":
    main()
