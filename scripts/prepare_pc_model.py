#!/usr/bin/env python3
"""Fetch the pinned PC text or ASR checkpoint and verify every file."""
import argparse
import fcntl
import json
import time
from urllib.parse import urlencode

from llama_runtime import ROOT, atomic_json, command, digest


def install_model(entry):
    destination = ROOT / entry["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache/kvharbor"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / (entry["id"] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, spec in entry["files"].items():
            target = destination / name
            if target.is_file():
                if target.stat().st_size != spec["bytes"] or digest(target) != spec["sha256"]:
                    raise RuntimeError("Existing checkpoint file differs from the lock: " + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            query = urlencode({"Revision": spec["revision"], "FilePath": spec.get("path", name)})
            url = f"https://modelscope.cn/api/v1/models/{entry['repository']}/repo?{query}"
            if not partial.exists() or partial.stat().st_size != spec["bytes"]:
                command(["curl", "-fsSL", "--retry", "3", "--connect-timeout", "15",
                         "--continue-at", "-", "-o", str(partial), url])
            if partial.stat().st_size != spec["bytes"] or digest(partial) != spec["sha256"]:
                rejected = partial.with_name(partial.name + ".rejected-" + str(time.time_ns()))
                partial.rename(rejected)
                raise RuntimeError("Downloaded checkpoint failed verification: " + str(rejected))
            partial.rename(target)
        atomic_json(destination / ".kvharbor-asset.json", entry)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("Qwen3.5-0.8B", "Qwen3-ASR-1.7B-int8"))
    args = parser.parse_args()
    if args.model == "Qwen3.5-0.8B":
        filename, key = "edge-llm.json", "pc_reference_model"
    else:
        filename, key = "audio.json", "asr_1_7b"
    entry = json.loads((ROOT / "configs/runtime" / filename).read_text())[key]
    print(install_model(entry))


if __name__ == "__main__":
    main()
