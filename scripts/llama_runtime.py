#!/usr/bin/env python3
"""Build and run the pinned llama.cpp CUDA runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/runtime/llama.json"


def command(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def paths(lock):
    source = ROOT / ".cache" / ("llama.cpp-" + lock["revision"])
    build = ROOT / "build-llama-cuda"
    return source, build


def prepare(lock):
    source, _ = paths(lock)
    archive = ROOT / ".cache/kvharbor/llama-11cd988.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected = lock["archive_sha256"]
    if not archive.is_file() or digest(archive) != expected:
        temp = archive.with_suffix(".download")
        command(["curl", "--fail", "--location", "--retry", "2",
                 "--connect-timeout", "15", "--max-time", "300",
                 "--output", temp, lock["archive_url"]])
        if digest(temp) != expected:
            temp.unlink()
            raise RuntimeError("llama.cpp archive SHA-256 mismatch")
        os.replace(temp, archive)
    marker = source / ".kvharbor-source.json"
    if source.exists():
        if not marker.is_file() or json.loads(marker.read_text())["archive_sha256"] != expected:
            raise RuntimeError("existing source does not match lock; choose a new source directory")
        return source
    with tempfile.TemporaryDirectory(dir=archive.parent, prefix="llama-extract-") as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        extracted = Path(tmp) / ("llama.cpp-" + lock["revision"])
        atomic_json(extracted / ".kvharbor-source.json", lock)
        os.replace(extracted, source)
    return source


def build_runtime(lock, jobs):
    source = prepare(lock)
    _, build = paths(lock)
    build.mkdir(exist_ok=True)
    script = "\n".join([
        "set -euo pipefail",
        "cmake -S /llama -B /build -G Ninja " + " ".join(lock["cmake_flags"]),
        "cmake --build /build --target llama-cli llama-server --parallel " + str(jobs),
    ])
    command(["docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
             "--mount", f"type=bind,src={source},dst=/llama,readonly",
             "--mount", f"type=bind,src={build},dst=/build",
             "--entrypoint", "bash", lock["build_image"], "-c", script])
    atomic_json(build / "kvharbor-build.json", {
        "lock": lock,
        "built_at_unix": time.time(),
        "binaries": {name: digest(build / "bin" / name) for name in ("llama-cli", "llama-server")},
    })


def model_manifest(lock, model, output):
    model = model.resolve(strict=True)
    actual = digest(model)
    if model.stat().st_size != lock["model"]["bytes"] or actual != lock["model"]["sha256"]:
        raise RuntimeError("model does not match the pinned model identity")
    _, build = paths(lock)
    gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                   "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    base = command(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    atomic_json(output, {
        "schema_version": 1, "created_at_unix": time.time(), "git_head": base,
        "git_dirty": bool(command(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout),
        "model": {**lock["model"], "actual_sha256": actual, "path": str(model)},
        "runtime": json.loads((build / "kvharbor-build.json").read_text()),
        "gpu": gpu, "kernel": os.uname().release,
    })
    print(output)


def start(lock, args):
    source, build = paths(lock)
    model = args.model.resolve(strict=True)
    if model.stat().st_size != lock["model"]["bytes"]:
        raise RuntimeError("unexpected model size; run manifest for complete identity validation")
    if not (build / "kvharbor-build.json").is_file():
        raise RuntimeError("build the pinned runtime before starting")
    # A name collision fails instead of replacing another running container.
    argv = ["docker", "run", "--detach", "--pull=never", "--name", args.name,
            "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
            "--publish", f"127.0.0.1:{args.port}:8080", "--stop-timeout", "30",
            "--mount", f"type=bind,src={build},dst=/build,readonly",
            "--mount", f"type=bind,src={model},dst=/models/model.gguf,readonly",
            "--entrypoint", "/build/bin/llama-server", lock["build_image"],
            "--model", "/models/model.gguf", "--alias", lock["model"]["id"],
            "--host", "0.0.0.0", "--port", "8080", "--ctx-size", str(args.context),
            "--parallel", str(args.slots), "--n-gpu-layers", "99", "--threads", str(args.threads),
            "--batch-size", str(args.batch), "--ubatch-size", str(args.ubatch),
            "--metrics", "--slots", "--jinja", "--reasoning", "off"]
    command(argv)
    (ROOT / ".cache/kvharbor").mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / ".cache/kvharbor/llama-start.json", {
        "container": args.name, "argv": argv, "context": args.context,
        "slots": args.slots, "port": args.port, "started_at_unix": time.time(),
    })
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                if response.status == 200:
                    print("healthy", f"http://127.0.0.1:{args.port}")
                    return
        except (OSError, urllib.error.HTTPError):
            time.sleep(1)
    raise RuntimeError("runtime did not become healthy; inspect docker logs (container retained)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("build")
    p.add_argument("--jobs", type=int, default=4, choices=range(1, 33))
    p = sub.add_parser("manifest")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("start")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--name", default="kvharbor-llama-server")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--wait", type=int, default=180)
    for action in ("stop", "status"):
        p = sub.add_parser(action)
        p.add_argument("--name", default="kvharbor-llama-server")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    if args.action == "prepare": print(prepare(lock))
    elif args.action == "build": build_runtime(lock, args.jobs)
    elif args.action == "manifest": model_manifest(lock, args.model, args.output)
    elif args.action == "start": start(lock, args)
    elif args.action == "stop": command(["docker", "stop", "--time", "30", args.name])
    elif args.action == "status": command(["docker", "inspect", "--format", "{{json .State}}", args.name])


if __name__ == "__main__":
    main()
