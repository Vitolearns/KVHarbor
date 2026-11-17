#!/usr/bin/env python3
"""Verify and unpack the pinned Nsight Systems CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from llama_runtime import ROOT, atomic_json, digest


def prepare():
    lock = json.loads((ROOT / "configs/runtime/profiler.json").read_text())
    cache = ROOT / ".cache/kvharbor/profiler"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / lock["file"]
    if not archive.is_file() or archive.stat().st_size != lock["bytes"] or digest(archive) != lock["sha256"]:
        partial = archive.with_suffix(archive.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--retry-all-errors",
                        "--connect-timeout", "20", "--max-time", "900", "--continue-at", "-",
                        "--output", str(partial), lock["url"]], check=True)
        if partial.stat().st_size != lock["bytes"] or digest(partial) != lock["sha256"]:
            raise RuntimeError("Nsight Systems package digest or length mismatch")
        os.replace(partial, archive)
    destination = ROOT / lock["install_directory"]
    binary = destination / lock["binary"]
    if not binary.is_file():
        subprocess.run(["dpkg-deb", "--extract", str(archive), str(destination)], check=True)
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if lock["version"] not in version:
        raise RuntimeError("installed profiler version does not match the lock")
    atomic_json(cache / "installation.json", {**lock, "binary": str(binary.relative_to(ROOT)),
                                            "installed_version": version})
    return binary


if __name__ == "__main__":
    print(prepare())
