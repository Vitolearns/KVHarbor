#!/usr/bin/env python3
"""Build a separate llama.cpp runtime for the Q4_K warp experiment."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil

from llama_runtime import ROOT, LOCK, atomic_json, command, digest, prepare

PATCH = ROOT / "patches/llama.cpp/perf-003-q4k-warps.patch"
TEST_PATCH = ROOT / "patches/llama.cpp/perf-003-mmvq-tests.patch"
SOURCE = ROOT / ".cache/kvharbor/perf003/llama.cpp"
BUILD = ROOT / "build-llama-perf003"
KERNEL = "ggml/src/ggml-cuda/mmvq.cu"
BASE_SOURCE_SHA = "6c8a629f890f64c8a514555326049df356e946d4091d3e22aa3f5f7403803b3b"
PATCHED_SOURCE_SHA = "8c01668be7acc27c8470e47a1d2fddcb15232fbc21b9ca65ae6247b573e77f59"
TEST_SOURCE = "tests/test-backend-ops.cpp"
BASE_TEST_SHA = "b93fd63ed2bed3e01000a2cc87dd425da59d8a7029555171719a04ecfc4e785a"
PATCHED_TEST_SHA = "fd6ca1af70f5ea33f3f25b43e5e9fd14e698d62bf983161b872cc462e80e05b9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=2)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    base = prepare(lock)
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, SOURCE, symlinks=True)
    for path, before, after, patch in (
        (KERNEL, BASE_SOURCE_SHA, PATCHED_SOURCE_SHA, PATCH),
        (TEST_SOURCE, BASE_TEST_SHA, PATCHED_TEST_SHA, TEST_PATCH),
    ):
        if digest(base / path) != before:
            raise RuntimeError(f"Original pinned source checksum mismatch: {path}")
        patch_command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", patch]
        command([*patch_command, "--dry-run"], cwd=base)
        if digest(SOURCE / path) == before:
            command(patch_command, cwd=SOURCE)
        if digest(SOURCE / path) != after:
            raise RuntimeError(f"Patched source checksum mismatch: {path}")

    baseline_build = ROOT / "build-llama-cuda"
    if not (baseline_build / "kvharbor-build.json").is_file():
        raise RuntimeError("Build the pinned baseline with scripts/llama_runtime.py first")
    if not BUILD.exists():
        shutil.copytree(baseline_build, BUILD, symlinks=True)
    # The original CMake build uses these same container paths. Only this
    # separate build directory receives the CUDA source change.
    cmake_flags = " ".join(shlex.quote(flag) for flag in lock["cmake_flags"])
    build = "\n".join([
        "set -euo pipefail",
        f"cmake -S /llama -B /build -G Ninja {cmake_flags} -DLLAMA_BUILD_TESTS=ON",
        f"cmake --build /build --target llama ggml-cuda test-backend-ops --parallel {args.jobs}",
    ])
    command(["docker", "run", "--rm", "--pull=never", "--network", "none",
             "--user", f"{os.getuid()}:{os.getgid()}",
             "--mount", f"type=bind,src={SOURCE},dst=/llama,readonly",
             "--mount", f"type=bind,src={BUILD},dst=/build",
             "--entrypoint", "bash", lock["build_image"], "-c", build])
    atomic_json(BUILD / "kvharbor-perf003-build.json", {
        "revision": lock["revision"], "build_image": lock["build_image"],
        "base_source_sha256": BASE_SOURCE_SHA, "patched_source_sha256": PATCHED_SOURCE_SHA,
        "patch_sha256": digest(PATCH), "cmake_flags": [*lock["cmake_flags"], "-DLLAMA_BUILD_TESTS=ON"],
        "test_patch_sha256": digest(TEST_PATCH), "patched_test_source_sha256": PATCHED_TEST_SHA,
        "binaries": {name: digest(BUILD / "bin" / name)
                     for name in ("libllama.so", "libggml-cuda.so", "test-backend-ops")},
    })


if __name__ == "__main__":
    main()
