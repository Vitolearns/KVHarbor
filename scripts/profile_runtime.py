#!/usr/bin/env python3
"""Capture a native LLM CUDA timeline without CPU sampling or elevated container capabilities."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

from llama_runtime import ROOT, atomic_json, digest
from prepare_profiler import prepare


def trace_summary(database):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    queries = {
        "kernels": "select s.value name,count(*) calls,sum(k.end-k.start)/1e6 total_ms from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by k.shortName order by total_ms desc limit 12",
        "cuda_api": "select s.value name,count(*) calls,sum(k.end-k.start)/1e6 total_ms from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId group by k.nameId order by total_ms desc limit 12",
        "device_to_host": "select bytes,count(*) calls,sum(end-start)/1e6 total_ms from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 group by bytes order by calls desc",
        "mmvq_launches": """select d.value name,k.gridX rows,k.blockX block_x,k.blockY warps,
            k.registersPerThread registers_per_thread,k.staticSharedMemory shared_bytes,
            count(*) calls,sum(k.end-k.start)/1e6 total_ms
            from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName
            join StringIds d on d.id=k.demangledName where s.value='mul_mat_vec_q'
            group by k.demangledName,k.gridX,k.blockX,k.blockY,k.registersPerThread,k.staticSharedMemory
            order by total_ms desc""",
    }
    result = {key: [dict(row) for row in connection.execute(sql)] for key, sql in queries.items()}
    result["all_kernel_ms"] = connection.execute("select sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL request bodies or fixture rows containing body")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--sampling", choices=("legacy", "chain", "direct", "backend"), default="direct")
    parser.add_argument("--engine-build", type=Path, default=ROOT / "build-llama-cuda")
    parser.add_argument("--max-output-rows", type=int, choices=(0, 1), default=0)
    parser.add_argument("--q4k-warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if not 128 <= args.context <= 32768 or not 1 <= args.threads <= 128 or args.warmup < 0 or args.repeat < 1:
        parser.error("invalid context, threads, warmup or repeat")
    model = args.model.resolve(strict=True)
    engine = args.engine_build.resolve(strict=True)
    binary = ROOT / "build-serving-cuda/kvharbor-runtime-bench"
    nsys = prepare()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("home", "tmp"):
        (out / name).mkdir()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    workload = [row.get("body", row) for row in rows]
    if not workload:
        parser.error("empty workload")
    (out / "input.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in workload))
    shutil.copy2(binary, out / "runtime-bench")
    lock = json.loads((ROOT / "configs/runtime/llama.json").read_text())
    container_nsys = "/workspace/" + str(nsys.relative_to(ROOT))
    args_native = ["/run-output/runtime-bench", "--model", "/models/model.gguf", "--reuse-context",
                   "--context", str(args.context), "--threads", str(args.threads), "--sampling", args.sampling,
                   "--max-output-rows", str(args.max_output_rows),
                   "--input", "/run-output/input.jsonl", "--output", "/run-output/requests.jsonl",
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
    command = ["docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
               "--gpus", "all", "--network", "none",
               "--mount", f"type=bind,src={ROOT},dst=/workspace,readonly",
               "--mount", f"type=bind,src={engine},dst=/build,readonly",
               "--mount", f"type=bind,src={model},dst=/models/model.gguf,readonly",
               "--mount", f"type=bind,src={out},dst=/run-output",
               "--env", "LD_LIBRARY_PATH=/build/bin", "--env", "HOME=/run-output/home",
               "--env", f"KVHARBOR_CUDA_Q4K_WARPS={args.q4k_warps}",
               "--env", "TMPDIR=/run-output/tmp", "--entrypoint", container_nsys, lock["build_image"],
               "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
               "--force-overwrite=true", "--output=/run-output/trace", *args_native]
    sources = [*ROOT.glob("serving/src/*.cc"), ROOT / "serving/include/kvharbor/serving/runtime.h",
               ROOT / "serving/CMakeLists.txt", Path(__file__), ROOT / "configs/runtime/profiler.json"]
    manifest = {"schema_version": 1, "scope": "CUDA trace; profile timings are not unprofiled performance measurements",
                "sampling": args.sampling, "binary_sha256": digest(binary), "input_sha256": digest(out / "input.jsonl"),
                "max_output_rows": args.max_output_rows, "q4k_warps_requested": args.q4k_warps,
                "engine_sha256": {name: digest(engine / "bin" / name) for name in ("libllama.so", "libggml-cuda.so")},
                "model": {"file": model.name, "bytes": model.stat().st_size, "sha256": digest(model)},
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources}, "command": command}
    atomic_json(out / "manifest.json", manifest)
    with (out / "profile.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"Nsight Systems exited {code}; inspect {out / 'profile.log'}")
    with (out / "export.log").open("w") as log:
        subprocess.run([str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
                        "--output=" + str(out / "trace.sqlite"), str(out / "trace.nsys-rep")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = trace_summary(out / "trace.sqlite")
    result["requests"] = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    result["artifacts_sha256"] = {name: digest(out / name) for name in ("trace.nsys-rep", "trace.sqlite", "requests.jsonl")}
    atomic_json(out / "summary.json", result)
    print(out / "summary.json")
    print(json.dumps({"sampling": args.sampling, "device_to_host": result["device_to_host"]}))


if __name__ == "__main__":
    main()
