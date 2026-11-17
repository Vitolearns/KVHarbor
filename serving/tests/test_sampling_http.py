"""Compare real SSE requests across greedy paths and verify backend cancellation recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from llama_runtime import atomic_json, digest
from stream_bench import get_json


def start(model, out, mode, tag):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    name = "kvharbor-sampling-" + uuid.uuid4().hex[:12]
    env = {**os.environ, "GGUF_MODEL": str(model), "KVHARBOR_OUTPUT_DIR": str(out),
           "KVHARBOR_NATIVE_NAME": name, "KVHARBOR_AUDIO": "0"}
    command = ["bash", "scripts/native_serving.sh", "serve", "--port", str(port), "--context", "2048",
               "--workers", "1", "--threads", "4", "--reuse-context", "--sampling", mode]
    log = (out / (tag + "-server.log")).open("w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"name": name, "base": f"http://127.0.0.1:{port}", "process": process, "log": log,
            "tag": tag, "mode": mode, "command": command}


def ready(server):
    end = time.monotonic() + 90
    while time.monotonic() < end:
        if server["process"].poll() is not None:
            raise RuntimeError("server exited before readiness: " + server["tag"])
        try:
            health = get_json(server["base"], "/health", timeout=1)
            if health["status"] == "ok":
                assert health["runtimes"]["llm"]["sampling_mode"] == server["mode"]
                return health
        except OSError:
            pass
        time.sleep(.2)
    raise RuntimeError("server readiness timed out")


def stop(server):
    if server["process"].poll() is None:
        subprocess.run(["docker", "stop", "--timeout", "30", server["name"]], check=True, capture_output=True)
    code = server["process"].wait(timeout=40)
    server["log"].close()
    if code:
        raise RuntimeError("server did not shut down cleanly: " + server["tag"])


def validate_ledger(out, server, state):
    rows = []
    for line in (out / (server["tag"] + "-server.log")).read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "slot_released" in row:
            rows.append(row)
    assert len(rows) == len({row["id"] for row in rows}) == state["accepted"]
    assert sum(row["slot_released"] for row in rows) == state["slot_releases"]
    assert state["active"] == 0 and all(not task["queued"] for task in state["tasks"].values())
    return {"terminal_records": len(rows), "state": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=ROOT / "benchmarks/kvharbor_sampling.jsonl")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "scope": "HTTP/SSE, four balanced rounds, 16 measured requests per mode",
              "order": [["legacy", "direct"], ["direct", "legacy"], ["legacy", "direct"], ["direct", "legacy"]],
              "workload_sha256": digest(args.workload), "blocks": [], "status": "running"}
    active = None
    try:
        for round_id, modes in enumerate(report["order"]):
            for position, mode in enumerate(modes):
                tag = f"r{round_id}-{position}-{mode}"
                active = start(model, out, mode, tag)
                health = ready(active)
                for phase in ("warmup", "measured"):
                    target = out / (tag + "-" + phase + ".json")
                    command = ["python3", "scripts/stream_bench.py", "--base-url", active["base"],
                               "--workload", str(args.workload), "--requests", "4", "--concurrency", "1", "--output", str(target)]
                    with (out / (tag + "-client.log")).open("a") as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                measured = json.loads(target.read_text())
                assert measured["summary"]["outcomes"]["completed"] == 4
                state = get_json(active["base"], "/state")
                stop(active)
                ledger = validate_ledger(out, active, state)
                report["blocks"].append({"round": round_id, "position": position, "mode": mode, "health": health,
                                         "summary": measured["summary"], "requests": measured["requests"], "ledger": ledger,
                                         "server_command": active["command"]})
                active = None
                atomic_json(out / "report.json", report)
                print(json.dumps({"round": round_id, "mode": mode, "completed": 4, "summary": measured["summary"]}), flush=True)
        grouped = {}
        for block in report["blocks"]:
            grouped.setdefault(block["round"], {})[block["mode"]] = block["requests"]
        for modes in grouped.values():
            assert [row["output_sha256"] for row in modes["legacy"]] == [row["output_sha256"] for row in modes["direct"]]
        report["verified_matching_http_outputs"] = 16
        active = start(model, out, "backend", "backend-lifecycle")
        report["backend_health"] = ready(active)
        with (out / "backend-smoke.log").open("w") as log:
            subprocess.run(["python3", "scripts/llama_smoke.py", "--native", "--base-url", active["base"],
                            "--output", str(out / "backend-smoke.json")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state = get_json(active["base"], "/state")
        stop(active)
        report["backend_lifecycle"] = validate_ledger(out, active, state)
        active = None
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if active is not None:
            stop(active)
        atomic_json(out / "report.json", report)
    print(json.dumps({"status": "completed", "matched_http_outputs": 16, "backend_lifecycle": report["backend_lifecycle"]}), flush=True)


if __name__ == "__main__":
    main()
