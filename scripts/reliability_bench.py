#!/usr/bin/env python3
"""Verify live cancellation, deadlines and optional supervised process recovery."""

import argparse
from datetime import datetime
import json
import re
import subprocess
import time
from pathlib import Path

from capacity_bench import endpoint, idle_state, ledger_delta, request_outcome
from stream_bench import post_stream


def stream(args, body, **kwargs):
    request = {"model": args.model, "stream": True, "temperature": 0,
               "stream_options": {"include_usage": True}, **body}
    return post_stream(args.url, "/v1/chat/completions", request,
                       time.perf_counter_ns(), timeout=args.recovery_seconds, **kwargs)


def inspect_container(name):
    result = subprocess.run(["docker", "inspect", "--", name], check=True,
                            capture_output=True, text=True, timeout=10)
    return json.loads(result.stdout)[0]


def wait_healthy(args, minimum_restart=None):
    begin = time.monotonic()
    while time.monotonic() - begin < args.recovery_seconds:
        try:
            if minimum_restart is not None and inspect_container(args.container)["RestartCount"] < minimum_restart:
                time.sleep(.1)
                continue
            health = endpoint(args, "health")
            if health.get("status") == "ok":
                return health, (time.monotonic() - begin) * 1000
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        time.sleep(.1)
    raise RuntimeError("Service did not recover before the recovery deadline")


def terminal_delta(args, before):
    after, wait_ms = idle_state(args)
    delta = ledger_delta(before, after)
    if not (delta["accepted"] == delta["terminal"] == delta["started"] == delta["slot_releases"] == 1):
        raise RuntimeError("Expected one accepted request, terminal outcome and worker release")
    return delta, wait_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8083")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("llama.cpp", "tensorrt-edge-llm"), default="llama.cpp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--container")
    parser.add_argument("--crash", action="store_true",
                        help="kill the server child of an explicitly labelled acceptance container")
    parser.add_argument("--recovery-seconds", type=float, default=120)
    args = parser.parse_args()
    args.drain_seconds = args.recovery_seconds
    if args.recovery_seconds <= 0:
        parser.error("recovery-seconds must be positive")
    container = None
    if args.crash:
        if not args.container or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
            parser.error("--crash requires an explicit container name")
        container = inspect_container(args.container)
        if (container["Config"].get("Labels", {}).get("kvharbor.role") != "acceptance"
                or not container["HostConfig"].get("Init")
                or container["HostConfig"]["RestartPolicy"]["Name"] != "on-failure"):
            parser.error("Crash tests require native_serving.sh supervise and its acceptance label")
    args.output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "scope": "real-http-sse", "model_id": args.model}
    begin = time.monotonic()
    long_body = {"prompt": "Write a numbered list of 100 detailed facts about water. Continue the list.",
                 "max_tokens": 512}
    followup = {"prompt": "The sum of 17 and 25 is", "max_tokens": 24}
    try:
        health, _ = wait_healthy(args)
        actual = health.get("runtimes", {}).get("llm", {}).get("backend")
        if actual != args.backend:
            raise RuntimeError(f"Expected {args.backend} in /health, received {actual!r}")
        report["health"] = health
        warmup = stream(args, followup)
        if request_outcome(warmup) != "completed":
            raise RuntimeError("Warm-up did not finish with [DONE]")
        idle_state(args)

        before, _ = idle_state(args)
        cancelled = stream(args, long_body, cancel_after=1)
        delta, wait_ms = terminal_delta(args, before)
        if (cancelled["outcome"] != "cancelled"
                or delta["outcomes"].get("client_disconnected") != 1):
            raise RuntimeError("Live disconnect was not recorded exactly once")
        report["disconnect"] = {"client": cancelled, "ledger": delta, "time_to_idle_ms": wait_ms}

        before, _ = idle_state(args)
        timed = stream(args, long_body, deadline_ms=50)
        delta, wait_ms = terminal_delta(args, before)
        if timed["done"] or delta["outcomes"].get("deadline") != 1:
            raise RuntimeError("Running deadline was not recorded exactly once")
        report["deadline"] = {"client": timed, "client_outcome": request_outcome(timed),
                              "ledger": delta, "time_to_idle_ms": wait_ms}
        before, _ = idle_state(args)
        recovered = stream(args, followup)
        if request_outcome(recovered) != "completed":
            raise RuntimeError("Request after cancellation and deadline did not recover")
        delta, wait_ms = terminal_delta(args, before)
        report["followup"] = {"client": recovered, "ledger": delta, "time_to_idle_ms": wait_ms}

        if args.crash:
            # A foreground Docker CLI exiting is not proof of automatic recovery.
            # Kill the labelled container's server child and observe RestartCount.
            # --init keeps the server out of PID 1's special signal handling.
            restart_before = inspect_container(args.container)["RestartCount"]
            started_at = inspect_container(args.container)["State"]["StartedAt"]
            uptime = time.time() - datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
            time.sleep(max(0, 10 - uptime))
            injected = []

            def crash_after_content(_):
                if injected:
                    return
                injected.append(time.monotonic())
                script = ('pid=$(cat /tmp/kvharbor-server.pid); '
                          'test "$pid" -gt 1 && '
                          'test "$(readlink /proc/$pid/exe)" = /native/kvharbor-server && '
                          'kill -KILL "$pid"')
                subprocess.run(["docker", "exec", "--", args.container, "sh", "-c", script],
                               check=True, capture_output=True, timeout=10)

            interrupted = stream(args, long_body, on_content=crash_after_content)
            if not injected or interrupted["done"]:
                raise RuntimeError("Crash did not interrupt an active SSE request")
            new_health, _ = wait_healthy(args, minimum_restart=restart_before + 1)
            ready_ms = (time.monotonic() - injected[0]) * 1000
            recovered = stream(args, followup)
            if request_outcome(recovered) != "completed":
                raise RuntimeError("Post-restart inference did not complete")
            state, _ = idle_state(args)
            report["crash_recovery"] = {
                "client": interrupted, "restart_count_before": restart_before,
                "restart_count_after": inspect_container(args.container)["RestartCount"],
                "crash_to_health_ms": ready_ms, "post_restart_request": recovered,
                "post_restart_state": state, "health": new_health,
                "interrupted_request_replayed": False,
            }
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - begin
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
