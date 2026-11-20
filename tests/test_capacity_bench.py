"""Regression checks for failed streams, SLO queueing and worker accounting."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.capacity_bench import fingerprints, ledger_state, project, request_outcome


class CapacityTests(unittest.TestCase):
    def test_edge_engine_fingerprint_requires_real_engine_and_matching_health(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(scope="real-http-sse", model="qwen", expected_backend="tensorrt-edge-llm",
                                   engine_dir=Path(directory), server_binary=None, model_file=None)
            runtime = {"backend": "llama.cpp", "model": "qwen", "sdk_version": "0.12.0"}
            health = {"runtimes": {"llm": runtime}}
            with self.assertRaisesRegex(RuntimeError, "differs"):
                fingerprints(args, health)
            runtime["backend"] = args.expected_backend
            with self.assertRaisesRegex(RuntimeError, "No serialized engine"):
                fingerprints(args, health)
            (args.engine_dir / "llm.engine").write_bytes(b"fixture")
            report = fingerprints(args, health)
            self.assertEqual(report["runtime_lock"]["backend"], args.expected_backend)
            self.assertIn("llm.engine", report["engine_sha256"])

    def test_client_timeout_after_http_headers(self):
        self.assertEqual(request_outcome({"http_status": 200, "outcome": "timed_out"}), "timeout")

    def test_incomplete_http_200_is_not_success_or_inferred_timeout(self):
        self.assertEqual(request_outcome({"http_status": 200, "done": False}), "incomplete_sse")
        self.assertEqual(request_outcome({"http_status": 429}), "rejected")

    def test_client_queue_counts_towards_slo(self):
        row = {"index": 0, "http_status": 200, "done": True, "outcome": "completed",
               "usage": {"completion_tokens": 2}, "client_queue_ms": 1950,
               "ttft_ms": 100, "tpot_ms": 20, "dispatch_ns": 1, "e2e_ms": 120}
        result = project({"requests": [row], "summary": {"measurement_s": 1}},
                         SimpleNamespace(ttft_ms=2000, tpot_ms=75))
        self.assertEqual(result["slo_compliant"], 0)
        self.assertEqual(result["requests"][0]["arrival_ttft_ms"], 2050)

    def test_missing_worker_release_is_detected(self):
        state = {"active": 0, "accepted": 2, "started": 2, "rejected": 0,
                 "slot_releases": 1, "outcomes": {"completed": 2}, "tasks": {"llm": {"queued": 0}}}
        with self.assertRaisesRegex(RuntimeError, "started workers"):
            ledger_state(state)

    def test_queued_deadline_never_acquired_worker(self):
        state = {"active": 0, "accepted": 3, "started": 2, "rejected": 1,
                 "slot_releases": 2, "outcomes": {"completed": 2, "deadline": 1},
                 "tasks": {"llm": {"queued": 0}}}
        self.assertEqual(ledger_state(state)["terminal"], 3)

    def test_pending_requests_must_balance(self):
        state = {"active": 1, "accepted": 4, "started": 2, "rejected": 0,
                 "slot_releases": 1, "outcomes": {"completed": 1},
                 "tasks": {"llm": {"queued": 1}}}
        with self.assertRaisesRegex(RuntimeError, "accepted requests"):
            ledger_state(state)


if __name__ == "__main__":
    unittest.main()
