"""Validate audio input rejection and lease accounting against a running native service."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audio_bench import asr
from stream_bench import get_json


def wav_bytes(channels=1, rate=16000, seconds=.1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(int(rate * seconds) * channels * 2))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    initial = get_json(base, "/state")
    assert initial["active"] == 0 and all(not t["queued"] for t in initial["tasks"].values()), initial
    report = {"scope": "native audio HTTP validation; no model quality claim",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "initial_state": initial, "cases": []}
    cases = [
        ("malformed-wav", "/v1/audio/transcriptions", b"invalid WAV", "audio/wav"),
        ("stereo-wav", "/v1/audio/transcriptions", wav_bytes(channels=2), "audio/wav"),
        ("unsupported-rate", "/v1/audio/transcriptions", wav_bytes(rate=4000), "audio/wav"),
        ("over-30-seconds", "/v1/audio/transcriptions", wav_bytes(seconds=30.01), "audio/wav"),
        ("invalid-mime", "/v1/audio/transcriptions", b"invalid WAV", "application/json"),
        ("invalid-tts-speed", "/v1/audio/speech", b'{"input":"Hello.","voice":2,"speed":0.01}', "application/json"),
    ]
    for name, endpoint, body, content_type in cases:
        request = urllib.request.Request(base + endpoint, data=body, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
            error.read()
        assert status == 400, (name, status)
        report["cases"].append({"id": name, "expected_status": 400, "status": status,
                                "body_sha256": hashlib.sha256(body).hexdigest()})
    # The parser must reject an oversized declared body before waiting for its bytes.
    url = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    try:
        connection.putrequest("POST", "/v1/audio/transcriptions")
        connection.putheader("Content-Type", "audio/wav")
        connection.putheader("Content-Length", str(2 * 1024 * 1024 + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413, response.status
        response.read()
        report["cases"].append({"id": "oversized-declared-body", "declared_bytes": 2 * 1024 * 1024 + 1,
                                "expected_status": 413, "status": response.status})
    finally:
        connection.close()
    report["recovery_probe"] = asr(base, args.sample)
    final = get_json(base, "/state")
    accepted = final["accepted"] - initial["accepted"]
    released = final["slot_releases"] - initial["slot_releases"]
    outcomes = sum(final["outcomes"].values()) - sum(initial["outcomes"].values())
    assert final["active"] == 0 and all(not t["queued"] for t in final["tasks"].values()), final
    assert accepted == released == outcomes, (accepted, released, outcomes)
    report.update(status="ok", final_state=final,
                  ledger_delta={"accepted": accepted, "slot_releases": released, "terminal_outcomes": outcomes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "ok", "cases": len(report["cases"]), "ledger_delta": report["ledger_delta"]}))


if __name__ == "__main__":
    main()
