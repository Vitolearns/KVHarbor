#!/usr/bin/env python3
"""Measure the pinned model's upstream demo recordings, separate from TTS round trips."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
import urllib.error
import wave

from audio_bench import asr, edit_distance, wait_idle
from stream_bench import get_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ("f1_noise.wav", "noise2.wav", "cantonese.wav", "codeswitch.wav", "de.wav", "ja1.wav")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def character_error(reference, hypothesis):
    def chars(text):
        return [c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()]
    ref, hyp = chars(reference), chars(hypothesis)
    if not ref:
        raise ValueError("reference has no scored characters")
    errors = edit_distance(ref, hyp)
    return {"errors": errors, "reference_characters": len(ref), "cer": errors / len(ref)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/audio/qwen3-asr-0.6b-int8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    lock = json.loads((ROOT / "configs/runtime/audio.json").read_text())["asr"]
    marker = json.loads((args.model_dir / ".kvharbor-asset.json").read_text())
    if marker["sha256"] != lock["sha256"]:
        parser.error("ASR model install does not match the pinned archive")
    if len(set(args.samples)) != len(args.samples):
        parser.error("sample names must be unique")
    directory = args.model_dir / "test_wavs"
    transcript = directory / "transcript.txt"
    references = dict(line.split(maxsplit=1) for line in transcript.read_text().splitlines() if line.strip())
    inputs = []
    for name in args.samples:
        if Path(name).name != name or not name.endswith(".wav") or name not in references:
            parser.error("expected a named WAV and reference from the pinned model package")
        path = directory / name
        with wave.open(str(path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE" or not 0 < duration <= 30:
                parser.error("demo fixture must be PCM16 mono and at most 30 seconds")
            if path.stat().st_size > 2 * 1024 * 1024:
                parser.error("demo fixture exceeds the HTTP body limit")
        inputs.append((name, path, duration))
    health = get_json(base, "/health")
    if health["runtimes"]["asr"]["model"] != lock["id"]:
        parser.error("running ASR model does not match the lock file")
    report = {
        "schema_version": 1,
        "scope": "upstream demonstration recordings; small functional sample, not held-out accuracy evaluation",
        "reference_source": "test_wavs/transcript.txt from the pinned ASR archive",
        "archive_sha256": lock["sha256"], "transcript_sha256": sha256(transcript),
        "normalization": "Unicode NFKC, casefold, retain alphanumeric characters; micro CER across clips",
        "health": health, "initial_state": get_json(base, "/state"), "samples": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        partial = args.output.with_suffix(args.output.suffix + ".part")
        partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        partial.replace(args.output)

    save()
    # Keep one explicit warm-up outside the reported per-clip latency measurements.
    try:
        report["warmup"] = asr(base, inputs[0][1])
        for name, path, duration in inputs:
            row = {"fixture": name, "wav_sha256": sha256(path), "audio_seconds": duration,
                   "reference": references[name]}
            try:
                row["result"] = asr(base, path)
                row["quality"] = character_error(row["reference"], row["result"]["text"])
                row["outcome"] = "completed"
            except (OSError, ValueError, urllib.error.URLError) as error:
                row["outcome"] = "failed"
                row["error"] = type(error).__name__ + ": " + str(error)
            report["samples"].append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["final_state"] = wait_idle(base, "asr")
        complete = [row for row in report["samples"] if row["outcome"] == "completed"]
        errors = sum(row["quality"]["errors"] for row in complete)
        characters = sum(row["quality"]["reference_characters"] for row in complete)
        report["summary"] = {"completed": len(complete), "failed": len(inputs) - len(complete),
                             "errors": errors, "reference_characters": characters,
                             "micro_cer": errors / characters if characters else None}
        report["status"] = "completed" if len(complete) == len(inputs) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        save()
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
