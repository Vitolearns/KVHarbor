#!/usr/bin/env python3
"""Archive the exact serving sources identified by a measurement manifest."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(manifest, output, root, fallback=None):
    expected = manifest["measured_source_sha256"]
    files, excluded = {}, []
    for name, wanted in sorted(expected.items()):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source names must be relative to the project")
        if relative.suffix == ".pyc" or "__pycache__" in relative.parts:
            excluded.append(name)
            continue
        candidates = [root / name] + ([fallback / name] if fallback else [])
        for candidate in candidates:
            if candidate.is_file():
                content = candidate.read_bytes()
                if digest(content) == wanted:
                    files[name] = content
                    break
        else:
            raise ValueError("No retained source matches the measured digest: " + name)
    for name in ("configs/runtime/llama.json", "benchmarks/kvharbor_text_mixed.jsonl"):
        files.setdefault(name, (root / name).read_bytes())
    snapshot = {
        "schema_version": 1, "server_binary_sha256": manifest["server_binary_sha256"],
        "model_sha256": manifest["model_sha256"],
        "files_sha256": {name: digest(content) for name, content in files.items()},
        "generated_files_excluded": excluded,
    }
    files["snapshot.json"] = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, content in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(content), 0o644, 0
                    archive.addfile(info, io.BytesIO(content))
    return {"path": str(output), "sha256": digest(output.read_bytes()),
            "source_and_fixture_files": len(files) - 1,
            "generated_files_excluded": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--fallback-root", type=Path)
    args = parser.parse_args()
    result = freeze(json.loads(args.manifest.read_text()), args.output,
                    args.source_root, args.fallback_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
