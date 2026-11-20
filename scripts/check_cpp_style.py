#!/usr/bin/env python3
"""Check Google formatting and report strict C++ migration blockers.

The strict screen is a failing gate for known disallowed constructs. Passing
the screen is not a substitute for compiler-backed checks and code review.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".inc"}
EXTERNAL = {"third_party", "vendor", ".cache", "build", "results"}


def source_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True)
    paths = []
    for name in sorted(set(result.stdout.decode().split("\0")) - {""}):
        path = Path(name)
        if path.suffix not in EXTENSIONS or path.parts[0] in EXTERNAL:
            continue
        absolute = ROOT / path
        try:
            absolute.lstat()
        except FileNotFoundError:
            # A working-tree rename can leave the old path in the index.
            continue
        if not absolute.is_file() or not absolute.resolve().is_relative_to(ROOT):
            raise ValueError("Expected an owned source file: " + name)
        paths.append(path)
    return paths


def code_only(text):
    """Blank comments and string literals while retaining source line numbers."""
    output = list(text)
    index = 0
    while index < len(text):
        stop = None
        if text.startswith("//", index):
            stop = text.find("\n", index)
            if stop == -1:
                stop = len(text)
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            stop = len(text) if end == -1 else end + 2
        else:
            raw = re.match(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(', text[index:index + 24])
            if raw:
                end = text.find(")" + raw.group(1) + '"', index + raw.end())
                stop = len(text) if end == -1 else end + len(raw.group(1)) + 2
            elif text[index] in ('"', "'"):
                if text[index] == "'" and index + 1 < len(text):
                    start = index - 1
                    while start >= 0 and (text[start].isalnum() or text[start] in "_'."):
                        start -= 1
                    prefix = text[start + 1:index]
                    if prefix and prefix[0].isdigit() and text[index + 1].isalnum():
                        index += 1
                        continue
                quote, end = text[index], index + 1
                while end < len(text):
                    if text[end] == "\\":
                        end += 2
                    elif text[end] == quote:
                        end += 1
                        break
                    else:
                        end += 1
                stop = end
        if stop is not None:
            for offset in range(index, min(stop, len(text))):
                if output[offset] != "\n":
                    output[offset] = " "
            index = stop
        else:
            index += 1
    return "".join(output)


def screen(paths):
    rules = {
        "exceptions": (
            r"\b(?:throw|catch)\b|\bstd::(?:exception_ptr|nested_exception|"
            r"current_exception|make_exception_ptr|rethrow_exception|"
            r"rethrow_if_nested|throw_with_nested)\b"
        ),
        "filesystem": r"\bstd::filesystem\b",
        "disallowed_standard_library": r"\bstd::ratio(?:_[a-z_]+)?\b",
        "pragma_once": r"#\s*pragma\s+once\b",
        "using_directive": r"\busing\s+namespace\b",
        "rtti": r"\b(?:dynamic_cast|typeid)\b",
    }
    disallowed_headers = {
        "filesystem": "filesystem",
        "ratio": "disallowed_standard_library",
        "cfenv": "disallowed_standard_library",
        "fenv.h": "disallowed_standard_library",
    }
    include_directive = re.compile(r"(?m)^[ \t]*#[ \t]*include\b")
    header_operand = re.compile(
        r'[ \t]*(?:/\*.*?\*/[ \t]*)*(?:<([^>\n]+)>|"([^"\n]+)")'
    )
    findings = []
    for path in paths:
        if path.suffix in {".cpp", ".hpp", ".cuh"}:
            findings.append({"file": str(path), "line": 1, "rule": "file_extension"})
        original = (ROOT / path).read_text()
        text = code_only(original)
        if path.suffix in {".h", ".hpp", ".cuh"}:
            guard = "KVHARBOR_" + re.sub(r"[^A-Za-z0-9]", "_", str(path.with_suffix(""))).upper() + "_H_"
            beginning = r"^\s*#ifndef\s+" + guard + r"\s*\n#define\s+" + guard + r"\s*\n"
            if not re.search(beginning, text) or not re.search(r"#endif\s*$", text):
                findings.append({"file": str(path), "line": 1, "rule": "header_guard"})
        for rule, pattern in rules.items():
            for match in re.finditer(pattern, text):
                # The vendor API throws. Only its boundary may catch and translate
                # an SDK error into the service's Failure type.
                if (rule == "exceptions" and match.group() == "catch"
                        and path == Path("serving/src/edge_sdk_boundary.h")):
                    continue
                findings.append({"file": str(path), "line": text.count("\n", 0, match.start()) + 1,
                                 "rule": rule})
        for directive in include_directive.finditer(text):
            operand = header_operand.match(original, directive.end())
            if operand is None:
                continue
            rule = disallowed_headers.get(operand[1] or operand[2])
            if rule is not None:
                findings.append({"file": str(path), "line": text.count("\n", 0, directive.start()) + 1,
                                 "rule": rule})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang-format", default="clang-format-18")
    parser.add_argument("--format-only", action="store_true",
                        help="check layout only; does not establish full Google conformance")
    parser.add_argument("--fix-format", action="store_true",
                        help="apply Google formatting before checking")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tool = shutil.which(args.clang_format)
    if not tool:
        parser.error("clang-format-18 is required; set --clang-format to its path")
    version = subprocess.check_output([tool, "--version"], text=True).strip()
    if not re.search(r"version 18\.", version):
        parser.error("Use clang-format major version 18 for reproducible formatting")
    paths = source_files()
    if args.fix_format:
        subprocess.run([tool, "--style=file", "-i", *map(str, paths)], cwd=ROOT, check=True)
    formatted = subprocess.run([tool, "--style=file", "--dry-run", "--Werror", *map(str, paths)],
                               cwd=ROOT, capture_output=True, text=True)
    findings = screen(paths)
    report = {"schema_version": 1, "formatter": version, "owned_cpp_files": len(paths),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "format_config_sha256": hashlib.sha256((ROOT / ".clang-format").read_bytes()).hexdigest(),
              "format_ok": formatted.returncode == 0,
              "strict_screen_ok": not findings,
              "full_guide_compliance_established": False,
              "counts": dict(Counter(item["rule"] for item in findings)), "findings": findings,
              "requires_further_review": ["identifier naming", "self-contained headers and includes",
                                          "language and API rules", "CUDA portability boundaries"]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, indent=2))
    if formatted.returncode:
        print(formatted.stderr[:12000], file=sys.stderr)
    if not args.format_only and findings:
        print("Strict migration is incomplete. See the report; no legacy suppression is applied.", file=sys.stderr)
    return 1 if formatted.returncode or (not args.format_only and findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
