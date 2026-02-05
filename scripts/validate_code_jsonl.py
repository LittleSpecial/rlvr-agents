#!/usr/bin/env python3
"""
Validate CodeEnv JSONL dataset files (schema + basic python syntax checks).

Usage:
  python3 scripts/validate_code_jsonl.py datasets/code/mbpp_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_FIELDS = ("task_id", "prompt")
OPTIONAL_FIELDS = ("initial_code", "test_code", "expected_output", "language", "timeout", "metadata")


def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _validate_record(rec: Dict[str, Any], *, path: Path, line_no: int) -> List[str]:
    errors: List[str] = []
    for k in REQUIRED_FIELDS:
        if k not in rec:
            errors.append(f"missing required field: {k}")
    for k in rec.keys():
        if k not in REQUIRED_FIELDS and k not in OPTIONAL_FIELDS:
            # allow extra fields but warn
            pass

    if "task_id" in rec and not isinstance(rec["task_id"], str):
        errors.append("task_id must be str")
    if "prompt" in rec and not (isinstance(rec["prompt"], str) and rec["prompt"].strip()):
        errors.append("prompt must be non-empty str")

    lang = rec.get("language", "python")
    if not isinstance(lang, str):
        errors.append("language must be str")
        lang = "python"

    if "timeout" in rec and not isinstance(rec["timeout"], (int, float)):
        errors.append("timeout must be number")

    if "metadata" in rec and rec["metadata"] is not None and not _is_dict(rec["metadata"]):
        errors.append("metadata must be dict")

    # Syntax check for python tests (if provided)
    if lang == "python":
        test_code = rec.get("test_code")
        if test_code is not None and not isinstance(test_code, str):
            errors.append("test_code must be str when provided")
        if isinstance(test_code, str) and test_code.strip():
            try:
                compile(test_code, filename=f"{path}:{line_no}", mode="exec")
            except SyntaxError as e:
                errors.append(f"test_code syntax error: {e.msg} (line {e.lineno})")

        initial_code = rec.get("initial_code")
        if initial_code is not None and not isinstance(initial_code, str):
            errors.append("initial_code must be str when provided")
        if isinstance(initial_code, str) and initial_code.strip():
            try:
                compile(initial_code, filename=f"{path}:{line_no}:initial_code", mode="exec")
            except SyntaxError as e:
                errors.append(f"initial_code syntax error: {e.msg} (line {e.lineno})")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--max_errors", type=int, default=50)
    args = parser.parse_args()

    path = args.jsonl
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    seen_ids = set()
    total = 0
    total_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                total_errors += 1
                print(f"[ERROR] {path}:{line_no} invalid json: {e}")
                if total_errors >= args.max_errors:
                    raise SystemExit(1)
                continue

            if not isinstance(rec, dict):
                total_errors += 1
                print(f"[ERROR] {path}:{line_no} record must be an object")
                if total_errors >= args.max_errors:
                    raise SystemExit(1)
                continue

            tid = rec.get("task_id")
            if isinstance(tid, str):
                if tid in seen_ids:
                    total_errors += 1
                    print(f"[ERROR] {path}:{line_no} duplicated task_id: {tid}")
                else:
                    seen_ids.add(tid)

            errs = _validate_record(rec, path=path, line_no=line_no)
            if errs:
                total_errors += len(errs)
                for err in errs:
                    print(f"[ERROR] {path}:{line_no} {err}")
                if total_errors >= args.max_errors:
                    raise SystemExit(1)

    if total_errors:
        print(f"Found {total_errors} errors in {total} records: {path}")
        raise SystemExit(1)
    print(f"OK: {total} records validated: {path}")


if __name__ == "__main__":
    main()

