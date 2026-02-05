#!/usr/bin/env python3
"""
Export common code benchmarks from HuggingFace `datasets` into this repo's CodeEnv JSONL format.

Usage examples:
  python3 scripts/prepare_code_dataset_hf.py --dataset mbpp --split train --out datasets/code/mbpp_train.jsonl
  python3 scripts/prepare_code_dataset_hf.py --dataset openai_humaneval --split test --out datasets/code/humaneval_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _jsonl_write(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mbpp_to_records(ds: Any, *, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break

        task_id = item.get("task_id", i)
        prompt = item.get("text") or item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        initial_code = item.get("starter_code") or item.get("initial_code") or ""

        setup = item.get("test_setup_code") or item.get("test_setup") or ""
        test_list = item.get("test_list") or item.get("tests") or []
        if not isinstance(test_list, list):
            test_list = []
        test_lines = [t for t in test_list if isinstance(t, str)]

        test_code_parts: List[str] = []
        if isinstance(setup, str) and setup.strip():
            test_code_parts.append(setup.rstrip() + "\n")
        if test_lines:
            test_code_parts.append("\n".join(test_lines).rstrip() + "\n")
        test_code = "\n".join(test_code_parts).strip()

        records.append({
            "task_id": f"mbpp/{task_id}",
            "prompt": prompt,
            "initial_code": initial_code,
            "test_code": test_code,
            "language": "python",
            "timeout": 30.0,
            "metadata": {
                "source": "mbpp",
                "split": split,
                "task_type": item.get("metadata", {}).get("task_type") if isinstance(item.get("metadata"), dict) else None,
            },
        })
    return records


def _humaneval_to_records(ds: Any, *, split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break

        task_id = item.get("task_id", f"HumanEval/{i}")
        prompt = item.get("prompt")
        test = item.get("test")
        entry_point = item.get("entry_point")
        if not (isinstance(prompt, str) and isinstance(test, str) and isinstance(entry_point, str)):
            continue

        test_code = test.rstrip() + "\n\n" + (
            "if __name__ == '__main__':\n"
            f"    check(globals()[{entry_point!r}])\n"
        )

        records.append({
            "task_id": str(task_id),
            "prompt": prompt,
            "initial_code": "",
            "test_code": test_code,
            "language": "python",
            "timeout": 30.0,
            "metadata": {"source": "openai_humaneval", "split": split, "task_type": "code_gen"},
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mbpp", "openai_humaneval"])
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=str, default=None, help="Optional HF dataset config name")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: `datasets`.\n"
            "Install with: python3 -m pip install datasets\n"
            f"Original error: {e}"
        )

    if args.dataset == "mbpp":
        ds = load_dataset("mbpp", args.config, split=args.split) if args.config else load_dataset("mbpp", split=args.split)
        records = _mbpp_to_records(ds, split=args.split, max_samples=args.max_samples)
    else:
        ds = load_dataset("openai_humaneval", args.config, split=args.split) if args.config else load_dataset("openai_humaneval", split=args.split)
        records = _humaneval_to_records(ds, split=args.split, max_samples=args.max_samples)

    _jsonl_write(records, args.out)
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()

