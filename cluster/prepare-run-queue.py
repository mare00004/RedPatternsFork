#!/usr/bin/env python3
"""Validate runs.jsonl and write one run_id per line for HTCondor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from red_patterns.sweep_jobs import load_runs_jsonl, run_ids_from_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate runs.jsonl and write one run_id per line for HTCondor."
    )
    parser.add_argument("--runs-jsonl", required=True, help="Path to runs.jsonl.")
    parser.add_argument("--output", required=True, help="Output queue file path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_jsonl = Path(args.runs_jsonl)
    output_path = Path(args.output)
    try:
        runs = load_runs_jsonl(runs_jsonl)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    run_ids = run_ids_from_runs(runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(run_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(run_ids)} run_ids to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
