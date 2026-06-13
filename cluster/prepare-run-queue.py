#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate runs.jsonl and write one run_id per line for HTCondor."
    )
    parser.add_argument("--runs-jsonl", required=True, help="Path to runs.jsonl.")
    parser.add_argument("--output", required=True, help="Output queue file path.")
    return parser


def load_run_ids(path: Path) -> list[str]:
    run_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: expected a JSON object.")
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"line {line_number}: missing non-empty `run_id`.")
            if run_id in seen:
                raise ValueError(f"line {line_number}: duplicate run_id {run_id!r}.")
            seen.add(run_id)
            run_ids.append(run_id)
    return run_ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_ids = load_run_ids(Path(args.runs_jsonl))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(run_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(run_ids)} run_ids to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
