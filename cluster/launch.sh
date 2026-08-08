#!/usr/bin/env bash
set -Eeuo pipefail

exec /opt/red-patterns/.venv/bin/python /opt/red-patterns/sweep/run_one.py --runs-jsonl runs.jsonl --run-id "$1"
