#!/usr/bin/env bash
set -Eeuo pipefail

exec python3 /opt/red-patterns/sweep/run_one.py --runs-jsonl runs.jsonl --run-id "$1"
