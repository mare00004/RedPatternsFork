#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USAGE="usage: submit-sweep.sh SWEEP_DIR IMAGE_TAG"

RUN_DIR="${1:?$USAGE}"
COMMIT_HASH="${2:?$USAGE}"
if [[ ! -d "${RUN_DIR}" ]]; then
	echo "${RUN_DIR} does not exist!" >&2
	exit 1
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd -P)"

RUNS_JSONL="${RUN_DIR}/runs.jsonl"
RUN_IDS_FILE="${RUN_DIR}/run_ids.txt"

if [[ ! -f "${RUNS_JSONL}" ]]; then
	echo "Missing ${RUNS_JSONL}" >&2
	exit 1
fi

if [[ ! -f "${RUN_IDS_FILE}" ]]; then
	echo "Missing ${RUN_IDS_FILE}" >&2
	exit 1
fi

RUN_TAG="$(basename "${RUN_DIR}")"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results"
while IFS= read -r run_id; do
	[[ -n "${run_id}" ]] || continue
	mkdir -p "${RUN_DIR}/results/${run_id}"
done < "${RUN_IDS_FILE}"

condor_submit "${SCRIPT_DIR}/sweep.submit" \
	RUN_TAG="${RUN_TAG}" \
	RUN_DIR="${RUN_DIR}" \
	RUNS_JSONL="${RUNS_JSONL}" \
	RUN_IDS_FILE="${RUN_IDS_FILE}" \
	LAUNCH_SH="${SCRIPT_DIR}/launch.sh" \
	COMMIT_HASH="${COMMIT_HASH}"
