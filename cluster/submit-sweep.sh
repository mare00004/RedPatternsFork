#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USAGE="usage: submit-sweep.sh PARAMS_FILE [COMMIT_HASH] [RUN_DIR] \n\n\tExpects PARAMS_FILE to have a unique name like meeting_<date>.txt"

PARAMS_FILE="${1:?$USAGE}"
COMMIT_HASH="${2:-1935885}"
RUN_DIR=$3

if [[ ! -f "$PARAMS_FILE" ]]; then
	echo "$PARAMS_FILE does not exist!"
	exit 1
fi

PARAMS_FILE_NAME=$(basename "$PARAMS_FILE" .txt)

if [[ -z "$RUN_DIR" ]]; then
	RUN_DIR="${SCRIPT_DIR}/${PARAMS_FILE_NAME}"
fi

mkdir -p "$RUN_DIR/logs"

condor_submit "${SCRIPT_DIR}/sweep.submit" \
	RUN_TAG="${PARAMS_FILE_NAME}" \
	RUN_DIR="${RUN_DIR}" \
	PARAMS_FILE="${PARAMS_FILE}" \
	COMMIT_HASH="${COMMIT_HASH}"
