#!/usr/bin/env bash
set -Eeuo pipefail

CID="" ; PID=""
FWD_ARGS=()
while [[ $# -gt 0 ]]; do
        case "$1" in
                --cluster) CID="$2"; shift 2 ;;
                --proc) PID="$2"; shift 2 ;;
                *) FWD_ARGS+=("$1"); shift ;;
        esac
done
if [[ -z "${CID}" || -z "${PID}" ]]; then
        echo "Missing --cluster / --proc" >&2; exit 2
fi

PROJECT_DIR="$PWD"


# SCRATCH_BASE="/scratch/physik/felixmilan.maurer"
# SCRATCH_DIR="${SCRATCH_BASE}/test/${CID}.${PID}"
SCRATCH_DIR="${_CONDOR_SCRATCH_DIR}/redp.${CID}.${PID}"
mkdir -p "${SCRATCH_DIR}"

RESULTS_DIR="${PROJECT_DIR}/results/${CID}.${PID}"
mkdir -p "${RESULTS_DIR}"

echo "--> CID=${CID} PID=${PID}"
echo "--> SCRATCH=${SCRATCH_DIR}"
echo "--> RESULTS=${RESULTS_DIR}"

SIM_BIN="/bin/red-patterns"
if [[ ! -x "${SIM_BIN}" ]]; then
        echo "Binary not found at ${SIM_BIN}" >&2; exit 3
fi

OUT_DIR="${SCRATCH_DIR}"
echo "--> Running ${SIM_BIN} ${FWD_ARGS[*]} --out-dir=${OUT_DIR}"
"${SIM_BIN}" "${FWD_ARGS[@]}" --out-dir="${OUT_DIR}"

cp -v "${OUT_DIR}/run.h5" "${RESULTS_DIR}/"
echo "launched at $(date -Is): ${SIM_BIN} ${FWD_ARGS[*]} --out-dir=${OUT_DIR}" > "${RESULTS_DIR}/run_manifest.txt"

echo "--> Done!"

