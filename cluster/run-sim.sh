#!/usr/bin/env bash
set -Eeuo pipefail

JOB_ID="${JOB_ID:-}"
if [[ -z "${JOB_ID}" ]]; then
	echo "Missing JOB_ID environment variable... Should be set in sweep.submit with environment = ..."
	exit 2
fi

SWEEP_DIR="$PWD"

RESULTS_DIR="${SWEEP_DIR}/results/${JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# SCRATCH_BASE="/scratch/physik/felixmilan.maurer"
SCRATCH_BASE="${_CONDOR_SCRATCH_DIR}"
SCRATCH_DIR="${SCRATCH_BASE}/redp.${JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

SIM_BIN="/bin/red-patterns"
if [[ ! -x "${SIM_BIN}" ]]; then
        echo "Binary not found at ${SIM_BIN}" >&2
	exit 3
fi

OUT_DIR="${SCRATCH_DIR}"

START_TS="$(date -Is)"
MANIFEST="${RESULTS_DIR}/run_manifest.txt"

{
  echo "job_id=${JOB_ID}"
  echo "started_at=${START_TS}"
  echo "sweep_dir=${SWEEP_DIR}"
  echo "scratch_dir=${SCRATCH_DIR}"
  echo "results_dir=${RESULTS_DIR}"
  echo "sim_bin=${SIM_BIN}"
  echo "run_tag=${RUN_TAG-}"
  echo "commit_hash=${COMMIT_HASH-}"
  echo -n "cmd="; printf '%q ' "${SIM_BIN}" "$@" --out-dir="${OUT_DIR}"
  echo ""
} > "${MANIFEST}"

START_S="$(date +%s)"

# Still call the finish function even if something went wrong
finish() {
	ec=$?

	END_S="$(date +%s)"
	ELAPSED_S=$((END_S - START_S))

	# Get Runtime in H:MM:SS or MM:SS
	if (( ELAPSED_S >= 3600 )); then
    		RUNTIME="$(printf '%d:%02d:%02d' $((ELAPSED_S/3600)) $(((ELAPSED_S%3600)/60)) $((ELAPSED_S%60)))"
  	else
    		RUNTIME="$(printf '%02d:%02d' $((ELAPSED_S/60)) $((ELAPSED_S%60)))"
  	fi

	{
		echo "finished_at=$(date -Is)"
		echo "runtime=${RUNTIME}"
		echo "exit_code=${ec}"
	} >> "${MANIFEST}"
	exit "${ec}"
}
trap finish EXIT

echo "--> JOB_ID=${JOB_ID}"
echo "--> SCRATCH=${SCRATCH_DIR}"
echo "--> RESULTS=${RESULTS_DIR}"
echo "--> Running ${SIM_BIN} ${FWD_ARGS[*]} --out-dir=${SCRATCH_DIR}"

"${SIM_BIN}" "$@" --out-dir="${OUT_DIR}"

if [[ ! -f "${SCRATCH_DIR}/run.h5" ]]; then
  echo "ERROR: expected output not found: ${SCRATCH_DIR}/run.h5" >&2
  exit 10
fi

tmp="${RESULTS_DIR}/run.h5.tmp"
cp -f "${SCRATCH_DIR}/run.h5" "${tmp}"
mv -f "${tmp}" "${RESULTS_DIR}/run.h5"

echo "--> Done!"
