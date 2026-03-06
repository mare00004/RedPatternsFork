#!/usr/bin/env bash

echo "Submitting 'Notebook Server' Job..."
condor_submit marimo.submit &> /dev/null

# Get ClusterID and ProcessID of last submitted job
CLUSTER_SPACE_PROC="$(condor_q -limit 1 -nobatch -af ClusterID ProcID)"
JOBID="${CLUSTER_SPACE_PROC// /.}"

echo "Job submitted with JobID=$JOBID"

# Wait till job is running
while true; do
	STATUS="$(condor_q -limit 1 -nobatch -af JobStatus)"
	if [[ -z "$STATUS" ]]; then
		sleep 1
		continue
	fi

	case "$STATUS" in
		1) echo "Job is idle..."; sleep 1; continue;;
		2) echo "Job is running"; break;;
		*) echo "ALARM!!!"; exit 1;;
	esac
done

# Start SSH Port forwarding
EXECUTION_PORT="8080"
IPADDR="127.0.0.1" # Defined in `run.sh`
SUBMIT_PORT="3718" # On which port the notebook should be available on the submission node

sleep 5
echo "Starting SSH Port Forwarding..."

condor_ssh_to_job -auto-retry -ssh "ssh -N -o ExitOnForwardFailure=yes -L ${SUBMIT_PORT}:${IPADDR}:${EXECUTION_PORT}" $JOBID
