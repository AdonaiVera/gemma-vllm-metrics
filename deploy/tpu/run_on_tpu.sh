#!/usr/bin/env bash
# Drives one benchmark run on the TPU VM from this machine, then pulls the
# results back. Does NOT delete the VM; run teardown.sh yourself afterwards.
# Dry run by default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/config.sh"
parse_execute "${1:-}"
require_target

SSH=(gcloud compute tpus tpu-vm ssh "$NODE_NAME" --project "$PROJECT" --zone "$ZONE")
SCP=(gcloud compute tpus tpu-vm scp --project "$PROJECT" --zone "$ZONE")
TUNNEL_PORT=18000

if [ "$EXECUTE" = "yes" ]; then
  trap 'echo; echo "!! A STEP FAILED. The VM is still running and billing (hard stop after '"$MAX_RUN"').";
        echo "!! Inspect, or delete it now: deploy/tpu/teardown.sh --execute"' ERR
fi

remote() { run "${SSH[@]}" --command "$1"; }

echo "== 1/6 Copy the benchmark code to the VM"
remote "mkdir -p ~/$REMOTE_DIR"
run "${SCP[@]}" --recurse "$ROOT/benchmarks" "$HERE/remote.sh" "$NODE_NAME:~/$REMOTE_DIR/"

echo "== 2/6 Install vLLM-TorchTPU (stop here and report if this fails)"
remote "bash ~/$REMOTE_DIR/remote.sh install"

echo "== 3/6 Start the server and wait until ready (stop here and report if it fails)"
remote "MODEL=$MODEL bash ~/$REMOTE_DIR/remote.sh serve"

echo "== 4/6 Run the 27-run benchmark matrix on the VM"
remote "ENV_LABEL=$ENV_LABEL MODEL=$MODEL bash ~/$REMOTE_DIR/remote.sh bench"

echo "== 5/6 Pull results back"
run "${SCP[@]}" --recurse "$NODE_NAME:~/$REMOTE_DIR/benchmarks/results/$ENV_LABEL" "$ROOT/benchmarks/results/"
run "${SCP[@]}" "$NODE_NAME:~/$REMOTE_DIR/metrics_sample.txt" "$ROOT/benchmarks/results/$ENV_LABEL/metrics_sample.txt"
run "${SCP[@]}" "$NODE_NAME:~/$REMOTE_DIR/serve.log" "$ROOT/benchmarks/results/$ENV_LABEL/serve.log"

echo "== 6/6 Agent-task check from here, through an SSH tunnel"
if [ "$EXECUTE" = "yes" ]; then
  "${SSH[@]}" -- -N -L "$TUNNEL_PORT:localhost:8000" &
  tunnel=$!
  trap 'kill $tunnel 2>/dev/null || true' EXIT
  sleep 8
  python3 "$ROOT/benchmarks/agent_eval.py" --env "$ENV_LABEL" --model "$MODEL" \
    --base-url "http://localhost:$TUNNEL_PORT/v1" --repeats 3
else
  echo "+ ${SSH[*]} -- -N -L $TUNNEL_PORT:localhost:8000 &"
  echo "+ python3 benchmarks/agent_eval.py --env $ENV_LABEL --model $MODEL --base-url http://localhost:$TUNNEL_PORT/v1 --repeats 3"
fi

echo
echo "Done. Metrics are in benchmarks/results/$ENV_LABEL/. NOW tear the VM down:"
echo "  deploy/tpu/teardown.sh --execute"
