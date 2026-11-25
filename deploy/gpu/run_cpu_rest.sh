#!/usr/bin/env bash
# Finishes the cloud CPU leg on a fresh VM: medium + long-c1 cells, then the agent check.
# Same detached/marker-file pattern as run_on_gpu.sh. Does NOT delete the VM.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/config.sh"; parse_execute "${1:-}"; require_target
SSH=(gcloud compute ssh "$VM_NAME" --project "$PROJECT" --zone "$ZONE")
SCP=(gcloud compute scp --project "$PROJECT" --zone "$ZONE")
rexec() { run "${SSH[@]}" --command "$1"; }
detached() {
  rexec "cd ~/$REMOTE_DIR && setsid nohup env SKIP_VLLM=1 MODEL=$MODEL CPU_ENV=$CPU_ENV_LABEL bash remote.sh $1 > run_$1.log 2>&1 < /dev/null &"
  [ "$EXECUTE" = yes ] || return 0
  for _ in $(seq 1 $(( $2 * 2 ))); do
    sleep 30
    s=$("${SSH[@]}" --command "cd ~/$REMOTE_DIR; ls DONE_$1 FAILED_$1 2>/dev/null || true" 2>/dev/null || true)
    case "$s" in *DONE_*) echo "step $1 done"; return 0;; *FAILED_*) echo "step $1 FAILED"; "${SSH[@]}" --command "tail -40 ~/$REMOTE_DIR/$1.out"; return 1;; esac
  done; echo "step $1 timed out"; return 1
}
rexec "mkdir -p ~/$REMOTE_DIR/docker ~/$REMOTE_DIR/benchmarks"
run "${SCP[@]}" "$ROOT/benchmarks/run_benchmarks.py" "$ROOT/benchmarks/workloads.json" "$VM_NAME:~/$REMOTE_DIR/benchmarks/"
run "${SCP[@]}" "$ROOT/docker/compose.cpu.yml" "$VM_NAME:~/$REMOTE_DIR/docker/"
run "${SCP[@]}" "$HERE/remote.sh" "$VM_NAME:~/$REMOTE_DIR/"
echo "== setup"; detached setup 25
echo "== CPU medium + long c1"; detached cpu_rest 110
echo "== agent check on the cloud CPU"
if [ "$EXECUTE" = yes ]; then
  "${SSH[@]}" -- -N -L 18081:localhost:8081 & tun=$!; sleep 8
  python3 "$ROOT/benchmarks/agent_eval.py" --env "$CPU_ENV_LABEL" --model "$MODEL" --base-url http://localhost:18081/v1 --repeats 3
  kill $tun 2>/dev/null || true
fi
rexec "bash ~/$REMOTE_DIR/remote.sh stopcpu"
echo "== pull"
mkdir -p "$ROOT/benchmarks/results/$CPU_ENV_LABEL"
run "${SCP[@]}" --recurse "$VM_NAME:~/$REMOTE_DIR/benchmarks/results/$CPU_ENV_LABEL/*" "$ROOT/benchmarks/results/$CPU_ENV_LABEL/"
echo "Done. NOW: deploy/gpu/teardown.sh --execute (ZONE=$ZONE)"
