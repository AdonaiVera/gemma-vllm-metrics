#!/usr/bin/env bash
# Drives the whole run from this machine. Long steps run DETACHED on the VM and
# are polled via marker files (a stalled SSH session cannot stall the run).
# Does NOT delete the VM; run teardown.sh afterwards. Dry run by default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/config.sh"; parse_execute "${1:-}"; require_target
SSH=(gcloud compute ssh "$VM_NAME" --project "$PROJECT" --zone "$ZONE")
SCP=(gcloud compute scp --project "$PROJECT" --zone "$ZONE")
[ "$EXECUTE" = yes ] && trap 'echo "!! A STEP FAILED. VM still billing until the hard stop (${MAX_RUN}). Inspect or: deploy/gpu/teardown.sh --execute"' ERR
rexec() { run "${SSH[@]}" --command "$1"; }
detached() { # step name, timeout minutes
  rexec "cd ~/$REMOTE_DIR && setsid nohup env MODEL=$MODEL GPU_ENV=$GPU_ENV_LABEL CPU_ENV=$CPU_ENV_LABEL bash remote.sh $1 > run_$1.log 2>&1 < /dev/null &"
  [ "$EXECUTE" = yes ] || return 0
  for _ in $(seq 1 $(( $2 * 2 ))); do
    sleep 30
    s=$("${SSH[@]}" --command "cd ~/$REMOTE_DIR; ls DONE_$1 FAILED_$1 2>/dev/null || true" 2>/dev/null || true)
    case "$s" in *DONE_*) echo "step $1 done"; return 0;; *FAILED_*) echo "step $1 FAILED"; "${SSH[@]}" --command "tail -40 ~/$REMOTE_DIR/$1.out"; return 1;; esac
  done
  echo "step $1 timed out waiting"; return 1
}
echo "== 1 copy code";   rexec "mkdir -p ~/$REMOTE_DIR/docker ~/$REMOTE_DIR/benchmarks"
run "${SCP[@]}" --recurse "$ROOT/benchmarks/run_benchmarks.py" "$ROOT/benchmarks/workloads.json" "$VM_NAME:~/$REMOTE_DIR/benchmarks/"
run "${SCP[@]}" "$ROOT/docker/compose.bench.yml" "$ROOT/docker/compose.cpu.yml" "$VM_NAME:~/$REMOTE_DIR/docker/"
run "${SCP[@]}" "$HERE/remote.sh" "$VM_NAME:~/$REMOTE_DIR/"
echo "== 2 setup (driver, docker, images)"; detached setup 25
echo "== 3 GPU matrix (27 runs)";           detached gpu 75
echo "== 4 agent check through tunnel"
if [ "$EXECUTE" = yes ]; then
  "${SSH[@]}" -- -N -L 18000:localhost:8000 & tun=$!; sleep 8
  python3 "$ROOT/benchmarks/agent_eval.py" --env "$GPU_ENV_LABEL" --model "$MODEL" --base-url http://localhost:18000/v1 --repeats 3
  kill $tun 2>/dev/null || true
else echo "+ agent_eval.py via ssh tunnel"; fi
rexec "bash ~/$REMOTE_DIR/remote.sh stopgpu"
echo "== 5 CPU matrix on the same VM (reduced matrix, same as local CPU)"; detached cpu 60
if [ "$EXECUTE" = yes ]; then
  "${SSH[@]}" -- -N -L 18081:localhost:8081 & tun=$!; sleep 8
  python3 "$ROOT/benchmarks/agent_eval.py" --env "$CPU_ENV_LABEL" --model "$MODEL" --base-url http://localhost:18081/v1 --repeats 3
  kill $tun 2>/dev/null || true
fi
echo "== 6 pull results"
mkdir -p "$ROOT/benchmarks/results"
run "${SCP[@]}" --recurse "$VM_NAME:~/$REMOTE_DIR/benchmarks/results/$GPU_ENV_LABEL" "$VM_NAME:~/$REMOTE_DIR/benchmarks/results/$CPU_ENV_LABEL" "$ROOT/benchmarks/results/"
run "${SCP[@]}" "$VM_NAME:~/$REMOTE_DIR/metrics_sample.txt" "$VM_NAME:~/$REMOTE_DIR/serve_gpu.log" "$VM_NAME:~/$REMOTE_DIR/serve_cpu.log" "$ROOT/benchmarks/results/$GPU_ENV_LABEL/"
echo "Done. NOW: deploy/gpu/teardown.sh --execute"
