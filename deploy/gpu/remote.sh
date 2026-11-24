#!/usr/bin/env bash
# Runs ON the VM. Subcommands: setup | gpu | cpu. Each writes a DONE_<step> or
# FAILED_<step> marker so the laptop can check real state instead of trusting its own script.
set -euo pipefail
B="$HOME/gemma-bench"; cd "$B"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
GPU_ENV="${GPU_ENV:-cloud-gpu-l4-fp8}"; CPU_ENV="${CPU_ENV:-cloud-cpu-llamacpp}"
step() { rm -f "DONE_$1" "FAILED_$1"; if "cmd_$1" > "$1.out" 2>&1; then touch "DONE_$1"; else touch "FAILED_$1"; tail -30 "$1.out"; return 1; fi; }
wait_ready() { # port
  for _ in $(seq 1 120); do curl -sf "localhost:$1/v1/models" >/dev/null && return 0; sleep 10; done; return 1; }
cmd_setup() {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sudo sh; }
  sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L || {
    echo "docker cannot see the GPU; installing nvidia-container-toolkit";
    sudo apt-get install -y -qq nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
    sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L; }
  sudo docker volume create docker_hf-cache >/dev/null
  sudo docker pull vllm/vllm-openai:gemma4
  sudo docker pull ghcr.io/ggml-org/llama.cpp:server
  lscpu | grep -E 'Model name|^CPU\(s\)'
}
cmd_gpu() {
  sudo docker compose -f docker/compose.bench.yml up -d
  wait_ready 8000
  ver="$(curl -s localhost:8000/version || true)"
  sudo env "PATH=$PATH" python3 benchmarks/run_benchmarks.py --env "$GPU_ENV" --base-url http://localhost:8000 \
    --meta hardware="NVIDIA L4 24GB (GCE g2-standard-8)" --meta engine=vllm --meta precision=fp8 \
    --meta checkpoint="$MODEL" --meta max_model_len=9216 --meta max_num_seqs=8 --meta server_version="$ver"
  sudo chown -R "$USER" benchmarks/results
  curl -s localhost:8000/metrics | head -c 2000 > metrics_sample.txt || true
  sudo docker logs gemma-vllm-bench > serve_gpu.log 2>&1 || true
  # keep the server up: the laptop runs the agent check through a tunnel, then calls "stopgpu"
}
cmd_stopgpu() { sudo docker compose -f docker/compose.bench.yml down; }
cmd_cpu() {
  sudo docker compose -f docker/compose.cpu.yml up -d
  wait_ready 8081
  m=(--env "$CPU_ENV" --base-url http://localhost:8081 --meta hardware="GCE g2-standard-8 vCPUs (Intel, no GPU used)" --meta engine=llama.cpp --meta precision=GGUF-Q8_0 --meta checkpoint="$MODEL")
  r() { sudo env "PATH=$PATH" python3 benchmarks/run_benchmarks.py "${m[@]}" "$@"; }
  r --only short --trials 3
  r --only medium --trials 1
  r --only long --concurrencies 1 --trials 1
  sudo chown -R "$USER" benchmarks/results
  sudo docker logs gemma-llama-cpu > serve_cpu.log 2>&1 || true
}
cmd_stopcpu() { sudo docker compose -f docker/compose.cpu.yml down; }
case "${1:-}" in
  setup|gpu|cpu) step "$1" ;;
  stopgpu|stopcpu) "cmd_$1" ;;
  *) echo "usage: remote.sh setup|gpu|cpu|stopgpu|stopcpu" >&2; exit 2 ;;
esac
