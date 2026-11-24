#!/usr/bin/env bash
# Runs ON the TPU VM. Subcommands: install | serve | bench
# Install steps follow the vLLM-TorchTPU docs (Python 3.12, CPU torch index,
# `vllm-torchtpu` from PyPI). Verify against the live docs on first use.
set -euo pipefail

BASE="$HOME/gemma-bench"
VENV="$HOME/uv_venv"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
ENV_LABEL="${ENV_LABEL:-tpu-v6e-fp8}"
# Flags that may not exist on this backend are kept separate so a failed
# start can be retried without them; the log shows which one was rejected.
TOOL_FLAGS="${TOOL_FLAGS:---enable-auto-tool-choice --tool-call-parser gemma4}"
EXTRA_FLAGS="${EXTRA_FLAGS:---async-scheduling}"
export VLLM_XLA_CACHE_PATH="$BASE/xla_cache"

activate() { source "$VENV/bin/activate"; }

cmd_install() {
  sudo apt-get update -qq && sudo apt-get install -y -qq libibverbs1
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 "$VENV"
  activate
  export UV_TORCH_BACKEND=cpu
  uv pip install vllm-torchtpu
  python -c "from vllm.platforms import current_platform; print('platform:', current_platform.device_name)"
}

cmd_serve() {
  activate
  mkdir -p "$BASE"
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" \
    --quantization fp8 \
    --tensor-parallel-size 1 \
    --max-model-len 9216 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    $EXTRA_FLAGS $TOOL_FLAGS \
    > "$BASE/serve.log" 2>&1 &
  echo "$!" > "$BASE/serve.pid"
  echo "Waiting for the server (first start compiles TPU graphs and can take several minutes)..."
  for _ in $(seq 1 120); do
    if curl -sf localhost:8000/v1/models >/dev/null; then echo "READY"; return 0; fi
    if ! kill -0 "$(cat "$BASE/serve.pid")" 2>/dev/null; then
      echo "SERVER EXITED. Last log lines:"; tail -40 "$BASE/serve.log"; return 1
    fi
    sleep 10
  done
  echo "TIMED OUT waiting for the server. Last log lines:"; tail -40 "$BASE/serve.log"; return 1
}

cmd_bench() {
  activate
  cd "$BASE/benchmarks"
  version="$(curl -s localhost:8000/version || true)"
  pkgs="$( (uv pip list 2>/dev/null || pip list 2>/dev/null) | grep -iE '^(vllm|vllm-torchtpu|torch|torch-tpu|libtpu) ' | tr -s ' ' '=' | tr '\n' ';' || true)"
  python3 run_benchmarks.py --env "$ENV_LABEL" --base-url http://localhost:8000 \
    --meta hardware="TPU v6e-1" --meta engine=vllm-torchtpu --meta precision=fp8 \
    --meta checkpoint="$MODEL" --meta kv_cache_dtype=auto --meta max_model_len=9216 \
    --meta max_num_seqs=8 --meta server_version="$version" --meta packages="$pkgs" \
    --meta extra_flags="$EXTRA_FLAGS" --meta tool_flags="$TOOL_FLAGS"
  # Prometheus endpoint check for the blog (is /metrics available on this backend?).
  curl -s localhost:8000/metrics | head -c 2000 > "$BASE/metrics_sample.txt" || true
}

case "${1:-}" in
  install) cmd_install ;;
  serve) cmd_serve ;;
  bench) cmd_bench ;;
  *) echo "usage: remote.sh install|serve|bench" >&2; exit 2 ;;
esac
