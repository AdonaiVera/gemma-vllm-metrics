#!/usr/bin/env bash
# Brings up vLLM's OpenAI-compatible server for the configured Gemma model.
# The plugin never manages this process itself -- it's a separate, long-lived
# service the panel/operators talk to over HTTP. See README for rationale.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../docker"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created docker/.env from .env.example -- edit it to change MODEL_ID/VLLM_PORT."
fi

docker compose up -d
echo "Waiting for vLLM to report ready on /v1/models ..."

PORT=$(grep -E '^VLLM_PORT=' .env | cut -d= -f2)
PORT="${PORT:-8000}"

for _ in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "vLLM is up: http://localhost:${PORT}/v1"
    exit 0
  fi
  sleep 5
done

echo "vLLM did not become ready in time. Check logs with: docker compose -f docker/compose.yml logs -f"
exit 1
