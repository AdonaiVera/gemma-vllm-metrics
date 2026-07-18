#!/usr/bin/env bash
# Exposes the local vLLM server to the internet via ngrok, gated by a bearer
# token checked at the ngrok edge (before traffic ever reaches vLLM). The
# same token goes in voxel-agent's "API Key" field for this provider --
# LiteLLM already sends it as `Authorization: Bearer <key>`, so no code
# changes are needed on either side.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../docker"

if ! command -v ngrok &> /dev/null; then
  echo "ngrok not found on PATH. Install it: https://ngrok.com/download" >&2
  exit 1
fi

if [ ! -f .env.ngrok ]; then
  echo "NGROK_BEARER_TOKEN=$(openssl rand -hex 32)" > .env.ngrok
  chmod 600 .env.ngrok
  echo "Generated a new bearer token in docker/.env.ngrok"
fi

# shellcheck disable=SC1091
source .env.ngrok

sed "s/__NGROK_BEARER_TOKEN__/${NGROK_BEARER_TOKEN}/" traffic-policy.yml.example > traffic-policy.yml
chmod 600 traffic-policy.yml

PORT=$(grep -E '^VLLM_PORT=' .env 2>/dev/null | cut -d= -f2)
PORT="${PORT:-8000}"

echo "Bearer token for this session (put this in voxel-agent's Settings -> Providers -> API Key field):"
echo "  ${NGROK_BEARER_TOKEN}"
echo
echo "Starting ngrok tunnel to localhost:${PORT} ..."
exec ngrok http "${PORT}" --traffic-policy-file=traffic-policy.yml
