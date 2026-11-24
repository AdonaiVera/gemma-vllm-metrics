#!/usr/bin/env bash
# Shared settings for the one-shot TPU benchmark run. Sourced by the other
# scripts. This targets a PRODUCTION project, so every script refuses to act
# unless the active account and project match what is pinned here.

# Real values live in config.local.sh (untracked). Copy config.local.example.sh.
EXPECTED_ACCOUNT="you@example.com"
PROJECT="your-gcp-project"
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/config.local.sh"

ZONE="${ZONE:-us-east5-a}"
ACCEL="v6e-1"
RUNTIME="v2-alpha-tpuv6e"

# Everything this run creates carries this exact name and label, and the
# teardown script deletes only this name.
QR_NAME="gemma-bench-qr"
NODE_NAME="gemma-bench-v6e1"
LABELS="purpose=torchtpu-sprint-benchmark,owner=adonai,delete-after=2026-11-26"

# Backstops so a forgotten VM cannot bill for long.
MAX_RUN="3h"       # hard stop once running (Flex-start)
QUEUE_WAIT="${QUEUE_WAIT:-1h}"    # give up waiting for capacity after this

# Rough ceilings used only to print a worst-case cost before creating.
PRICE_ON_DEMAND_PER_HOUR="2.70"
PRICE_FLEX_PER_HOUR="1.35"

ENV_LABEL="tpu-v6e-fp8"
MODEL="google/gemma-4-E2B-it"
REMOTE_DIR="gemma-bench"

GCLOUD=(gcloud --project "$PROJECT")

require_target() {
  local account project
  account="$(gcloud config get-value account 2>/dev/null)"
  project="$(gcloud config get-value project 2>/dev/null)"
  if [ "$account" != "$EXPECTED_ACCOUNT" ]; then
    echo "REFUSING: active account is '$account', expected '$EXPECTED_ACCOUNT'." >&2
    exit 1
  fi
  if [ "$project" != "$PROJECT" ]; then
    echo "NOTE: gcloud default project is '$project'; every command here passes --project $PROJECT explicitly."
  fi
}

# Runs a command, or only prints it unless EXECUTE=yes.
run() {
  echo "+ $*"
  if [ "${EXECUTE:-no}" = "yes" ]; then
    "$@"
  fi
}

parse_execute() {
  EXECUTE=no
  if [ "${1:-}" = "--execute" ]; then
    EXECUTE=yes
  else
    echo "DRY RUN: nothing will be changed. Pass --execute to act."
  fi
}
