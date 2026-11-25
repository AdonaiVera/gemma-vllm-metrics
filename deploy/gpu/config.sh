#!/usr/bin/env bash
# Shared settings for the one-shot cloud GPU benchmark run (one L4 VM).
# PRODUCTION project: every script refuses to act unless account matches.
# Real values live in config.local.sh (untracked). Copy config.local.example.sh.
EXPECTED_ACCOUNT="you@example.com"
PROJECT="your-gcp-project"
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.local.sh" ] && source "$(dirname "${BASH_SOURCE[0]}")/config.local.sh"
ZONE="${ZONE:-us-central1-a}"
MACHINE="g2-standard-8"          # 1x NVIDIA L4 (24 GB), 8 vCPU, 32 GB RAM
ACCEL_TYPE="nvidia-l4"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_GB=120
VM_NAME="gemma-bench-l4"         # the ONLY thing these scripts create/delete
LABELS="purpose=torchtpu-sprint-benchmark,owner=adonai,delete-after=2026-11-30"
MAX_RUN="${MAX_RUN:-2h}"                     # hard stop enforced by Google (deletes the VM)
WATCHDOG_MIN="${WATCHDOG_MIN:-110}"   # second layer: this machine deletes it too
PRICE_PER_HOUR_ESTIMATE="1.00"   # only for the printed ceiling; verify before relying on it
GPU_ENV_LABEL="cloud-gpu-l4-fp8"
CPU_ENV_LABEL="cloud-cpu-llamacpp"
MODEL="google/gemma-4-E2B-it"
REMOTE_DIR="gemma-bench"

require_target() {
  local account; account="$(gcloud config get-value account 2>/dev/null)"
  [ "$account" = "$EXPECTED_ACCOUNT" ] || { echo "REFUSING: account is '$account', expected '$EXPECTED_ACCOUNT'." >&2; exit 1; }
}
run() { echo "+ $*"; [ "${EXECUTE:-no}" = "yes" ] && "$@"; return 0; }
parse_execute() {
  EXECUTE=no
  if [ "${1:-}" = "--execute" ]; then EXECUTE=yes; else echo "DRY RUN: nothing will be changed. Pass --execute to act."; fi
}
