#!/usr/bin/env bash
# Deletes ONLY the queued resource and node created by create.sh, by exact
# name, then proves they are gone. Dry run by default.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
parse_execute "${1:-}"
require_target

case "$QR_NAME" in gemma-bench-*) ;; *) echo "REFUSING: unexpected resource name '$QR_NAME'." >&2; exit 1;; esac

echo "Will delete (exact names only): queued resource $QR_NAME, node $NODE_NAME in $PROJECT/$ZONE"

# --force also deletes the node the queued resource created.
run gcloud alpha compute tpus queued-resources delete "$QR_NAME" \
  --project "$PROJECT" --zone "$ZONE" --force --quiet

if [ "$EXECUTE" = "yes" ]; then
  echo
  echo "Verifying..."
  gone=yes
  gcloud alpha compute tpus queued-resources describe "$QR_NAME" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1 && { echo "STILL EXISTS: queued resource $QR_NAME"; gone=no; }
  gcloud compute tpus tpu-vm describe "$NODE_NAME" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1 && { echo "STILL EXISTS: node $NODE_NAME (re-run, or delete it by hand)"; gone=no; }
  echo "TPU nodes left in $ZONE:"; gcloud compute tpus tpu-vm list --project "$PROJECT" --zone "$ZONE" 2>&1 | sed 's/^/  /'
  echo "Queued resources left in $ZONE:"; gcloud alpha compute tpus queued-resources list --project "$PROJECT" --zone "$ZONE" 2>&1 | sed 's/^/  /'
  [ "$gone" = "yes" ] && echo "Teardown verified: nothing from this run remains." || exit 1
fi
