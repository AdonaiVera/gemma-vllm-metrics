#!/usr/bin/env bash
# Deletes ONLY the VM named in config.sh (and its auto-delete boot disk), then proves it is gone.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
parse_execute "${1:-}"; require_target
case "$VM_NAME" in gemma-bench-*) ;; *) echo "REFUSING: unexpected name $VM_NAME" >&2; exit 1;; esac
echo "Will delete exactly: $VM_NAME in $PROJECT/$ZONE"
run gcloud compute instances delete "$VM_NAME" --project "$PROJECT" --zone "$ZONE" --quiet --delete-disks=boot
if [ "$EXECUTE" = yes ]; then
  gcloud compute instances describe "$VM_NAME" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1 && { echo "STILL EXISTS: $VM_NAME"; exit 1; }
  echo "Disks named like ours left:"; gcloud compute disks list --project "$PROJECT" --filter="name~$VM_NAME" --format='value(name)' | sed 's/^/  /'
  echo "Teardown verified: $VM_NAME is gone."
fi
