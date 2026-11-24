#!/usr/bin/env bash
# Creates exactly one VM. Google deletes it after MAX_RUN (termination action
# DELETE), and the boot disk is auto-deleted with it. Dry run by default.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
parse_execute "${1:-}"; require_target
if gcloud compute instances describe "$VM_NAME" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
  echo "REFUSING: $VM_NAME already exists." >&2; exit 1; fi
if [ "$EXECUTE" = yes ]; then
  echo "Creates ONE $MACHINE ($ACCEL_TYPE) in production project '$PROJECT' ($ZONE)."
  echo "Hard stop and delete after $MAX_RUN. Run watchdog.sh right after."
  echo "Type the project name to confirm:"; read -r a
  [ "$a" = "$PROJECT" ] || { echo "Not confirmed. Nothing created."; exit 1; }
fi
run gcloud compute instances create "$VM_NAME" --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" --accelerator "type=$ACCEL_TYPE,count=1" \
  --maintenance-policy TERMINATE --provisioning-model STANDARD \
  --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
  --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced --boot-disk-auto-delete \
  --metadata "install-nvidia-driver=True" --labels "$LABELS"
echo "Kill it: deploy/gpu/teardown.sh --execute"
