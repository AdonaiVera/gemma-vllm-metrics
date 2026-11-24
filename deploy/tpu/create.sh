#!/usr/bin/env bash
# Creates exactly one labeled v6e-1 via a Flex-start queued resource with a
# hard max run duration. Dry run by default.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
parse_execute "${1:-}"
require_target

# PROVISIONING=flex-start (default, cheaper, may never be granted) or on-demand.
PROVISIONING="${PROVISIONING:-flex-start}"
case "$PROVISIONING" in
  flex-start) PM_ARGS=(--provisioning-model flex-start --max-run-duration "$MAX_RUN") ;;
  # STANDARD rejects --max-run-duration: there is NO hard stop. Use watchdog.sh.
  on-demand)  PM_ARGS=() ;;
  *) echo "PROVISIONING must be flex-start or on-demand" >&2; exit 1 ;;
esac

if gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
     --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
  echo "REFUSING: queued resource $QR_NAME already exists." >&2
  exit 1
fi

if [ "$EXECUTE" = "yes" ]; then
  echo "This will create ONE $ACCEL in production project '$PROJECT' ($ZONE)."
  echo "Provisioning: $PROVISIONING"
  [ "$PROVISIONING" = "flex-start" ] && echo "Hard stop after $MAX_RUN." || echo "NO hard stop (on-demand): run watchdog.sh right after this."
  echo "Gives up waiting for capacity after $QUEUE_WAIT."
  echo "Type the project name to confirm:"
  read -r answer
  [ "$answer" = "$PROJECT" ] || { echo "Not confirmed. Nothing created."; exit 1; }
fi

run gcloud alpha compute tpus queued-resources create "$QR_NAME" \
  --project "$PROJECT" --zone "$ZONE" \
  --node-id "$NODE_NAME" \
  --accelerator-type "$ACCEL" \
  --runtime-version "$RUNTIME" \
  "${PM_ARGS[@]}" \
  --valid-until-duration "$QUEUE_WAIT" \
  --labels "$LABELS"

echo
echo "Watch it:   gcloud alpha compute tpus queued-resources describe $QR_NAME --project $PROJECT --zone $ZONE --format='value(state.state)'"
echo "Kill it:    deploy/tpu/teardown.sh --execute"
