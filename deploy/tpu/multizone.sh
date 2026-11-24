#!/usr/bin/env bash
# Asks for one on-demand v6e-1 in several zones at once and keeps whichever is
# granted FIRST; every other request is deleted by exact name. Billing only
# starts for a VM that actually exists. Dry run by default.
#   multizone.sh create [--execute]   submit the requests
#   multizone.sh watch  [--execute]   pick the winner, delete the rest, arm the watchdog
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
ZONES="${ZONES:-us-central1-b us-central1-a us-central1-c us-east1-d us-east4-a us-east5-a us-east5-b us-south1-a}"
QUEUE_WAIT="${QUEUE_WAIT:-2h}"
export QUEUE_WAIT
sub="${1:-}"; parse_execute "${2:-}"; require_target

state() { gcloud alpha compute tpus queued-resources describe "$QR_NAME" --project "$PROJECT" --zone "$1" --format='value(state.state)' 2>/dev/null || echo GONE; }

case "$sub" in
create)
  if [ "$EXECUTE" = "yes" ]; then
    echo "This will submit ONE on-demand $ACCEL request in each of: $ZONES"
    echo "Queue window $QUEUE_WAIT. NO hard stop once running: watchdog.sh limits it."
    echo "Type the project name to confirm:"; read -r a
    [ "$a" = "$PROJECT" ] || { echo "Not confirmed."; exit 1; }
  fi
  for z in $ZONES; do
    run gcloud alpha compute tpus queued-resources create "$QR_NAME" --project "$PROJECT" --zone "$z" \
      --node-id "$NODE_NAME" --accelerator-type "$ACCEL" --runtime-version "$RUNTIME" \
      --valid-until-duration "$QUEUE_WAIT" --labels "$LABELS" --async \
      || echo "!! request in $z failed (continuing)"
  done ;;
watch)
  [ "$EXECUTE" = "yes" ] || { echo "(dry run) would poll $ZONES, keep the first ACTIVE, delete the others"; exit 0; }
  winner=""
  while [ -z "$winner" ]; do
    alive=0
    for z in $ZONES; do
      s="$(state "$z")"
      case "$s" in ACTIVE) winner="$z"; break ;; FAILED|GONE|"") ;; *) alive=$((alive+1)) ;; esac
    done
    if [ -z "$winner" ] && [ "$alive" -eq 0 ]; then echo "$(date -u +%T)Z no request left alive; none granted"; exit 2; fi
    [ -z "$winner" ] && sleep 10
  done
  echo "$(date -u +%T)Z WINNER: $winner"; echo "$winner" > "$HERE/.winner_zone"
  # arm the stop for the winner FIRST, then delete every other request
  ( ZONE="$winner" nohup "$HERE/watchdog.sh" > "$HERE/.watchdog_$winner.log" 2>&1 & )
  for z in $ZONES; do
    [ "$z" = "$winner" ] && continue
    ZONE="$z" "$HERE/teardown.sh" --execute > "$HERE/.teardown_$z.log" 2>&1 \
      && echo "deleted request in $z" || echo "!! could not delete request in $z (check $HERE/.teardown_$z.log)"
  done
  echo "Keep: ZONE=$winner. Next: ZONE=$winner deploy/tpu/run_on_tpu.sh --execute" ;;
*) echo "usage: multizone.sh create|watch [--execute]" >&2; exit 2 ;;
esac
