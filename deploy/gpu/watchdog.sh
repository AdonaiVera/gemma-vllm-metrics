#!/usr/bin/env bash
# Second safety layer: deletes the VM WATCHDOG_MIN minutes after it is seen RUNNING.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config.sh"; require_target
start=""; echo "$(date -u +%T)Z watchdog: delete $VM_NAME ${WATCHDOG_MIN}min after RUNNING"
while true; do
  st=$(gcloud compute instances describe "$VM_NAME" --project "$PROJECT" --zone "$ZONE" --format='value(status)' 2>/dev/null || echo GONE)
  case "$st" in RUNNING) [ -z "$start" ] && { start=$(date +%s); echo "$(date -u +%T)Z RUNNING; clock started"; } ;;
    GONE|TERMINATED|"") echo "$(date -u +%T)Z state=$st; nothing to guard"; exit 0 ;; esac
  if [ -n "$start" ] && [ $(( $(date +%s)-start )) -ge $((WATCHDOG_MIN*60)) ]; then
    echo "LIMIT REACHED: tearing down"; "$HERE/teardown.sh" --execute; exit 0; fi
  sleep 30
done
