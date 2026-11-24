#!/usr/bin/env bash
# Software stop for an on-demand TPU (which has no --max-run-duration).
# Counts from the moment the queued resource turns ACTIVE and deletes it by
# exact name after LIMIT_MIN minutes. Exits when the resource is gone/failed.
# Only protects you while this machine is awake and online.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
require_target
LIMIT_MIN="${LIMIT_MIN:-150}"
start=""
echo "$(date -u +%T)Z watchdog on: delete $QR_NAME ${LIMIT_MIN}min after ACTIVE"
while true; do
  state=$(gcloud alpha compute tpus queued-resources describe "$QR_NAME" --project "$PROJECT" --zone "$ZONE" --format='value(state.state)' 2>/dev/null || echo GONE)
  case "$state" in
    ACTIVE) [ -z "$start" ] && { start=$(date +%s); echo "$(date -u +%T)Z ACTIVE; clock started"; } ;;
    FAILED|GONE|"") echo "$(date -u +%T)Z state=$state; nothing to guard"; exit 0 ;;
  esac
  if [ -n "$start" ] && [ $(( $(date +%s) - start )) -ge $(( LIMIT_MIN * 60 )) ]; then
    echo "$(date -u +%T)Z LIMIT REACHED: tearing down"
    "$HERE/teardown.sh" --execute
    exit 0
  fi
  sleep 30
done
