#!/usr/bin/env bash
# Tries zones in order until one grants the L4 VM, then arms the watchdog for it.
# A failed attempt creates nothing. Typed confirmation is supplied by the caller.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
for z in "$@"; do
  echo "== $(date -u +%T)Z $z"
  echo "$PROJECT" | ZONE=$z "$HERE/create.sh" --execute > "/tmp/gpu_create_$z.log" 2>&1
  sleep 20   # a VM can appear as STAGING and then be withdrawn; only count it if RUNNING
  st=$(gcloud compute instances describe gemma-bench-l4 --project "$PROJECT" --zone "$z" --format='value(status)' 2>/dev/null)
  if [ "$st" = RUNNING ]; then
    echo "$z" > /tmp/gpu_zone; echo "CREATED and RUNNING in $z"
    ZONE=$z WATCHDOG_MIN="$WATCHDOG_MIN" setsid nohup "$HERE/watchdog.sh" > /tmp/gpu-watchdog.log 2>&1 < /dev/null &
    exit 0
  fi
  echo "no: $(grep -iE 'STOCKOUT|quota|not have enough|Invalid|ERROR' /tmp/gpu_create_$z.log | head -1 | cut -c1-160)"
done
echo "NONE GRANTED"
