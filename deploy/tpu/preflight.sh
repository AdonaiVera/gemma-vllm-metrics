#!/usr/bin/env bash
# Read-only checks. Creates and changes nothing. Run before create.sh.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

fail=0
check() { # description, command...
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "ok    $desc"; else echo "FAIL  $desc"; fail=1; fi
}

require_target
echo "Project: $PROJECT   Zone: $ZONE   Accelerator: $ACCEL   Runtime: $RUNTIME"
echo

check "TPU API enabled" bash -c "gcloud services list --enabled --project $PROJECT --filter='name:tpu.googleapis.com' --format='value(name)' | grep -q tpu"
check "Compute API enabled" bash -c "gcloud services list --enabled --project $PROJECT --filter='name:compute.googleapis.com' --format='value(name)' | grep -q compute"
check "$ACCEL offered in $ZONE" bash -c "gcloud compute tpus tpu-vm accelerator-types list --project $PROJECT --zone $ZONE --format='value(type)' | grep -qx $ACCEL"
check "runtime $RUNTIME exists in $ZONE" bash -c "gcloud compute tpus tpu-vm versions list --project $PROJECT --zone $ZONE --format='value(version)' | grep -qx $RUNTIME"
check "alpha queued-resources supports --max-run-duration" bash -c "gcloud alpha compute tpus queued-resources create --help 2>&1 | grep -q max-run-duration"

echo
echo "Existing TPU nodes in the zone (should be none, or unrelated):"
gcloud compute tpus tpu-vm list --project "$PROJECT" --zone "$ZONE" 2>&1 | sed 's/^/  /'
echo "Existing queued resources in the zone:"
gcloud alpha compute tpus queued-resources list --project "$PROJECT" --zone "$ZONE" 2>&1 | sed 's/^/  /'

if gcloud alpha compute tpus queued-resources describe "$QR_NAME" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
  echo "FAIL  a queued resource named $QR_NAME already exists"; fail=1
fi

echo
echo "Worst-case cost if the VM runs the full $MAX_RUN:"
python3 - <<EOF
hours = float("${MAX_RUN%h}")
print(f"  Flex-start  ~\${hours * ${PRICE_FLEX_PER_HOUR}:.2f}")
print(f"  On-demand   ~\${hours * ${PRICE_ON_DEMAND_PER_HOUR}:.2f}  (upper bound; prices unverified, see benchmarks/pricing.json)")
EOF
echo
[ "$fail" -eq 0 ] && echo "Preflight passed." || { echo "Preflight FAILED."; exit 1; }
echo "Not checked here: TPU v6e quota and capacity (only the create call reveals these),"
echo "and whether the billing account has credits (check in the console)."
