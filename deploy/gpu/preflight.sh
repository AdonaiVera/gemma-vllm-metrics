#!/usr/bin/env bash
# Read-only. Creates and changes nothing.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
require_target
fail=0
ck() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then echo "ok    $d"; else echo "FAIL  $d"; fail=1; fi; }
echo "Project $PROJECT  zone $ZONE  machine $MACHINE ($ACCEL_TYPE)"
ck "Compute API enabled" bash -c "gcloud services list --enabled --project $PROJECT --filter='name:compute.googleapis.com' --format='value(name)' | grep -q compute"
ck "$ACCEL_TYPE offered in $ZONE" bash -c "gcloud compute accelerator-types list --project $PROJECT --filter='name=$ACCEL_TYPE AND zone~$ZONE' --format='value(name)' | grep -q ."
ck "image family $IMAGE_FAMILY exists" gcloud compute images describe-from-family "$IMAGE_FAMILY" --project "$IMAGE_PROJECT"
ck "instances create supports --max-run-duration" bash -c "gcloud compute instances create --help 2>&1 | grep -q max-run-duration"
ck "no VM named $VM_NAME yet" bash -c "! gcloud compute instances describe $VM_NAME --project $PROJECT --zone $ZONE"
echo; echo "L4 quota (region of $ZONE):"
gcloud compute regions describe "${ZONE%-*}" --project "$PROJECT" --format=json 2>/dev/null | python3 -c "
import sys,json
for q in json.load(sys.stdin)['quotas']:
    if 'L4' in q['metric'] or q['metric']=='GPUS_ALL_REGIONS': print('  ',q['metric'],'limit',q['limit'],'usage',q['usage'])"
echo "Existing instances in project (not ours):"; gcloud compute instances list --project "$PROJECT" --format='value(name,zone.basename(),status)' | sed 's/^/  /'
python3 -c "print(f'Worst-case cost at {float(\"${MAX_RUN%h}\")}h: ~\${float(\"${MAX_RUN%h}\")*${PRICE_PER_HOUR_ESTIMATE}:.2f} (rate is an estimate)')"
[ $fail -eq 0 ] && echo "Preflight passed." || { echo "Preflight FAILED."; exit 1; }
