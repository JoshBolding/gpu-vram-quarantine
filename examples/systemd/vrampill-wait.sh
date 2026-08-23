#!/bin/bash
# Block until the quarantine marker exists, or fail. Nothing that fills VRAM on
# Card C may start before this succeeds.
set -u
READY="${READY:-/run/user/$(id -u)/vrampill.ready}"
# Must exceed vrampill's --find-seconds + --quiet-seconds (default 1800 + 180).
TIMEOUT="${TIMEOUT:-2100}"
for i in $(seq 1 "$TIMEOUT"); do
  if [ -f "$READY" ]; then
    echo "quarantine ready after ${i}s:"
    sed 's/^/  /' "$READY"
    exit 0
  fi
  sleep 1
done
echo "FATAL: no quarantine marker after ${TIMEOUT}s. Refusing to let VRAM"
echo "workloads start on a card whose defective cell is not isolated."
exit 1
