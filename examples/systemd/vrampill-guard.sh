#!/bin/bash
# Locate the target GPU by BOARD SERIAL and quarantine its defective VRAM cells.
#
# Resolving by serial rather than CUDA index is deliberate: the index changes
# whenever cards are moved between slots. The serial does not.
#
# If the target card is not installed at all (e.g. it is away for RMA), this
# writes the marker and exits 0, so dependent services are NOT blocked. The
# quarantine is only needed when the defective card is actually present.
#
# Required:
#   BAD_SERIAL   board serial of the defective card
#                (find it with: nvidia-smi --query-gpu=index,serial --format=csv)
set -u

: "${BAD_SERIAL:?set BAD_SERIAL to the defective card's board serial}"
READY="${READY:-/run/user/$(id -u)/vrampill.ready}"
PILL="${PILL:-$HOME/gpu-vram-quarantine/vrampill}"
CHUNK_MIB="${CHUNK_MIB:-8}"
FIND_SECONDS="${FIND_SECONDS:-1800}"
QUIET_SECONDS="${QUIET_SECONDS:-180}"

rm -f "$READY"

if [ ! -x "$PILL" ]; then
  echo "FATAL: $PILL missing or not executable"
  exit 5
fi

# On cold boot systemd can beat the driver to readiness.
for _ in $(seq 1 30); do
  nvidia-smi -L >/dev/null 2>&1 && break
  sleep 2
done

MAP=$(nvidia-smi --query-gpu=index,serial --format=csv,noheader 2>/dev/null)
if [ -z "$MAP" ]; then
  echo "FATAL: nvidia-smi returned no GPUs"
  exit 6
fi
echo "GPUs present:"
echo "$MAP" | sed 's/^/  /'

IDX=$(echo "$MAP" | awk -F', *' -v s="$BAD_SERIAL" '$2==s {print $1; exit}')

if [ -z "$IDX" ]; then
  echo "Card with serial $BAD_SERIAL is NOT installed. No quarantine required."
  { echo "status=card-absent"; echo "serial=$BAD_SERIAL"; } > "$READY"
  exit 0
fi

echo "Target card found at GPU index $IDX. Starting quarantine."
exec "$PILL" --device "$IDX" --chunk-mib "$CHUNK_MIB" \
     --find-seconds "$FIND_SECONDS" --quiet-seconds "$QUIET_SECONDS" \
     --ready-file "$READY"
