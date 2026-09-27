#!/bin/bash
# Locate the target GPU by BOARD SERIAL or UUID and quarantine its defective VRAM cells.
#
# Resolving by a hardware ID rather than CUDA index is deliberate: the index
# changes whenever cards are moved between slots. Serial/UUID do not.
#
# If the target card is not installed at all (e.g. it is away for RMA), this
# writes the marker and exits 0, so dependent services are NOT blocked.
#
# Required (set ONE of):
#   BAD_SERIAL   board serial of the defective card
#   BAD_UUID     GPU UUID of the defective card (use when serial reports N/A)
#
#   Find them with:
#     nvidia-smi --query-gpu=index,serial,uuid --format=csv,noheader
set -u

# --- resolve which identifier to use ---------------------------------------
BAD_SERIAL="${BAD_SERIAL:-}"
BAD_UUID="${BAD_UUID:-}"

if [ -z "$BAD_SERIAL" ] && [ -z "$BAD_UUID" ]; then
  echo "FATAL: set BAD_SERIAL or BAD_UUID to identify the defective card"
  echo "  (nvidia-smi --query-gpu=index,serial,uuid --format=csv,noheader)"
  exit 1
fi

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

# --- pick the query column and match ----------------------------------------
if [ -n "$BAD_SERIAL" ]; then
  QUERY="index,serial"
  TARGET="$BAD_SERIAL"
  LABEL="serial"
else
  QUERY="index,uuid"
  TARGET="$BAD_UUID"
  LABEL="UUID"
fi

MAP=$(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader 2>/dev/null)
if [ -z "$MAP" ]; then
  echo "FATAL: nvidia-smi returned no GPUs"
  exit 6
fi
echo "GPUs present ($QUERY):"
echo "$MAP" | sed 's/^/  /'

IDX=$(echo "$MAP" | awk -F', *' -v t="$TARGET" '$2==t {print $1; exit}')

if [ -z "$IDX" ]; then
  echo "Card with $LABEL $TARGET is NOT installed. No quarantine required."
  { echo "status=card-absent"; echo "$LABEL=$TARGET"; } > "$READY"
  exit 0
fi

echo "Target card found at GPU index $IDX ($LABEL: $TARGET). Starting quarantine."
exec "$PILL" --device "$IDX" --chunk-mib "$CHUNK_MIB" \
     --find-seconds "$FIND_SECONDS" --quiet-seconds "$QUIET_SECONDS" \
     --ready-file "$READY"
