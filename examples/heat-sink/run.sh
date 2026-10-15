#!/usr/bin/env sh
# A 15 W chip on an extruded aluminum heat sink in a 40 C cabinet with no fan.
set -e
cd "$(dirname "$0")"
wobbl simulate heatsink.py "Will a 15 W chip stay under 85 C on this heat sink without a fan?" --physics heat \
  -m 6063-t5 --power "15 W" --heat-faces "bottom and box 17.5,17.5,-1,42.5,42.5,0.5" \
  --film "6 W/m2K @ all" --ambient "40 C" --radiation 0.85 --limit "85 C" -o output
