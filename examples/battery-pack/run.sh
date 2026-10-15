#!/usr/bin/env sh
# A 2S2P 18650 pack making 6 W inside a sealed plastic box at 35 C, over a 60 minute discharge.
set -e
cd "$(dirname "$0")"
wobbl simulate pack.py "Will this battery pack overheat in a sealed box during a one hour discharge?" --physics heat \
  -m li-ion-cell --power "6 W" --film "5 W/m2K @ all" --ambient "35 C" \
  --radiation 0.9 --duration "60 min" --limit "60 C" -o output
