#!/usr/bin/env sh
# A liquid cooled plate: 150 W of electronics on its underside and 2 L/min of water through its drilled channel.
# Wobbl finds the two ports, solves the water's flow through the channel, puts its heat transfer on the channel
# walls and works out the plate's temperature, with the pressure the pump has to supply.
set -e
cd "$(dirname "$0")"
wobbl simulate plate.py \
  "Does it stay under 60 C with 150 W on the bottom and 2 L/min of water at 25 C flowing through it?" \
  --physics cooling -m 6061-t6 --set heat-faces=bottom --no-open -o output
