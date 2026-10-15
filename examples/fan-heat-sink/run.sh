#!/usr/bin/env sh
# The extruded heat sink from ../heat-sink, now with a small fan blowing along its fins in a 40 C cabinet. The
# airflow through the fins is solved in 3D, its heat transfer goes onto every face, and CalculiX finds the chip's
# temperature.
set -e
cd "$(dirname "$0")"
wobbl simulate heatsink.py \
  "Does a 30 W chip stay under 85 C with a 10 CFM, 20 Pa fan blowing along the fins in a 40 C cabinet?" \
  -m 6063-t5 --set heat-faces="bottom and box 17.5,17.5,-1,42.5,42.5,0.5" --set direction=+y \
  --set mode=ducted --set temperature="40 C" --no-open -o output
