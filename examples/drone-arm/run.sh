#!/usr/bin/env sh
# PA12 drone arm at full throttle: 25 N of thrust and 0.5 N m of motor torque on the pad.
set -e
cd "$(dirname "$0")"
PAD="top and within 15 of 150,0,15"
wobbl simulate arm.py "Is this printed arm strong enough at full throttle, with a safety factor of 3?" --physics stress \
  -m pa12 --target-sf 3 --fix "bolt holes d=3.2 and x<0" \
  --load "25 N up at 150,0,15 @ $PAD rigid" --load "0.5 N*m +z at 150,0,15 @ $PAD rigid" --budget 600 -o output
