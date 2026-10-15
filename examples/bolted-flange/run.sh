#!/usr/bin/env sh
# Two steel flanges held by four M8 bolts at 20 N m, pulled apart with 30 kN.
set -e
cd "$(dirname "$0")"
wobbl simulate flange.py "Do four M8 bolts at 20 N m keep this flange closed under a 30 kN pull?" --physics bolts \
  -m mild-steel --fix "body 1 and bottom" --load "30 kN +z @ body 2 and top" --bolt "M8 8.8 torque 20 N*m" \
  --quality fast --set playback=true --no-open -o output
