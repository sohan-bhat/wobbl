#!/usr/bin/env sh
# A pull solenoid exported from CAD as three solids: a square steel frame with a pole piece (body 1), a copper coil
# (body 2) and a steel plunger 3 mm short of the pole (body 3). Wobbl solves its magnetic field in 3D.
set -e
cd "$(dirname "$0")"
wobbl simulate actuator.step "does the plunger pull 5 N at 0.8 A?" \
  --assign "body 1: steel 1010" --assign "body 2: coil 1100 turns 0.8 A" --assign "body 3: steel 1010" \
  --force-on "body 3" --no-open -o output
