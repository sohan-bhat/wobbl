#!/usr/bin/env sh
# A sand cast A356-T6 motor mount for a belt conveyor: does it meet a safety factor of 2 on yield?
# The part ships with Wobbl as a validation part.
set -e
cd "$(dirname "$0")"
PART="../../src/wobbl/validation/parts/cast_motor_mount.step"
wobbl run "$PART" -m "custom:E=72400,nu=0.33,yield=165" --target-sf 2 --fix "z=min" \
  --load "2000 N -0.8660254,0,-0.5 at 0,-30,66 @ holes d=6.6" \
  --question "Does this cast motor mount meet our safety factor 2 rule under a 2 kN emergency stop belt pull?" \
  --budget 600 -o output
