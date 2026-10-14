#!/usr/bin/env sh
# How far can simulation stand in for testing a cast motor mount, and which physical tests does it still need?
# The strength study, then its probability of failure under load and material scatter, a comparison with a small
# synthetic strain gauge dataset (made from the model, not from a real test), a credibility assessment and a test plan.
set -e
cd "$(dirname "$0")"
PART="../../src/wobbl/validation/parts/cast_motor_mount.step"
wobbl run "$PART" -m "custom:E=72400,nu=0.33,yield=165,process=cast" --target-sf 2 --fix "z=min" \
  --load "2000 N -0.8660254,0,-0.5 at 0,-30,66 @ holes d=6.6" --no-images \
  --question "How far can simulation stand in for testing this cast motor mount under a 2 kN belt pull?" \
  --budget 600 -o output
wobbl uncertainty output --vary "load lognormal 10%" --vary "yield 8%" --vary "E 3%"
wobbl validate-against output readings.csv
wobbl credibility output --influence high --consequence medium
