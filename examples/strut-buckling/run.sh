#!/usr/bin/env sh
# A 600 mm aluminum tube strut, clamped at its base and held sideways at the top, pushed with 7 kN.
set -e
cd "$(dirname "$0")"
wobbl simulate strut.py "Does this aluminum strut carry 7 kN with a factor of 3 against buckling?" --physics buckling \
  -m 6061-t6 --fix "z=min" --fix "fixed-xy z=max" --load "7 kN -z @ z=max" --target-sf 3 -o output
