#!/usr/bin/env sh
# A 7075-T6 bracket whose lug is pulled at 30 degrees to its arm, with a 1.27 mm crack at the pin hole.
set -e
cd "$(dirname "$0")"
wobbl crack bracket.py -m 7075-t6 --fix "bolt holes d=6.6" --load "4 kN 0.866,0.5,0 bearing @ hole d=8" \
  --crack "through 1.27 mm at hole d=8" --r-ratio 0.1 --target-life 100000 --steps 14 --budget 3600 \
  --question "Where does a crack at the pin hole go, and do 99% of brackets last 100,000 cycles?" -o output
