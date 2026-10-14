#!/usr/bin/env sh
# A carbon fiber drone arm, a 25 mm tube of eight T300/5208 plies, clamped 25 mm deep in the hub, with 120 N
# up at the motor mount: three times the motor's hover thrust, for a hard climb or a gust.
set -e
cd "$(dirname "$0")"
wobbl simulate arm.py "Is this carbon fiber arm strong enough for 120 N at the motor?" --physics composite \
  --layup "[0/±45/0]s" --ply t300-5208 --fix "box -1,-13,-13,25,13,13" --load "120 N +z at 400,0,0 @ x=max rigid" \
  --no-open -o output
