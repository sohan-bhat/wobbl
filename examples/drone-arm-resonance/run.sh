#!/usr/bin/env sh
# The printed PA12 drone arm carrying an 80 g motor and propeller, checked for resonance from hover to full throttle.
set -e
cd "$(dirname "$0")"
PAD="top and within 15 of 150,0,15"
wobbl simulate ../drone-arm/arm.py "Will this printed arm resonate with its propeller anywhere from hover to full throttle?" --physics vibration \
  -m pa12 --fix "bolt holes d=3.2 and x<0" \
  --mass "80 g at 150,0,32 @ $PAD" --rpm "5000-12000 rpm" --blades 2 -o output
