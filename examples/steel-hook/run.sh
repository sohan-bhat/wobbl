#!/usr/bin/env sh
# A mild steel J hook bolted up by its eye, with a 3 kN sling in the bottom of the bend.
set -e
cd "$(dirname "$0")"
wobbl plastic hook.py -m mild-steel --fix "bolt holes d=8.5" --load "3 kN -z @ within 5 of 22,0,-16" --playback \
  --question "Will this steel hook bend for good under a 3 kN sling, and what load pulls it open?" -o output
