#!/usr/bin/env sh
# Printed PLA shelf bracket, screwed to a wall, carrying its share of an 8 kg shelf.
set -e
cd "$(dirname "$0")"
wobbl simulate bracket.py "Will this printed PLA shelf bracket hold its share of an 8 kg shelf?" --physics stress -m pla \
  --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6" -o output
wobbl fix output --apply
