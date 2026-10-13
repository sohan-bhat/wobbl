#!/usr/bin/env sh
# Polycarbonate enclosure top shell: a thumb pressing 120 N near one corner.
set -e
cd "$(dirname "$0")"
wobbl run enclosure.py -m pc --fix "z=min" --load "120 N down @ top and within 8 of 25,15,28" \
  --question "Does the PC shell survive a 120 N thumb press, and how far does the top dent?" --budget 600 -o output
