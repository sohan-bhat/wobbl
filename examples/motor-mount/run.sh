#!/usr/bin/env sh
# Printed PETG motor mount for a NEMA 17 belt drive: with and without its ribs.
set -e
cd "$(dirname "$0")"
LOAD="250 N down at 0,-15,31 @ normal +y and box -21,4,10,21,6,52 rigid"
wobbl run mount.py -m petg --print-direction z --fix "bolt holes d=5.5" --load "$LOAD" \
  --question "Will this printed PETG motor mount survive a 250 N belt pull?" -o output
wobbl run mount.py --param rib=0 -m petg --print-direction z --fix "bolt holes d=5.5" --load "$LOAD" \
  --question "The same mount without its ribs" -o output-noribs
