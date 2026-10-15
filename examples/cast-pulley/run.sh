#!/usr/bin/env sh
# A sand cast A356 belt pulley, lying flat: does it cast sound, or does the hub starve and leave shrinkage porosity?
set -e
cd "$(dirname "$0")"
wobbl simulate pulley.py "Will the hub of this sand cast aluminum pulley shrink and leave porosity?" \
  --physics casting --material a356 --no-open -o output
