#!/usr/bin/env sh
# A long flat ABS lid printed face down on an open printer: does it stay flat, or do its corners lift?
set -e
cd "$(dirname "$0")"
wobbl simulate lid.py "Will this long ABS lid warp or lift its corners when I print it on an open printer?" \
  --material abs --no-open -o output
