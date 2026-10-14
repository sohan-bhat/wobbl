#!/usr/bin/env sh
# A 54 mm KNSB motor of five BATES segments in a 6061-T6 casing, checked for chamber pressure and erosive burning.
set -e
cd "$(dirname "$0")"
wobbl motor motor.toml \
  --question "Will this 54 mm KNSB motor hold its casing, and what thrust curve does it give?" -o output
