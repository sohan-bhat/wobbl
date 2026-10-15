#!/usr/bin/env sh
# A sealed polycarbonate handheld housing with a 45 g battery in its cradle, dropped 1.2 m onto concrete. Wobbl
# tries every face and corner landing and keeps the worst.
set -e
cd "$(dirname "$0")"
wobbl simulate handheld.py "Will it survive a 1.2 m drop onto concrete with its 45 g battery?" --physics drop \
  -m pc --mass "45 g @ 0,0,6 r=26" --no-open -o output
