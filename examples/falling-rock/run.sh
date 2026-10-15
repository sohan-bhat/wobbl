#!/usr/bin/env sh
# A fist sized granite rock, 9 cm across, dropped from the top of a 300 m cliff.
set -e
cd "$(dirname "$0")"
wobbl simulate "a 9 cm rock dropped from 300 m" "How fast does a fist sized rock hit the ground after falling off a 300 m cliff?" \
  --physics trajectory --name rock --diameter "9 cm" --density "2700 kg/m3" --cd 0.6 --height "300 m" -o output
