#!/usr/bin/env sh
# A propane flame in air at the stoichiometric mixture, 1 atm and 25 C, burned to equilibrium.
set -e
cd "$(dirname "$0")"
wobbl simulate propane "How hot does a propane flame in air get, and how much CO and NO does it make?" --physics flame \
  --phi 1 --pressure "1 atm" --temperature "25 C" -o output
