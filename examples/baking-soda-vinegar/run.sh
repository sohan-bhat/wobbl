#!/usr/bin/env sh
# Baking soda dropped into vinegar: the reaction, its heat, the gas it gives off and how much it cools.
set -e
cd "$(dirname "$0")"
wobbl simulate "baking soda and vinegar" "how much gas does it make and does it get warm or cold?" --no-open -o output
