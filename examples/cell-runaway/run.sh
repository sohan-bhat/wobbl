#!/usr/bin/env sh
# Can a charged 18650 in the battery pack example run away on its own at the pack's hottest point?
set -e
cd "$(dirname "$0")"
wobbl simulate "lithium ion 18650 cell" "Can a cell at the hottest point of the sealed battery pack run away on its own?" \
  --physics runaway --film "sealed box" --from ../battery-pack/output -o output
