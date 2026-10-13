#!/usr/bin/env sh
# Regenerate every example output. Takes a while: each strength study refines until it converges.
set -e
cd "$(dirname "$0")"
for d in cast-motor-mount shelf-bracket motor-mount enclosure drone-arm heat-sink battery-pack circuits; do
  echo "== $d"
  sh "$d/run.sh"
done
