#!/usr/bin/env sh
# Ten years of starts and running for the cast motor mount, from its finished strength result.
set -e
cd "$(dirname "$0")"
wobbl fatigue ../cast-motor-mount/output --block "100% x 3650 R=0" --block "70% x 1e8 R=0.714" --repeats 10 \
  --question "Does the cast motor mount last ten years of starts and running without a fatigue crack?" -o output
