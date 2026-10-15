#!/usr/bin/env sh
# Ten years of starts and running for the cast motor mount, from its finished strength result.
set -e
cd "$(dirname "$0")"
wobbl simulate ../cast-motor-mount/output "Does the cast motor mount last ten years of starts and running without a fatigue crack?" --physics fatigue \
  --block "100% x 3650 R=0" --block "70% x 1e8 R=0.714" --repeats 10 -o output
