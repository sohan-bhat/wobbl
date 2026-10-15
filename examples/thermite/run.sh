#!/usr/bin/env sh
# Thermite, aluminium and rust: how much heat it releases and how hot the products get.
set -e
cd "$(dirname "$0")"
wobbl chem thermite "how much heat does it release and how hot does it get?" --no-open -o output
