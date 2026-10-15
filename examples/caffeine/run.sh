#!/usr/bin/env sh
# Caffeine from its name: its 3D structure, dipole, orbitals, IR spectrum and thermochemistry.
set -e
cd "$(dirname "$0")"
wobbl simulate caffeine "what does its structure and IR spectrum look like?" --no-open -o output
