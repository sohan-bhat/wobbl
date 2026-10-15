#!/usr/bin/env sh
# Copper as a crystal: its structure, lattice constant, density, stiffness and cohesive energy.
set -e
cd "$(dirname "$0")"
wobbl simulate copper "what is its crystal structure and how stiff is it?" --no-open -o output
