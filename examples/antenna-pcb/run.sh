#!/usr/bin/env sh
# An inverted F trace antenna on a 40 x 30 mm FR4 board: its resonance, S11, gain and pattern from a 3D full wave solve.
set -e
cd "$(dirname "$0")"
python make_board.py
wobbl simulate ifa.kicad_pcb "Does the trace antenna resonate at 2.4 GHz, and what is its S11 and gain?" \
  --physics antenna --set frequency=2.4GHz -o output
