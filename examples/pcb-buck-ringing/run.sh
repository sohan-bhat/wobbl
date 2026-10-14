#!/usr/bin/env sh
# The same 12 V to 5 V buck laid out twice, read from its KiCad board files: a loose two layer board whose ground
# returns as tracks on the bottom, and a tight one over a ground pour. Each board's traces are solved for their
# impedance, inductance and capacitance, and buck.cir runs with and without them.
set -e
cd "$(dirname "$0")"
wobbl simulate buck_loose.kicad_pcb "how hard does the switch node ring with this layout?" --netlist buck.cir \
  --no-open -o output/loose
wobbl simulate buck_tight.kicad_pcb "how hard does the switch node ring with this layout?" --netlist buck.cir \
  --no-open -o output/tight
