"""Context models for the buck converter boards whose switch node rings (examples/pcb-buck-ringing): a screw
terminal block with its captive screws, the electrolytic cans in their sleeves with vented tops, the switcher as a
SOIC with gull wing leads and the wound toroid. The copper keeps its live current colors, and so does the diode.

Both layouts are one example to the runtime (models are listed per example, not per run), so only parts that sit
the same way on both boards are dressed; the two boards differ in size and keep the scene's own board.

    python site/cad/pcb_buck_ringing.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import electronics as el  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

RUNS = os.path.join(HERE, '..', '..', 'examples', 'pcb-buck-ringing', 'output')


def local(sc, i, pts):
    P = np.c_[np.asarray(pts, dtype=np.float64), np.ones(len(pts))]
    return (P @ sc.inverse(i).T)[:, :3]


def main():
    shared = False
    for run in ('tight',):
        sc = Scene(os.path.join(RUNS, run, 'scene.wobbl'))
        if not shared:
            _, jo, _ = sc.named('J1')[0]
            export('pcb-buck-ringing/terminal.glb', [Node('terminal', el.screw_terminal(*jo['size'], 2))], tol=0.01, ang=0.3)
            _, so, _ = sc.named('J1 screw')[0]
            export('pcb-buck-ringing/terminal-screw.glb', [Node('screw', el.slotted_screw(*so['size']))], tol=0.005, ang=0.3)
            _, co, _ = sc.named('C1')[0]
            export('pcb-buck-ringing/can.glb', [Node('can', el.radial_can(*co['size']))], tol=0.01, ang=0.25)
            i, po, PM = sc.named('S1')[0]
            board_z = local(sc, i, [[0, 0, 1.65]])[0][2]
            export('pcb-buck-ringing/soic8.glb', [Node('soic8', el.soic(*po['size'], board_z))], tol=0.01, ang=0.3)
            shared = True
        # the toroid's leads stand 2.5 mm either side of its centre on both boards
        i, lo, LM = sc.named('L1 coil')[0]
        _, core, _ = sc.named('L1')[0]
        R, r = core['size']
        leads = [local(sc, i, [m[:3, 3] + m[:3, 2] * o['size'][1]])[0] for _, o, m in sc.named('L1 lead')]
        export('pcb-buck-ringing/winding.glb',
               [Node('winding', el.toroid_winding(R, r, (lo['size'][1] - r) * 1.6, 17, gap_at=-90, gap=70, leads=leads))],
               tol=0.01, ang=0.35)
        rep = [
            {'object': 'J1', 'file': 'pcb-buck-ringing/terminal.glb'},
            {'object': 'J1 screw', 'file': 'pcb-buck-ringing/terminal-screw.glb'},
            {'object': 'C1', 'file': 'pcb-buck-ringing/can.glb'},
            {'object': 'C2', 'file': 'pcb-buck-ringing/can.glb'},
            {'object': 'C1 top', 'file': 'circuits-bench/cap-top-3.04.glb'},
            {'object': 'C2 top', 'file': 'circuits-bench/cap-top-3.04.glb'},
            {'object': 'S1', 'file': 'pcb-buck-ringing/soic8.glb'},
            {'object': 'L1', 'file': 'circuits-bench/toroid-core.glb'},
            {'object': 'L1 coil', 'file': 'pcb-buck-ringing/winding.glb'},
        ]
        write_manifest('pcb-buck-ringing', {'replace': rep})


if __name__ == '__main__':
    main()
