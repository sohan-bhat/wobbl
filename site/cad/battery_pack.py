"""Context models for the 2S2P pack in a sealed box (examples/battery-pack): the four 18650s the effective block
stands for, in their wraps with positive caps and insulator rings, the nickel strips with their spot welds (the two
end strips on top that carry the pack's leads, the series strip below), silicone leads to an XT60 and a JST-XH
balance lead. The block keeps its temperature colors; the hardware over it is drawn as an X-ray.

    python site/cad/battery_pack.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import battery as bt  # noqa: E402
from build123d import Pos  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech.util import place  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'battery-pack', 'output', 'scene.wobbl')


def main():
    sc = Scene(RUN)
    cells = []
    for k in range(1, 5):
        i, o, M = sc.named(f'cell {k}')[0]
        cells.append((M[0, 3], M[1, 3], M[2, 3], o['size'][0], o['size'][1]))
    r, hh = cells[0][3], cells[0][4]
    top, bot = cells[0][2] + hh + 0.25, cells[0][2] - hh - 0.25
    nodes = []
    for k, (x, y, z, _, _) in enumerate(cells):
        nodes.append(Node(f'cell_{k + 1}', place(bt.cell(r, hh), Pos(x, y, z))))
    c = [(x, y) for x, y, *_ in cells]
    # cells 1 and 2 are one parallel group, 3 and 4 the other: the pack's minus and plus on top, the series link below
    strips = bt.strip(c[0], c[1], top + 0.8) + bt.strip(c[2], c[3], top + 0.8)
    strips += bt.strip(c[0], c[2], bot - 0.3) + bt.strip(c[1], c[3], bot - 0.3)
    nodes.append(Node('strips', strips))
    zt = top + 0.9
    # the main leads leave over the far corner, away from the view, to the XT60
    m0 = (c[0][0], (c[0][1] + c[1][1]) / 2, zt + 1.2)
    m1 = (c[2][0], (c[2][1] + c[3][1]) / 2, zt + 1.2)
    wires = bt.lead([m0, (m0[0] + 2, m0[1] + 4, zt + 9), (30, 46, zt + 8), (52, 54, zt - 6), (60, 50, zt - 18)], 1.3, 'silicone_black')
    wires += bt.lead([m1, (m1[0] + 2, m1[1] + 4, zt + 7), (40, 40, zt + 6), (56, 46, zt - 8), (62, 42, zt - 18)], 1.3, 'silicone_red')
    wires += bt.xt60(64, 46, zt - 26, '-z')
    bal = [(c[0][0] + 3, c[0][1] + 6, zt + 0.6), (c[2][0] - 3, c[2][1] + 6, zt + 0.6), (c[2][0] + 3, c[3][1] - 2, zt + 0.6)]
    for k, p in enumerate(bal):
        wires += bt.lead([p, (p[0], p[1], p[2] + 6), (14 + k * 1.2, 46, zt + 8), (14 + k * 1.2, 58, zt - 4)], 0.45, 'wire_white')
    wires += bt.jst_xh(15.2, 61.0, zt - 6, 3, '+y')
    nodes.append(Node('leads', wires))
    export('battery-pack/pack.glb', nodes, tol=0.03, ang=0.3)
    write_manifest('battery-pack', {'add': [{'file': 'battery-pack/pack.glb'}]})


if __name__ == '__main__':
    main()
