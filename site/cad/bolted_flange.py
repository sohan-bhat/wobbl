"""The bolted flange's hardware in CAD (examples/bolted-flange): four M8 x 40 class 8.8 hex bolts with rolled threads,
their washers and nuts, the full face fibre gasket between the two flanges, the weld neck pipe the pull comes in
through, and the fixture plate the lower flange sits on, pocketed for the nuts and screwed down to the machine table.

The run is drawn by the scene player, so each bolt model replaces one of the scene's bolts in its own frame (the
heads are part of it, so the scene's separate heads are hidden) and the plate replaces the scene's plate.

    python site/cad/bolted_flange.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Polyline, Pos, Rot, Solid, chamfer, fillet, make_face, revolve  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import place, plain  # noqa: E402

KEY = 'bolted-flange'
TOP = 28.0              # the upper flange's top face; the joint face is z = 14 and the lower flange sits on z = 0
PCD = 90.0
OD, BORE = 120.0, 50.0
PLATE_T = 7.2
looks.add('gasket_fibre', color=(0.05, 0.06, 0.055), metal=0.0, rough=0.85)
looks.add('weld_bead', color=(0.18, 0.17, 0.16), metal=1.0, rough=0.55)


def bolt_stack(x, y):
    """One bolt through both flanges: washer and head on top, washer and nut under the lower flange."""
    wt = fs.WASHER['M8'][2]
    m = fs.HEX['M8'][2]
    parts = [(Pos(0, 0, TOP + wt) * fs.hex_bolt('M8', 40), 'steel_zinc'), (Pos(0, 0, TOP) * fs.washer('M8'), 'steel_zinc'),
             (Pos(0, 0, -wt) * fs.washer('M8'), 'steel_zinc'), (Pos(0, 0, -wt - m) * fs.hex_nut('M8'), 'steel_zinc')]
    return place(parts, Pos(x, y, 0))


def gasket():
    """A 1 mm full face fibre gasket, its edge standing just proud of the flanges' rims."""
    g = Pos(0, 0, 13.5) * (Solid.make_cylinder(OD / 2 + 0.6, 1.0) - Solid.make_cylinder(BORE / 2 + 0.5, 1.0))
    for k in range(4):
        a = math.radians(45 + 90 * k)
        g = g - Pos(PCD / 2 * math.cos(a), PCD / 2 * math.sin(a), 13) * Solid.make_cylinder(4.5, 2)
    return [(g, 'gasket_fibre')]


def pipe():
    """The weld neck pipe the pull comes in through: 60.3 x 5.15 mm, welded to the upper flange with a fillet bead."""
    ro, ri, L = 30.15, 25.0, 40.0
    prof = [(ri, TOP), (ro, TOP), (ro, TOP + L - 1.0), (ro - 1.0, TOP + L), (ri, TOP + L)]
    p = revolve(make_face(Plane.XZ * Polyline(*prof, close=True)), Axis.Z)
    bead = revolve(make_face(Plane.XZ * Polyline((ro, TOP), (ro + 6.0, TOP), (ro + 5.0, TOP + 1.6), (ro + 2.5, TOP + 4.6), (ro, TOP + 6.0), close=True)), Axis.Z)
    return [(p, 'steel_mill'), (bead, 'weld_bead')]


def fixture_plate():
    """The plate under the lower flange: the bore, pockets the nuts sit through, and four cap screws into the
    machine table."""
    p = Pos(0, 0, -PLATE_T / 2) * Box(168, 168, PLATE_T)
    p = fillet(p.edges().filter_by(Axis.Z), 8.0)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 0.8)
    p = p - Pos(0, 0, -10) * Solid.make_cylinder(BORE / 2, 20)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        p = p - Pos(PCD / 2 * math.cos(a), PCD / 2 * math.sin(a), -10) * Solid.make_cylinder(11.0, 20)
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = p - Pos(sx * 70, sy * 70, -10) * Solid.make_cylinder(4.5, 20)
            out.append((Pos(sx * 70, sy * 70, 0) * fs.socket_cap('M8', 20, threads=False), 'steel_black_oxide'))
    return [(p, 'steel_mill')] + out


def main():
    sc = ex.scene(f'{KEY}/output')
    bolts = sc.named('M8 8.8')
    bi, bo, bM = bolts[0]
    x, y = bM[0][3], bM[1][3]
    Mb = ex.pose_mm(sc, bi)
    pi = sc.named('plate')[0][0]
    Mp = ex.pose_mm(sc, pi)
    files = {
        'm8-bolt': ([Node('m8-bolt', plain(ex.into(bolt_stack(x, y), Mb)), tol=0.05, ang=0.35)], {'object': 'M8 8.8'}),
        'plate': ([Node('fixture-plate', plain(ex.into(fixture_plate(), Mp)), tol=0.1, ang=0.3)], {'object': 'plate'}),
        'gasket-and-pipe': ([Node('gasket', gasket(), tol=0.1, ang=0.25), Node('weld-neck-pipe', pipe(), tol=0.08, ang=0.2)], {}),
    }
    ex.publish(KEY, files, extra={'hide': ['M8 8.8 head']})


if __name__ == '__main__':
    main()
