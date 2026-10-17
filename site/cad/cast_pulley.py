"""The pulley's sand mold set up for the pour in CAD (examples/cast-pulley): the no bake mold standing on its steel
bottom board, carried on two box section skids.

Built in the run's millimetre frame round the scene's mold block and added as it stands; the block itself stays
the see through sand the run draws. The page's camera sits inside the mold, so a pouring cup or weights on the
cope would only float overhead in that view, and are left out.

    python site/cad/cast_pulley.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Polyline, Pos, RectangleRounded, Rot, Solid, chamfer, extrude, fillet, make_face, revolve  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import plain  # noqa: E402

KEY = 'cast-pulley'


def mold_box(sc):
    i, o, M = sc.named('sand mold')[0]
    hx, hy, hz = o['size']
    c = M[:3, 3]
    return c, (hx, hy, hz)


def board(c, h):
    hx, hy, hz = h
    z0 = c[2] - hz
    b = Pos(c[0], c[1], z0 - 6.0) * Box(2 * hx + 40, 2 * hy + 40, 12.0)
    b = chamfer(b.edges().group_by(Axis.Z)[-1], 1.5)
    out = [(b, 'steel_mill')]
    for s in (-1, 1):
        rail = Pos(c[0] + s * (hx - 20), c[1], z0 - 12.0 - 25.0) * Box(50, 2 * hy + 40, 50)
        rail = rail - Pos(c[0] + s * (hx - 20), c[1], z0 - 12.0 - 25.0) * Box(42, 2 * hy + 42, 42)
        out.append((rail, 'paint_black'))
    return out


def main():
    sc = ex.scene(f'{KEY}/output')
    c, h = mold_box(sc)
    files = {
        'bottom-board': ([Node('bottom-board', plain(board(c, h)), tol=0.2, ang=0.3)], {}),
    }
    ex.publish(KEY, files)


if __name__ == '__main__':
    main()
