"""The shelf bracket's setting in CAD (examples/shelf-bracket): a plastered masonry wall, cut away in section along
the screws so the ribbed nylon wall plugs show in their drilled holes, two 4 mm pan head wood screws through the
bracket's holes, a 10 mm clear acrylic shelf on the arm with rubber bumpers, and a stainless knob weight standing where
the run's load presses.

The strength stage draws the part in its own millimetre frame (the wall face is x = 0, the arm's top z = 6), so
these are built in that frame and added as they stand.

    python site/cad/shelf_bracket.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Pos, Rot, Solid, chamfer, fillet  # noqa: E402

import looks  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'shelf-bracket'
SCREWS = [(12.5, 25.0), (12.5, 65.0)]     # (y, z) of the bracket's holes
LEG = 6.0                                 # the bracket's leg against the wall
ARM_TOP = 6.0
PLASTER = 12.0
looks.add('brick', color=(0.33, 0.16, 0.1), metal=0.0, rough=0.85, normal='cast_n', nstr=0.6, tiles=20)
looks.add('plug_nylon', color=(0.75, 0.72, 0.62), metal=0.0, rough=0.5)


SECTION = 65.0          # the wall is drawn cut level with the upper screw, so its plug and screw show in section


def wall():
    """Plaster on brick, 60 mm thick, its top cut level with the upper screw's axis like a section drawing."""
    y0, y1, z0, z1 = -146.0, 171.0, -158.0, SECTION
    plaster = Pos(-PLASTER / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(PLASTER, y1 - y0, z1 - z0)
    brick = Pos(-PLASTER - 24.0, (y0 + y1) / 2, (z0 + z1) / 2) * Box(48.0, y1 - y0, z1 - z0)
    for y, z in SCREWS:
        hole = Pos(0.5, y, z) * Rot(0, -90, 0) * Solid.make_cylinder(3.1, 45.0)
        plaster = plaster - hole
        brick = brick - hole
    return [(plaster, 'paint_wall'), (brick, 'brick')]


def fixings():
    out = []
    keep = Pos(-50.0, 12.5, SECTION + 20.0) * Box(100.0, 40.0, 40.0)     # what the section takes off inside the wall
    for y, z in SCREWS:
        M = frame(np.array([0.0, y, z]), [0, 1, 0], [1, 0, 0])       # z along +x, out of the wall
        parts = place([(fs.wall_plug(6.0, 30.0), 'plug_nylon')], M)
        Ms = frame(np.array([LEG, y, z]), [0, 1, 0], [1, 0, 0])
        parts += place([(fs.pan_phillips(4.0, 40.0, head_d=8.0, wood=True), 'steel_zinc')], Ms)
        if z >= SECTION:
            parts = [(s_ - keep, m) for s_, m in parts]
        out += parts
    return out


def shelf():
    """A 10 mm clear acrylic shelf, its edges polished, on two rubber bumpers on the arm."""
    s = Pos(48.0, 12.5, ARM_TOP + 0.8 + 5.0) * Box(84.0, 288.0, 10.0)
    s = fillet(s.edges(), 1.2)
    out = [(s, 'acrylic')]
    for x in (30.0, 72.0):
        b = Pos(x, 12.5, ARM_TOP) * Solid.make_cylinder(4.0, 0.8)
        out.append((b, 'rubber'))
    return out


def weight():
    """A stainless OIML knob weight standing on the shelf over the load point, 65 mm out from the wall."""
    top = ARM_TOP + 0.8 + 10.0
    prof = [(0, 0), (14.0, 0), (15.0, 1.0), (15.0, 27.0), (14.0, 28.0), (8.0, 29.5), (6.0, 30.2), (5.6, 31.0), (7.6, 33.0),
            (8.6, 35.0), (8.0, 37.6), (6.2, 38.8), (0, 39.0)]
    from build123d import Polyline, make_face, revolve
    body = revolve(make_face(Plane.XZ * Polyline(*prof, close=True)), Axis.Z)
    return [(Pos(65.0, 12.5, top) * body, 'steel_stainless')]


def main():
    files = {
        'wall': ([Node('wall', wall(), tol=0.2, ang=0.4)], ['wall']),
        'screws': ([Node('wood-screws-and-plugs', plain(fixings()), tol=0.05, ang=0.4)], ['bolt', 'bolt head']),
        'shelf': ([Node('acrylic-shelf', plain(shelf()), tol=0.1, ang=0.3)], ['shelf']),
        'weight': ([Node('weight', plain(weight()), tol=0.05, ang=0.25)], ['load on the shelf']),
    }
    entry, total = [], 0
    for slug, (nodes, covers) in files.items():
        rel = f'{KEY}/{slug}.glb'
        total += export(rel, nodes, unit=1.0)
        entry.append({'file': rel, 'covers': covers})
    write_manifest(KEY, {'add': entry})
    print(f'[cad] {KEY} {total / 1024:.0f} KB')


if __name__ == '__main__':
    main()
