"""The enclosure's test setting in CAD (examples/enclosure): the shell's mating base half held on an aluminium fixture
plate by four toe clamps on its flange, the four screws up through the base into the shell's bosses, and the thumb
simulant pressing near one corner, a 16 mm rubber pad on a stem under a button load cell on the actuator's rod.

Built in the part's millimetre frame (the shell's rim on z = 0, its top at z = 26.5) and added as it stands.

    python site/cad/enclosure.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from build123d import Axis, Box, Pos, RectangleRounded, Rot, Solid, chamfer, extrude, fillet  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import cable  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import plain  # noqa: E402

KEY = 'enclosure'
LX, LY, RC = 87.0, 57.0, 6.0
TOP = 26.5
PRESS = (25.0, 15.0)
BASE_H = 14.0
looks.add('pc_grey', color=(0.32, 0.33, 0.34), metal=0.0, rough=0.4)
looks.add('cable_grey', color=(0.12, 0.12, 0.13), metal=0.0, rough=0.5)


def base_half():
    """The base half the shell closes on: same outline, 2.5 mm walls, a floor and four screw holes."""
    outer = Pos(0, 0, -BASE_H) * extrude(RectangleRounded(LX, LY, RC), BASE_H)
    inner = Pos(0, 0, -BASE_H + 2.0) * extrude(RectangleRounded(LX - 5.0, LY - 5.0, RC - 2.5), BASE_H)
    b = outer - inner
    b = fillet(b.edges().group_by(Axis.Z)[0], 1.0)
    flange = Pos(0, 0, -BASE_H) * extrude(RectangleRounded(LX + 8.0, LY + 8.0, RC + 4.0), 2.5)
    b = b + flange
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * 36.0, sy * 21.0
            b = b - Pos(x, y, -BASE_H - 1) * Solid.make_cylinder(1.7, 4)
            out.append((Pos(x, y, -BASE_H) * Rot(180, 0, 0) * fs.pan_phillips(3.0, 18.0, head_d=5.6), 'steel_black_oxide'))
    return [(b, 'pc_grey')] + out


def fixture():
    """The fixture plate under the base, with four toe clamps on the base's flange."""
    p = Pos(0, 0, -BASE_H - 5.0) * Box(150, 110, 10)
    p = fillet(p.edges().filter_by(Axis.Z), 5.0)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 0.8)
    out = [(p, 'aluminum_machined')]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0 = sx * (LX / 2 + 9.0)
            y0 = sy * 16.0
            clamp = Pos(x0 - sx * 2.0, y0, -BASE_H + 5.0) * Box(18, 10, 5)
            clamp = fillet(clamp.edges().filter_by(Axis.Y), 1.5)
            step = Pos(x0 + sx * 4.5, y0, -BASE_H + 1.25) * Box(5, 10, 2.5)
            out.append((clamp + step, 'steel_black_oxide'))
            out.append((Pos(x0, y0, -BASE_H + 7.5) * fs.socket_cap('M5', 16, threads=False, lite=True), 'steel_black_oxide'))
    return out


def thumb():
    """A 16 mm rubber pad on a stem, the button load cell above it and the actuator's rod."""
    x, y = PRESS
    out = []
    pad = Pos(x, y, TOP) * Solid.make_cylinder(8.0, 4.0)
    pad = fillet(pad.edges().group_by(Axis.Z)[0], 1.5)
    out.append((pad, 'rubber'))
    holder = Pos(x, y, TOP + 4.0) * Solid.make_cylinder(8.5, 5.0)
    out.append((chamfer(holder.edges().group_by(Axis.Z)[-1], 0.6), 'aluminum_machined'))
    stem = Pos(x, y, TOP + 9.0) * Solid.make_cylinder(4.0, 22.0)
    out.append((stem, 'steel_bare'))
    cell = Pos(x, y, TOP + 31.0) * Solid.make_cylinder(20.0, 14.0)
    cell = chamfer(cell.edges(), 1.0)
    out.append((cell, 'steel_stainless'))
    out.append((Pos(x, y, TOP + 45.0) * Solid.make_cylinder(6.0, 70.0), 'steel_chrome'))
    gland = Pos(x + 20.0, y, TOP + 38.0) * Rot(0, 90, 0) * fs.hex_prism(8.0, 5.0)
    out.append((gland, 'plastic_black'))
    out.append((cable.tube([(x + 25, y, TOP + 38), (x + 40, y, TOP + 40), (x + 52, y + 4, TOP + 58), (x + 56, y + 6, TOP + 90)], 2.0, seg=10), 'cable_grey'))
    return out


def main():
    files = {
        'base-half': ([Node('base-half', plain(base_half()), tol=0.06, ang=0.3)], {}),
        'fixture': ([Node('fixture', plain(fixture()), tol=0.08, ang=0.35)], {}),
        'thumb-press': ([Node('thumb-press', plain(thumb()), tol=0.06, ang=0.3)], {}),
    }
    ex.publish(KEY, files)


if __name__ == '__main__':
    main()
