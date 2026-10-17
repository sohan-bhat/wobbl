"""The cracked bracket's setting in CAD (examples/bracket-crack, both its stress and its crack panes): two M6 x 20
cap screws with washers and nuts through a 10 mm aluminium mounting plate that stops short of the lug, and the
loading clevis on the lug, its 8 mm pin with head, washer and retaining clip, and the M10 rod pulling at 30 degrees
to the arm in the bracket's plane.

The run is drawn by the scene player: the screw model replaces the scene's bolts (heads included, so the scene's
heads are hidden), the plate replaces the scene's plate, and the clevis is added.

    python site/cad/bracket_crack.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Circle, Plane, Pos, Rectangle, Rot, Solid, chamfer, extrude, fillet  # noqa: E402

from gltf import Node  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'bracket-crack'
T = 5.0                         # the bracket's thickness
PIN = np.array([90.0, 0.0, T / 2])
PULL = np.array([math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0])
PLATE_T = 10.0


def screw(x, y):
    w = fs.WASHER['M6'][2]
    m = fs.HEX['M6'][2]
    parts = [(Pos(0, 0, T + w) * fs.socket_cap('M6', 20, threads=True), 'steel_black_oxide'), (Pos(0, 0, T) * fs.washer('M6'), 'steel_zinc'),
             (Pos(0, 0, -PLATE_T - w) * fs.washer('M6'), 'steel_zinc'), (Pos(0, 0, -PLATE_T - w - m) * fs.hex_nut('M6'), 'steel_zinc')]
    return place(parts, Pos(x, y, 0))


def plate():
    p = Pos(15.0, 0, -PLATE_T / 2) * Box(110, 96, PLATE_T)
    p = fillet(p.edges().filter_by(Axis.Z), 6.0)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 0.6)
    for y in (-15.0, 15.0):
        p = p - Pos(12.0, y, -20) * Solid.make_cylinder(3.3, 40)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = p - Pos(15.0 + sx * 45, sy * 38, -20) * Solid.make_cylinder(3.3, 40)
    out = [(p, 'aluminum_machined')]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((Pos(15.0 + sx * 45, sy * 38, 0) * fs.socket_cap('M6', 16, threads=False, lite=True), 'steel_black_oxide'))
    return out


def clevis():
    """A steel clevis round the lug: two cheeks either side of it joined by the body, the pin, and the rod."""
    out = []
    gap = 0.4
    ct = 4.0
    cheeks = None
    for s in (-1, 1):
        z0 = T / 2 + gap if s > 0 else -T / 2 - gap - ct
        f = Pos(12.0, 0) * Rectangle(24.0, 20.0) + Circle(10.0)
        f = f - Circle(4.1)
        c = Pos(0, 0, z0) * extrude(f, ct)
        c = fillet(c.edges().filter_by(Axis.Z, reverse=True).group_by(Axis.Z)[-1 if s > 0 else 0], 0.8)
        cheeks = c if cheeks is None else cheeks + c
    body = Pos(28.0, 0, 0) * Box(10.0, 20.0, T + 2 * gap + 2 * ct)
    body = fillet(body.edges().filter_by(Axis.X), 2.0)
    boss = Pos(33.0, 0, 0) * Rot(0, 90, 0) * Solid.make_cylinder(8.0, 8.0)
    out.append((cheeks + body + boss, 'steel_zinc'))
    rod = Pos(36.0 + 70.0, 0, 0) * Rot(0, 90, 0) * fs.threaded_rod(10.0, 70.0, 1.5, 0.0, True)
    out.append((rod, 'steel_bare'))
    out.append((Pos(41.0, 0, 0) * Rot(0, 90, 0) * fs.hex_nut('M10'), 'steel_zinc'))
    top = T / 2 + gap + ct
    pin = Pos(0, 0, -top - 3.0) * Solid.make_cylinder(4.0, 2 * top + 3.0)
    pin = chamfer(pin.edges().group_by(Axis.Z)[0], 0.4)
    head = Pos(0, 0, top) * Solid.make_cylinder(7.0, 3.0)
    head = chamfer(head.edges().group_by(Axis.Z)[-1], 0.5)
    out += [(pin + head, 'steel_bare'), (Pos(0, 0, -top - 1.0) * fs.washer('M8', thickness=1.0), 'steel_zinc')]
    clip = Pos(0, 0, -top - 2.2) * (Solid.make_cylinder(5.6, 0.8) - Solid.make_cylinder(3.5, 0.8) - Pos(4.0, 0, 0.4) * Box(4.0, 4.6, 2.0))
    out.append((clip, 'steel_black_oxide'))
    return place(out, frame(PIN, PULL, [0, 0, 1]))


def main():
    sc = ex.scene(f'{KEY}/output/stress')
    bi, _, Mb0 = sc.named('bolt')[0]
    Mb = ex.pose_mm(sc, bi)
    pi = sc.named('mounting plate')[0][0]
    Mp = ex.pose_mm(sc, pi)
    files = {
        'screw': ([Node('m6-screw', plain(ex.into(screw(Mb[0][3], Mb[1][3]), Mb)), tol=0.05, ang=0.35)], {'object': 'bolt'}),
        'mounting-plate': ([Node('mounting-plate', plain(ex.into(plate(), Mp)), tol=0.08, ang=0.3)], {'object': 'mounting plate'}),
        'clevis': ([Node('loading-clevis', plain(clevis()), tol=0.06, ang=0.3)], {}),
    }
    ex.publish(KEY, files, extra={'hide': ['bolt head']})


if __name__ == '__main__':
    main()
