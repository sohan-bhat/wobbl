"""The carbon arm's hub and motor mount in CAD (examples/carbon-arm): the tube's root clamped 25 mm deep in a split
aluminium hub clamp between the frame's carbon plates, two M5 cap screws closing each clamp's slit, and at the tip
a tube clamp motor mount carrying a heavy lift outrunner on its plate.

Built in the part's millimetre frame (the tube runs along x from 0 to 400, 24 mm across) and added as it stands.

    python site/cad/carbon_arm.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from build123d import Axis, Box, Pos, RectangleRounded, Rot, SlotCenterToCenter, Solid, chamfer, extrude, fillet  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech import motors  # noqa: E402
from mech.util import place, plain  # noqa: E402

KEY = 'carbon-arm'
R = 12.0
L = 400.0
looks.add('carbon_fibre', color=(0.018, 0.018, 0.02), metal=0.0, rough=0.28, normal='fine_n', nstr=0.5, tiles=80)


def tube_clamp(x0, x1, half_w, half_h, slit_down=True):
    """A split block clamp round the tube from x0 to x1: the bore, the slit and two cap screws across it."""
    blk = Pos((x0 + x1) / 2, 0, 0) * Box(x1 - x0, 2 * half_w, 2 * half_h)
    blk = fillet(blk.edges().filter_by(Axis.X), 3.0)
    blk = chamfer(blk.edges().filter_by(Axis.X, reverse=True), 0.6)
    blk = blk - Pos(x0 - 1, 0, 0) * Rot(0, 90, 0) * Solid.make_cylinder(R + 0.05, x1 - x0 + 2)
    zs = -half_h / 2 - R / 2 if slit_down else half_h / 2 + R / 2
    blk = blk - Pos((x0 + x1) / 2, 0, zs) * Box(x1 - x0 + 2, 1.2, half_h - R + 0.5 + 0.01)
    out = []
    zc = -(R + (half_h - R) / 2) if slit_down else R + (half_h - R) / 2
    for x in (x0 + (x1 - x0) * 0.27, x0 + (x1 - x0) * 0.73):
        blk = blk - Pos(x, -half_w - 1, zc) * Rot(-90, 0, 0) * Solid.make_cylinder(2.75, 2 * half_w + 2)
        blk = blk - Pos(x, -half_w - 1, zc) * Rot(-90, 0, 0) * Solid.make_cylinder(4.6, 6.0)
        out.append((Pos(x, -half_w + 5.0, zc) * Rot(90, 0, 0) * fs.socket_cap('M5', 2 * half_w - 2, threads=False, lite=True), 'steel_black_oxide'))
    return [(blk, 'aluminum_anodized')] + out


def hub():
    """The hub clamp on the root and the frame's top and bottom carbon plates it is screwed between."""
    out = tube_clamp(-15.0, 25.0, 20.0, 20.0)
    for z0, t in ((-23.0, 3.0), (20.0, 3.0)):
        f = Pos(-55.0, 0) * RectangleRounded(150.0, 110.0, 14.0)
        for x in (-95.0, -60.0):
            f = f - Pos(x, 0) * SlotCenterToCenter(20.0, 12.0)
        p = Pos(0, 0, z0) * extrude(f, t)
        out.append((chamfer(p.edges().filter_by(Axis.Z, reverse=True).group_by(Axis.Z)[-1], 0.4), 'carbon_fibre'))
    for x in (-8.0, 18.0):
        for y in (-13.0, 13.0):
            out.append((Pos(x, y, 23.0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    for x in (-120.0, -50.0):
        for y in (-44.0, 44.0):
            st = Pos(x, y, -20.0) * fs.hex_prism(6.0, 40.0)
            out.append((st, 'aluminum_machined'))
            out.append((Pos(x, y, 23.0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    return out


def motor_mount():
    """A tube clamp motor mount on the tip, its plate and a heavy lift outrunner on it."""
    out = tube_clamp(L - 30.0, L + 2.0, 19.0, 18.0)
    plate = Pos(L - 8.0, 0, 18.0) * extrude(RectangleRounded(78.0, 78.0, 12.0), 4.0)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        plate = plate - Pos(L - 8.0 + 21.2 * math.cos(a), 21.2 * math.sin(a), 10) * Solid.make_cylinder(2.2, 20)
    out.append((chamfer(plate.edges().group_by(Axis.Z)[-1], 0.6), 'aluminum_anodized'))
    for x in (L - 26.0, L - 4.0):
        for y in (-13.0, 13.0):
            out.append((Pos(x, y, 22.0) * fs.button_head('M3', 8, lite=True), 'steel_black_oxide'))
    parts, _ = motors.bldc(stator_d=56.0, stator_h=16.0, can_d=66.0, mount=30.0)
    out += place(parts, Pos(L - 8.0, 0, 22.0))
    return out


def main():
    files = {
        'hub': ([Node('hub-clamp-and-plates', plain(hub()), tol=0.08, ang=0.3)], {}),
        'motor-mount': ([Node('motor-mount', plain(motor_mount()), tol=0.08, ang=0.3)], {}),
    }
    ex.publish(KEY, files)


if __name__ == '__main__':
    main()
