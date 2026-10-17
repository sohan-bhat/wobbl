"""The drone arm's frame and motor in CAD (examples/drone-arm, and its resonance run): the arm's root clamped between
3 mm and 2 mm carbon plates on aluminium standoffs by two M3 cap screws with nylon insert nuts, a 2207 class
outrunner on the tip pad on four M3 screws from below, its leads run back along the arm, and a 7 inch two blade
propeller under its locknut.

Built in the part's millimetre frame and added as it stands; the resonance run reads the same arm, so it loads the
same files.

    python site/cad/drone_arm.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Pos, RectangleRounded, Rot, SlotCenterToCenter, Solid, chamfer, extrude, fillet  # noqa: E402

import looks  # noqa: E402
from gltf import Node, write_manifest  # noqa: E402
from mech import cable  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech import motors  # noqa: E402
from mech.util import place, plain  # noqa: E402

KEY = 'drone-arm'
ROOT_BOLTS = [(-14.0, 0.0), (-6.0, 0.0)]
ROOT_TOP = 20.0
PAD = np.array([150.0, 0.0, 15.0])
looks.add('carbon_fibre', color=(0.018, 0.018, 0.02), metal=0.0, rough=0.28, normal='fine_n', nstr=0.5, tiles=80)
looks.add('wire_black', color=(0.02, 0.02, 0.02), metal=0.0, rough=0.45)
looks.add('wire_yellow', color=(0.7, 0.55, 0.05), metal=0.0, rough=0.45)
looks.add('wire_red', color=(0.55, 0.03, 0.02), metal=0.0, rough=0.45)


def frame_plates():
    """The bottom and top carbon plates of the frame's centre with their lightening slots, and the standoffs."""
    out = []
    for z0, t, x0, x1, w in ((-3.0, 3.0, -124.0, 4.0, 88.0), (ROOT_TOP, 2.0, -114.0, 2.0, 76.0)):
        f = Pos((x0 + x1) / 2, 0) * RectangleRounded(x1 - x0, w, 12.0)
        for x in (-92.0, -60.0):
            f = f - Pos(x, 0) * SlotCenterToCenter(18.0, 9.0)
        for x, y in ROOT_BOLTS:
            f = f - Pos(x, y) * RectangleRounded(3.4, 3.4, 1.69)
        for x in (-104.0, -34.0):
            for y in (-w / 2 + 8, w / 2 - 8):
                f = f - Pos(x, y) * RectangleRounded(3.4, 3.4, 1.69)
        p = Pos(0, 0, z0) * extrude(f, t)
        p = chamfer(p.edges().filter_by(Axis.Z, reverse=True).group_by(Axis.Z)[-1], 0.3)
        out.append((p, 'carbon_fibre'))
    for x in (-104.0, -34.0):
        for y in (-30.0, 30.0):
            st = Pos(x, y + (np.sign(y) * 0.0), 0) * fs.hex_prism(5.0, ROOT_TOP)
            st = st - Pos(x, y, -1) * Solid.make_cylinder(1.25, ROOT_TOP + 2)
            out.append((st, 'aluminum_anodized_gold'))
            out.append((Pos(x, y, ROOT_TOP + 2.0) * fs.button_head('M3', 6, lite=True), 'steel_black_oxide'))
            out.append((Pos(x, y, -3.0) * Rot(180, 0, 0) * fs.button_head('M3', 6, lite=True), 'steel_black_oxide'))
    return out


def root_screws():
    out = []
    for x, y in ROOT_BOLTS:
        out.append((Pos(x, y, ROOT_TOP + 2.0) * fs.socket_cap('M3', 30, threads=False, lite=True), 'steel_black_oxide'))
        out.append((Pos(x, y, -3.0 - 3.0) * fs.hex_nut('M3', nylock=True), 'steel_zinc'))
    return out


def motor_and_prop():
    """The motor on the pad, its screws from under the pad, the leads back along the arm, the prop and its nut."""
    parts, ztop = motors.bldc()
    out = place(parts, Pos(*PAD))
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = PAD[0] + 8 * math.sqrt(2) * math.cos(a), PAD[1] + 8 * math.sqrt(2) * math.sin(a)
        out.append((Pos(x, y, 0) * Rot(180, 0, 0) * fs.socket_cap('M3', 18, threads=False, lite=True), 'steel_black_oxide'))
    zp = PAD[2] + ztop
    out += place(motors.propeller(), Pos(PAD[0], PAD[1], zp - 0.6) * Rot(0, 0, 30))
    nut = Pos(PAD[0], PAD[1], zp + 7.0 - 0.6) * fs.hex_nut('M5', nylock=True)
    out.append((nut, 'steel_zinc'))
    # the three phase leads down the side of the arm to the frame
    path = [(PAD[0] - 13.0, 0.0, PAD[2] + 2.0), (PAD[0] - 18.0, 0.0, 14.2), (100.0, 0.0, 15.8), (40.0, 0.0, 18.4), (6.0, 0.0, 20.2),
            (-10.0, 0.0, 21.0)]
    path = [(x, y, z + 0.9) for x, y, z in path]
    out += cable.ribbon(path, 0.8, ['wire_black', 'wire_yellow', 'wire_red'], side=(0, 1, 0))
    return out


def main():
    files = {
        'frame-plates': ([Node('frame-plates', plain(frame_plates()), tol=0.06, ang=0.3)], ['mounting plate']),
        'root-screws': ([Node('root-screws', plain(root_screws()), tol=0.05, ang=0.4)], ['bolt', 'bolt head']),
        'motor-and-prop': ([Node('motor-and-prop', plain(motor_and_prop()), tol=0.06, ang=0.3)], ['load point']),
    }
    total = ex.publish(KEY, {k: (n, {'covers': c}) for k, (n, c) in files.items()})
    write_manifest('drone-arm-resonance', {'add': [{'file': f'{KEY}/{k}.glb'} for k in files]})


if __name__ == '__main__':
    main()
