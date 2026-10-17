"""The cast motor mount's hardware in CAD (examples/cast-motor-mount, and its fatigue run): the 80 mm frame AC servo
the casting was drawn for, on its 70 mm pilot and four M6 cap screws through the 90 mm bolt circle with nuts on the
back face, an HTD 5M pulley keyed on the shaft 30 mm behind the upright with the 15 mm belt leaving 30 degrees below
horizontal, the four M8 bolts with large washers through the base's slots, and the steel frame plate.

The casting's holes come from its STEP file (a 70.5 mm pilot bore, 6.6 mm holes on a 63.6 mm square, which is a
90 mm bolt circle), so the servo is the one those holes fit rather than a NEMA stepper, whose 2 kN belt pull it
could not carry. Built in the part's millimetre frame and added as it stands.

    python site/cad/cast_motor_mount.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Plane, Pos, RectangleRounded, Rot, Solid, chamfer, extrude, fillet  # noqa: E402

from gltf import Node, export, write_manifest  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech import motors  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

FACE = 6.0                    # the upright's front face (the servo side), its back face is y = 0
AXIS = np.array([0.0, 0.0, 66.0])
BASE_T = 6.0
PLATE_T = 8.4
SLOTS = [(sx * 60.0, y) for sx in (-1, 1) for y in (24.0, 78.0)]
PULL = np.array([-math.cos(math.radians(30)), 0.0, -math.sin(math.radians(30))])


def servo():
    M = frame(np.array([0.0, FACE, AXIS[2]]), [1, 0, 0], [0, -1, 0])
    out = place(motors.servo80(length=118.0, shaft_len=52.0), M)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, z = 45 * math.cos(a), AXIS[2] + 45 * math.sin(a)
        # the cap screw sits in the flange's counterbore, 6.5 mm behind the face, its shank through to the back
        Ms = frame(np.array([x, FACE + 6.5, z]), [1, 0, 0], [0, 1, 0])
        out += place([(fs.socket_cap('M6', 25, threads=True), 'steel_black_oxide')], Ms)
        Mn = frame(np.array([x, 0.0, z]), [1, 0, 0], [0, -1, 0])
        out += place([(fs.washer('M6'), 'steel_zinc'), (Pos(0, 0, fs.WASHER['M6'][2]) * fs.hex_nut('M6'), 'steel_zinc')], Mn)
    return out


def pulley_and_belt():
    parts, total = motors.htd_pulley(teeth=20, bore=19.0, belt=15.0)
    y_hub = -30.0 + total / 2
    M = frame(np.array([0.0, y_hub, AXIS[2]]), [1, 0, 0], [0, -1, 0])
    out = place(parts, M)
    band_mid = y_hub - (12.0 + 1.5 + 8.5)
    pd = 5.0 * 20 / math.pi
    r_in, t = pd / 2 - 0.57, 3.4
    # the wrap round the side away from the pull, and the two runs leaving along it
    n = np.array([-PULL[2], 0.0, PULL[0]])            # square to the pull in the xz plane
    ring = Solid.make_cylinder(r_in + t, 15.0) - Solid.make_cylinder(r_in, 15.0)
    ang = math.degrees(math.atan2(PULL[2], PULL[0]))
    half = Rot(0, 0, ang) * Pos((r_in + t + 1) / 2, 0, 7.5) * Box(r_in + t + 1, 2 * (r_in + t + 1), 17)
    ring = ring - half
    out += place([(ring, 'rubber')], frame(np.array([0.0, band_mid + 7.5, AXIS[2]]), [1, 0, 0], [0, -1, 0]))
    ends = []
    for s in (-1, 1):
        start = AXIS + n * s * (r_in + t)
        length = (start[2] + PLATE_T * 0.5) / -PULL[2]
        strand = motors.belt_strand(15.0, length, thickness=t - 2.0, pitch=5.0, tooth=2.0)
        R = frame(np.array([start[0], band_mid, start[2]]), PULL, n * s)
        out += place([(strand, 'rubber')], R)
        ends.append(start + PULL * length)
    return out, ends


def base_bolts():
    out = []
    for x, y in SLOTS:
        stack = [(Pos(0, 0, 2.0) * fs.hex_bolt('M8', 30), 'steel_zinc'), (fs.washer('M8', outer=24.0, thickness=2.0), 'steel_zinc'),
                 (Pos(0, 0, -BASE_T - PLATE_T - 1.6) * fs.washer('M8'), 'steel_zinc'),
                 (Pos(0, 0, -BASE_T - PLATE_T - 1.6 - 6.8) * fs.hex_nut('M8'), 'steel_zinc')]
        out += place(stack, Pos(x, y, BASE_T))
    return out


def frame_plate(belt_ends):
    """The machine's steel frame plate under the mount: 392 mm square, 8.4 mm thick, mill scale, the belt's slot
    and the holes where it bolts to the frame."""
    cy = 44.43
    p = Pos(0, cy, -PLATE_T / 2) * Box(392, 392, PLATE_T)
    p = fillet(p.edges().filter_by(Axis.Z), 10.0)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 1.0)
    for x, y in SLOTS:
        p = p - Pos(x, y, -20) * Solid.make_cylinder(4.2, 40)
    xs = [e[0] for e in belt_ends]
    sx0, sx1 = min(xs) - 8, max(xs) + 8
    p = p - Pos((sx0 + sx1) / 2, -30.0, -20) * extrude(RectangleRounded(sx1 - sx0, 26, 12.9), 40)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = p - Pos(sx * 176, cy + sy * 176, -20) * Solid.make_cylinder(5.5, 40)
    out = [(p, 'steel_mill')]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((Pos(sx * 176, cy + sy * 176, 0) * fs.hex_bolt('M10', 12, threads=False), 'steel_zinc'))
    return out


def node(name, parts, tol=0.08, ang=0.3):
    return Node(name, plain(parts), tol=tol, ang=ang)


def build(key='cast-motor-mount'):
    pul, ends = pulley_and_belt()
    files = {
        'motor': ([node('servo-80', servo())], ['motor', 'motor flange', 'motor shaft']),
        'pulley': ([node('htd-pulley', pul)], ['pulley']),
        'bolts': ([node('m8-bolts', base_bolts(), 0.08, 0.45)], []),
        'plate': ([node('frame-plate', frame_plate(ends), 0.1, 0.3)], ['plate']),
    }
    entry, total = [], 0
    for slug, (nodes, cov) in files.items():
        rel = f'{key}/{slug}.glb'
        total += export(rel, nodes, unit=1.0)
        entry.append({'file': rel, 'covers': cov} if cov else {'file': rel})
    write_manifest(key, {'add': entry})
    # the fatigue run reads the same mount: it loads the same files, with nothing in its scene to cover
    write_manifest('motor-mount-fatigue', {'add': [{'file': e['file']} for e in entry]})
    print(f'[cad] {key} {total / 1024:.0f} KB')


def main():
    build()


if __name__ == '__main__':
    main()
