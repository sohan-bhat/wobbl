"""The printed NEMA 17 mount's hardware in CAD (examples/motor-mount): a 17HS4401 class stepper on the upright's front
face held by four M3 cap screws from behind, a 20 tooth GT2 pulley on the D flat 15 mm behind the upright with the
6 mm belt pulling straight down through a slot in the plate, the four M5 bolts with washers and nuts through the
base, and the aluminium plate the mount is bolted to.

The strength stage draws its part in its own millimetre frame, so these models are built in that frame and added
as they stand. Both runs (with and without ribs) share the hardware.

    python site/cad/motor_mount.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import Axis, Box, Pos, RectangleRounded, Solid, chamfer, extrude, fillet  # noqa: E402

from gltf import Node, export, write_manifest  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech import motors  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'motor-mount'
T = 5.0                 # base and upright thickness
CZ = 31.0               # motor axis height
BOLTS = [(sx * 29.0, y) for sx in (-1, 1) for y in (12.0, 42.0)]
PLATE_T = 3.5


def motor():
    """The stepper on the front (+y) face of the upright, shaft back through the pilot hole along -y."""
    M = frame(np.array([0.0, T, CZ]), [1, 0, 0], [0, -1, 0])
    return place(motors.nema17(length=39.6, shaft_len=27.0), M)


def motor_screws():
    """M3 x 10 cap screws from the back of the upright into the motor's tapped holes."""
    out = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            M = frame(np.array([sx * 15.5, 0.0, CZ + sz * 15.5]), [1, 0, 0], [0, -1, 0])
            out += place([(Pos(0, 0, 0.5) * fs.socket_cap('M3', 10, threads=False), 'steel_black_oxide'), (fs.washer('M3'), 'steel_zinc')], M)
    return out


def pulley_and_belt():
    """The GT2 pulley with its hub toward the motor, and the belt wrapped over it, both runs going down."""
    parts, total = motors.gt2_pulley()
    y_hub = -15.0 + total / 2               # the scene's pulley centre is 15 mm behind the upright
    M = frame(np.array([0.0, y_hub, CZ]), [1, 0, 0], [0, -1, 0])
    out = place(parts, M)
    band_mid = y_hub - (7.0 + 1.0 + 3.5)    # the middle of the toothed band
    r_in, t = 6.11, 1.38
    arc = Solid.make_cylinder(r_in + t, 6.0) - Solid.make_cylinder(r_in, 6.0)
    arc = arc - Pos(0, -(r_in + t + 1) / 2 - 0.0, 3.0) * Box(2 * (r_in + t + 1), r_in + t + 1, 8)
    out += place([(arc, 'rubber')], frame(np.array([0.0, band_mid + 3.0, CZ]), [1, 0, 0], [0, -1, 0]))
    drop = CZ + PLATE_T + 0.5      # the runs go down into the slot in the plate
    for s in (-1, 1):
        strand = motors.belt_strand(6.0, drop)
        # run down along -z, the back facing out and the teeth facing the pulley
        R = frame(np.array([s * (r_in + t), band_mid, CZ]), [0, 0, -1], [s, 0, 0])
        out += place([(strand, 'rubber')], R)
    return out


def bolts():
    out = []
    for x, y in BOLTS:
        out += place(fs.bolt_stack('M5', 16, T + PLATE_T, head='hex', threads=True), Pos(x, y, T))
    return out


def plate():
    """The 6061 plate the mount bolts to: 168 mm square, the bolt holes, a slot for the belt, broken edges and
    counterbored corner holes where it is screwed down in turn."""
    p = Pos(0, 25, -PLATE_T / 2) * Box(168, 168, PLATE_T)
    p = fillet(p.edges().filter_by(Axis.Z), 6.0)
    for x, y in BOLTS:
        p = p - Pos(x, y, -5) * Solid.make_cylinder(2.75, 10)
    p = p - Pos(0, -18.5, -5) * extrude(RectangleRounded(26, 10, 4.9), 10)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = p - Pos(sx * 72, 25 + sy * 72, -5) * Solid.make_cylinder(3.3, 10)
            p = p - Pos(sx * 72, 25 + sy * 72, -3.5 + 1.0 + 0.0) * Solid.make_cylinder(5.5, 4)
    p = chamfer(p.edges().group_by(Axis.Z)[-1], 0.5)
    out = [(p, 'aluminum_machined')]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((Pos(sx * 72, 25 + sy * 72, -1.0 - 1.5) * fs.socket_cap('M6', 8, threads=False, lite=True), 'steel_black_oxide'))
    return out


def node(name, parts, tol=0.05, ang=0.3):
    return Node(name, plain(parts), tol=tol, ang=ang)


def main():
    files = {
        'motor': ([node('nema17', motor())], ['motor', 'motor flange', 'motor shaft']),
        'pulley': ([node('gt2-pulley', pulley_and_belt())], ['pulley']),
        'bolts': ([node('m5-bolts', bolts(), 0.06, 0.45), node('m3-screws', motor_screws(), 0.06, 0.5)], ['bolt', 'bolt head']),
        'plate': ([node('plate', plate())], ['mounting plate']),
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
