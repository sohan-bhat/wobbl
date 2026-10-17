"""The strut's test frame in CAD (examples/strut-buckling): a two column load frame, the base platen with a split
clamp collar that clamps the tube's foot, and the crosshead platen whose spherical seat holds the top sideways while
leaving it free to turn, as the run's end conditions say.

The run is drawn by the scene player: the base and top platens replace the scene's two end plates, each in its
own frame, and the columns, bed and crosshead are added.

    python site/cad/strut_buckling.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from build123d import Axis, Box, Plane, Polyline, Pos, Rot, Solid, chamfer, fillet, make_face, revolve  # noqa: E402

import looks  # noqa: E402
from gltf import Node  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import plain  # noqa: E402

KEY = 'strut-buckling'
L = 600.0
R = 10.0                 # the tube's outer radius
P_T = 18.0               # platens
GAP = 8.0                # the spherical seat between the tube's top and the crosshead platen
COL_X = 125.0
looks.add('frame_paint', color=(0.05, 0.13, 0.28), metal=0.0, rough=0.42)


def platen(z0, down=True):
    """A ground steel platen 144 mm square, its edges broken and four counterbored cap screws into whatever carries
    it: the bed below (down) or the crosshead above."""
    p = Pos(0, 0, z0 + P_T / 2) * Box(144, 144, P_T)
    p = chamfer(p.edges(), 1.0)
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = p - Pos(sx * 60, sy * 60, z0 - 1) * Solid.make_cylinder(6.6, P_T + 2)
            if down:
                p = p - Pos(sx * 60, sy * 60, z0 + P_T - 12.5) * Solid.make_cylinder(10.0, 13)
                out.append((Pos(sx * 60, sy * 60, z0 + P_T - 12.0) * fs.socket_cap('M12', 30, threads=False, lite=True), 'steel_black_oxide'))
            else:
                p = p - Pos(sx * 60, sy * 60, z0 - 0.5) * Solid.make_cylinder(10.0, 13)
                out.append((Pos(sx * 60, sy * 60, z0 + 12.0) * Rot(180, 0, 0) * fs.socket_cap('M12', 30, threads=False, lite=True), 'steel_black_oxide'))
    return [(p, 'steel_bare')] + out


def base():
    """The base platen and the split clamp collar on it, its slit drawn shut by two M6 cap screws."""
    out = platen(-P_T)
    h = 30.0
    blk = Pos(0, 0, h / 2) * Box(56, 44, h)
    blk = fillet(blk.edges().filter_by(Axis.Z), 4.0)
    blk = chamfer(blk.edges().group_by(Axis.Z)[-1], 0.8)
    blk = blk - Pos(0, 0, -1) * Solid.make_cylinder(R + 0.05, h + 2)
    blk = blk - Pos(17, 0, h / 2) * Box(34, 1.2, h + 2)
    for z in (9.0, 21.0):
        blk = blk - Pos(20, -22, z) * Rot(-90, 0, 0) * Solid.make_cylinder(3.3, 44)
        blk = blk - Pos(20, -23, z) * Rot(-90, 0, 0) * Solid.make_cylinder(5.5, 9)
        out.append((Pos(20, -14.0, z) * Rot(90, 0, 0) * fs.socket_cap('M6', 30, threads=False, lite=True), 'steel_black_oxide'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            blk = blk - Pos(sx * 21, sy * 15, h - 6.5) * Solid.make_cylinder(4.8, 7)
            out.append((Pos(sx * 21, sy * 15, h - 6.0) * fs.socket_cap('M5', 20, threads=False, lite=True), 'steel_black_oxide'))
    out.append((blk, 'steel_bare'))
    return out


def top():
    """The crosshead platen raised on the spherical seat: a hardened end cap in the tube, the ball and its
    conical cup, and the platen above."""
    out = platen(L + GAP, down=False)
    cap = revolve(make_face(Plane.XZ * Polyline((0, L - 8), (R - 3.0, L - 8), (R - 3.0, L), (R, L), (R, L + 2.0), (R - 1, L + 3.0),
                                                 (0, L + 3.0), close=True)), Axis.Z)
    ball = Pos(0, 0, L + GAP - 4.0) * Solid.make_sphere(6.0)
    cup = Pos(0, 0, L + GAP - 3.0) * (Solid.make_cylinder(13.0, 3.0) - Solid.make_cone(7.0, 2.0, 3.0))
    out += [(cap, 'steel_bare'), (ball, 'steel_chrome'), (cup, 'steel_bare')]
    return out


def frame():
    """The bed, the two chromed columns and the crosshead of the load frame."""
    bed = Pos(0, 0, -P_T - 30.0) * Box(2 * COL_X + 80, 170, 60)
    bed = fillet(bed.edges().filter_by(Axis.Y), 6.0)
    z_top = L + GAP + P_T
    head = Pos(0, 0, z_top + 32.0) * Box(2 * COL_X + 80, 110, 64)
    head = fillet(head.edges().filter_by(Axis.Y), 6.0)
    out = [(bed, 'frame_paint'), (head, 'frame_paint')]
    for s in (-1, 1):
        col = Pos(s * COL_X, 0, -P_T) * Solid.make_cylinder(19.0, z_top + 64 + P_T)
        out.append((col, 'steel_chrome'))
        for z in (-P_T, z_top):
            out.append((Pos(s * COL_X, 0, z) * fs.hex_prism(46, 18, z0=0 if z < 0 else 64), 'steel_zinc'))
    return out


def main():
    sc = ex.scene(f'{KEY}/output')
    plates = sc.named('end plate')
    (ib, _, Mb0), (it, _, Mt0) = sorted(plates, key=lambda p: p[2][2][3])
    files = {
        'base-platen': ([Node('base-platen', plain(ex.into(base(), ex.pose_mm(sc, ib))), tol=0.08, ang=0.3)], {'object': 'end plate', 'index': ib}),
        'top-platen': ([Node('crosshead-platen', plain(ex.into(top(), ex.pose_mm(sc, it))), tol=0.08, ang=0.3)], {'object': 'end plate', 'index': it}),
        'load-frame': ([Node('load-frame', plain(frame()), tol=0.15, ang=0.3)], {}),
    }
    ex.publish(KEY, files)


if __name__ == '__main__':
    main()
