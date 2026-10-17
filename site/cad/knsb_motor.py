"""Context models for the KNSB motor's static fire (examples/knsb-motor), from a 54 mm amateur motor's drawings: the
steel nozzle with its graphite throat insert and O ring grooves and the forward closure with its pressure port,
transducer and cable, both cut open on the wedge the scene cuts the casing on so the grains still show; the radial
button head screws that hold them in the casing; the welded channel stand with its anchor bolts on a concrete pad;
two cradles with rubber liners and top straps; and the floating thrust plate on guide rods with linear bushings,
pressing a button load cell against the gusseted backstop. The casing keeps its FE mesh and stress colors.

    python site/cad/knsb_motor.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import battery  # noqa: E402, F401  (silicone colors)
from build123d import Align, Axis, Box, Compound, Cylinder, Polygon, Pos, Rot, chamfer, extrude, fillet  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from mech import cable  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import place  # noqa: E402
from rocketry import lathe  # noqa: E402
from scene import Scene  # noqa: E402

RUN = os.path.join(HERE, '..', '..', 'examples', 'knsb-motor', 'output', 'scene.wobbl')
EX = 'knsb-motor'


def profile_of(sc, name):
    P = sc.mesh(sc.find(name)[0])[0]
    r, z = np.hypot(P[:, 0], P[:, 1]), np.round(P[:, 2], 3)
    return [(float(zz), float(r[z == zz].min()), float(r[z == zz].max())) for zz in sorted(set(z))]


def wedge(half, z0, z1, R=80.0):
    """The open wedge round local -x, half angle half (radians), from z0 to z1."""
    pts = [(0, 0)] + [(R * math.cos(math.pi - half + 2 * half * i / 16), R * math.sin(math.pi - half + 2 * half * i / 16))
                      for i in range(17)]
    return Pos(0, 0, z0) * extrude(Polygon(*pts, align=None), z1 - z0)


def nozzle(st, half):
    (z0, ri0, ro0), (z1, ri1, ro1), (z2, ri2, ro2), (z3, ri3, ro3), (z4, ri4, ro4) = st
    outer = [(ri4, z4), (ro4 - 0.6, z4), (ro4, z4 - 0.6), (ro3, z3), (ro2, z2 + 1.2), (ro2 + 1.2, z2), (ro1 - 0.8, z1 + 0.6),
             (ro1, z1), (ro0, z0 + 0.8), (ro0 - 0.8, z0)]
    inner = [(ri0, z0), (ri1, z1), (ri2 + 3.4, z2 - 6), (ri2 + 3.4, z2 + 6), (ri3, z3), (ri4, z4)]
    nz = lathe(outer + inner)
    grooves = [Pos(0, 0, z0 + d) * (Cylinder(ro0 + 1, 2.4) - Cylinder(ro0 - 1.5, 3)) for d in (3.0, 7.0)]
    nz = nz - Compound(grooves) - wedge(half, z0 - 1, z4 + 1)
    insert = lathe([(ri2, z2 - 6), (ri2 + 3.4, z2 - 6), (ri2 + 3.4, z2 + 6), (ri2 + 1.0, z2 + 6), (ri2, z2 + 3), (ri2, z2 - 3),
                    (ri2 + 1.4, z2 - 6)]) - wedge(half, z2 - 7, z2 + 7)
    orings = [Pos(0, 0, z0 + d) * (Cylinder(ro0 - 0.05, 2.0) - Cylinder(ro0 - 1.4, 3)) - wedge(half, z0, z0 + 12) for d in (3.0, 7.0)]
    return [(nz, 'steel_bare'), (insert, 'graphite')] + [(o, 'rubber') for o in orings]


def closure(fc, half):
    fz0, fz1, fr = fc[0][0], fc[-1][0], fc[0][2]
    clo = lathe([(0, fz0), (fr - 0.6, fz0), (fr, fz0 + 0.6), (fr, fz1 - 3.6), (fr - 1.2, fz1 - 3.2), (fr - 1.2, fz1 - 1.6),
                 (fr, fz1 - 1.2), (fr, fz1), (0, fz1)])
    port = Pos(0, 0, fz0 - 1) * Cylinder(3.2, fz1 - fz0 + 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    clo = clo - port - wedge(half, fz0 - 1, fz1 + 1)
    # the pressure port, plugged flush for this firing by a brass set screw
    plug = Pos(0, 0, fz0 + 0.3) * Cylinder(3.2, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    out = [(clo, 'aluminum_machined'), (plug - wedge(half, fz0 - 1, fz1 + 1), 'brass')]
    return out


def stand(x_lo, x_hi):
    """Two C channel rails on cross members, anchored to a concrete pad by studs, nuts and washers."""
    out = []
    for s in (-1, 1):
        web = Pos((x_lo + x_hi) / 2, s * 30, 7.5) * Box(x_hi - x_lo, 4.0, 15)
        flanges = [Pos((x_lo + x_hi) / 2, s * 30 - s * 4, z) * Box(x_hi - x_lo, 12.0, 3.0) for z in (1.5, 13.5)]
        out.append((web + flanges[0] + flanges[1], 'steel_mill'))
    for x in (x_lo + 10, x_hi - 10):
        cross = Pos(x, 0, 6) * Box(20, 96, 12)
        out.append((chamfer(cross.edges(), 1.0), 'steel_mill'))
        for y in (-40, 40):
            out.append((Pos(x, y, 12) * (Cylinder(11, 2, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(5.2, 9)), 'steel_zinc'))
            nut = Pos(x, y, 14) * fs.hex_nut('M10')
            out.append((nut, 'steel_zinc'))
            out.append((Pos(x, y, 12) * Cylinder(5, 11.6, align=(Align.CENTER, Align.CENTER, Align.MIN)), 'steel_zinc'))
    pad = Pos(-40, 0, -35) * Box(600, 300, 70)
    out.append((fillet(pad.edges(), 6.0), 'concrete'))
    return out


def cradle(axis_z, rcase):
    """One cradle (in the saddle's frame, its box centred on the origin): a plate upright cut to the casing with a
    rubber liner, a foot bolted to the rails, gussets, and a steel strap over the casing with its two bolts."""
    out = []
    base_z = -17.0                           # the saddle box's floor, on the stand's rails
    top = base_z + 21.0                      # the foot's top face
    za = axis_z - 17.0                       # the casing's axis in the saddle's frame
    pts = [(-36, top), (36, top), (36, za)]
    for k in range(33):
        a = 2 * math.pi - (math.pi * k / 32)
        pts.append(((rcase + 1.6) * math.cos(a), za + (rcase + 1.6) * math.sin(a)))
    pts.append((-36, za))
    up = Rot(90, 0, 90) * extrude(Polygon(*pts, align=None), 4, both=True)
    out.append((up, 'steel_mill'))
    foot = Pos(0, 0, top - 3) * Box(36, 84, 6)
    out.append((chamfer(foot.edges().group_by(Axis.Z)[-1], 0.8), 'steel_mill'))
    for s in (-1, 1):
        for d in (-1, 1):
            g = Polygon((0, 0), (d * 14, 0), (0, 19), align=None)
            g = Pos(0, s * 30, top) * Rot(90, 0, 0) * extrude(g, 2.5, both=True)
            out.append((g, 'steel_mill'))
    liner = Pos(0, 0, za) * Rot(0, 90, 0) * (Cylinder(rcase + 1.65, 18) - Cylinder(rcase + 0.05, 20)) & \
        Pos(0, 0, za - 40) * Box(40, 200, 80)
    out.append((liner, 'rubber'))
    strap = Pos(0, 0, za) * Rot(0, 90, 0) * (Cylinder(rcase + 1.6, 16) - Cylinder(rcase + 0.02, 18)) & \
        Pos(0, 0, za + 40) * Box(40, 200, 80)
    out.append((strap, 'steel_zinc'))
    for s in (-1, 1):
        ear = Pos(0, s * (rcase + 7), za + 1.5) * Box(16, 12, 3)
        out.append((ear - Pos(0, s * (rcase + 8), za) * Cylinder(3.2, 10), 'steel_zinc'))
        bolt = Pos(0, s * (rcase + 8), za + 3) * fs.hex_bolt('M6', 20, threads=False, marks=False)
        out.append((bolt, 'steel_zinc'))
    for x in (-11, 11):
        for y in (-34, 34):
            out.append((Pos(x, y, top) * fs.hex_nut('M6'), 'steel_zinc'))
    return out


def thrust_rig(hx, hy, hz):
    """In the thrust plate's frame (its box centred): the plate floating on two chrome guide rods through linear
    bushings, the button load cell behind it with its cable, and the gusseted backstop welded to the stand."""
    out = []
    xf, xb = hx, -hx
    z0 = -hz
    axis = 57.0 - 42.0
    plate = Pos(0, 0, 8.5) * Box(2 * hx, 2 * hy, 2 * hz - 17)
    plate = chamfer(plate.edges().filter_by(Axis.X).group_by(Axis.Z)[-1], 6)
    plate = plate - Compound([Pos(0, y, 72 - 42) * Cylinder(6.5, 40, rotation=(0, 90, 0)) for y in (-32, 32)])
    out.append((plate, 'steel_mill'))
    cell = Pos(xb, 0, axis) * Rot(0, -90, 0) * lathe([(0, 0), (19, 0), (19.5, 1), (19.5, 17), (19, 18), (7, 18), (6, 20.5), (0, 20.5)])
    out.append((cell, 'steel_stainless'))
    gl = Pos(xb - 10, 19.2, axis) * Rot(-90, 0, 0) * Cylinder(4.5, 7, align=(Align.CENTER, Align.CENTER, Align.MIN))
    out.append((gl, 'plastic_black'))
    out.append((cable.tube([(xb - 10, 26, axis), (xb - 10, 40, axis - 4), (xb - 14, 52, axis - 30), (xb - 22, 58, z0 + 18),
                            (xb - 60, 62, z0 + 16)], 2.6, seg=12), 'rubber'))
    xs = xb - 20.5
    bs = Pos(xs - 7, 0, z0 + 48) * Box(14, 96, 96)
    bs = chamfer(bs.edges().filter_by(Axis.X).group_by(Axis.Z)[-1], 6)
    out.append((bs, 'steel_mill'))
    for y in (-40, 40):
        g = Pos(xs - 14, y, 0) * Rot(90, 0, 0) * extrude(Polygon((0, z0 + 15), (-46, z0 + 15), (0, z0 + 80), align=None), 4, both=True)
        out.append((g, 'steel_mill'))
    for y in (-32, 32):
        rod = Pos(xs, y, 72 - 42) * Rot(0, 90, 0) * Cylinder(6, 75, align=(Align.CENTER, Align.CENTER, Align.MIN))
        out.append((chamfer(rod.edges().group_by(Axis.Z)[-1], 0.4), 'steel_chrome'))
        bush = Pos(xb - 16, y, 72 - 42) * Rot(0, 90, 0) * (Cylinder(9, 16, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(6, 40))
        out.append((bush, 'steel_zinc'))
        out.append((Pos(xs + 75, y, 72 - 42) * Rot(0, 90, 0) * fs.hex_nut('M8'), 'steel_zinc'))
    return out


def main():
    sc = Scene(RUN)
    ci = sc.find('casing')[0]
    MOTOR = sc.matrix(ci)
    P = sc.mesh(ci)[0]
    r = np.hypot(P[:, 0], P[:, 1])
    half = math.acos(-P[:, 0].min() / r.max())
    st = profile_of(sc, 'nozzle')
    fc = profile_of(sc, 'forward closure')
    rcase = st[0][2]
    axis_z = float(MOTOR[2, 3])

    export(f'{EX}/nozzle.glb', [Node('nozzle', nozzle(st, half))], tol=0.03, ang=0.3)
    export(f'{EX}/forward-closure.glb', [Node('forward_closure', closure(fc, half))], tol=0.03, ang=0.3)
    bolts = []
    for zb in (fc[-1][0] + 6.5, st[0][0] - 6.5):
        for k in range(8):
            a = 2 * math.pi * k / 8
            if abs((a - math.pi + math.pi) % (2 * math.pi) - math.pi) < half + 0.2:
                continue
            s = fs.button_head('M5', 6)
            # a screw stands on +z with its head's underside at z = 0: turn +z out along the radius
            bolts.append((Pos(math.cos(a) * rcase, math.sin(a) * rcase, zb) * Rot(0, 0, math.degrees(a)) * Rot(0, 90, 0) * s,
                          'steel_black_oxide'))
    export(f'{EX}/casing-bolts.glb', [Node('casing_bolts', bolts, loc=MOTOR)], tol=0.03, ang=0.35)
    export(f'{EX}/stand.glb', [Node('stand', stand(-300.0, 175.0))], tol=0.05, ang=0.3)
    export(f'{EX}/cradle.glb', [Node('cradle', cradle(axis_z, rcase))], tol=0.03, ang=0.3)
    ti, to, TM = sc.named('thrust plate')[0]
    export(f'{EX}/thrust-plate.glb', [Node('thrust_plate', thrust_rig(*to['size']))], tol=0.03, ang=0.3)
    write_manifest(EX, {
        'replace': [
            {'object': 'nozzle', 'file': f'{EX}/nozzle.glb'},
            {'object': 'forward closure', 'file': f'{EX}/forward-closure.glb'},
            {'object': 'stand saddle', 'file': f'{EX}/cradle.glb'},
            {'object': 'thrust plate', 'file': f'{EX}/thrust-plate.glb'},
        ],
        'add': [{'file': f'{EX}/casing-bolts.glb'}, {'file': f'{EX}/stand.glb'}],
        'hide': ['ground'],
    })


if __name__ == '__main__':
    main()
