"""The wiper linkage in CAD: the motor's die cast gearbox with its ribbed cover and rubber mounted ears, the worm tube
and the painted can, the stamped crank with its pressed rib on the output shaft under a flanged nut, ball studs with
hex collars riveted into the crank and the rocker, the pressed coupler rod with its ball sockets and boots, the
rocker on the spindle, the cast pivot tower, the arm and the beam blade.

Every joint is where the run's scene.wobbl solved it, as the Blender models had them: the parts are built where the
linkage stands on the frame the page opens on, in millimetres, then moved into each object's own frame, so they
follow the motion. The links stand in the plane z = 0 of the scene, stacked about it like a real linkage: crank
and rocker plates just under it and the coupler riding over both on its ball joints.

    python site/cad/wiper_linkage.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import numpy as np  # noqa: E402
from build123d import (Axis, Box, Circle, Ellipse, Plane, Polyline, Pos, Rectangle, RectangleRounded, Rot,  # noqa: E402
                       Solid, Sketch, chamfer, extrude, fillet, loft, make_face, revolve, Location)

from gltf import Node, export, write_manifest  # noqa: E402
from mech import example as ex  # noqa: E402
from mech import fasteners as fs  # noqa: E402
from mech.util import frame, place, plain  # noqa: E402

KEY = 'wiper-linkage'
UNIT = 0.001
PLATE = (-3.0, 1.0)        # crank and rocker plates, z in mm
BALL_Z = 12.5              # ball joint centres above the link plane


def rev(profile, at=(0, 0, 0), axis='Z'):
    """A solid of revolution from (r, z) points, about z (or x) through at."""
    pts = []
    for p in profile:
        if not pts or abs(pts[-1][0] - p[0]) > 1e-6 or abs(pts[-1][1] - p[1]) > 1e-6:
            pts.append(p)
    s = revolve(make_face(Plane.XZ * Polyline(*pts, close=True)), Axis.Z)
    if axis == 'X':
        s = Rot(0, 90, 0) * s
    return Pos(*at) * s


def soft(op, part, edges, r):
    for k in (1.0, 0.6, 0.35):
        try:
            return op(edges, r * k)
        except Exception:
            continue
    return part


def hull_link(A, B, rA, rB):
    """The outline of a link round two circles, tangent lines between them."""
    d = B - A
    L = float(np.linalg.norm(d))
    a = math.degrees(math.atan2(d[1], d[0]))
    phi = math.asin((rA - rB) / L)
    pts = []
    n = 40
    for k in range(n + 1):        # the big end's arc, round the back
        t = math.pi / 2 + phi + k * (math.pi - 2 * phi) / n
        pts.append((rA * math.cos(t), rA * math.sin(t)))
    for k in range(n + 1):
        t = -math.pi / 2 + phi + k * (math.pi - 2 * phi) / n
        pts.append((L + rB * math.cos(t), rB * math.sin(t)))
    pts2 = [(x * math.cos(math.radians(a)) - y * math.sin(math.radians(a)) + A[0],
             x * math.sin(math.radians(a)) + y * math.cos(math.radians(a)) + A[1]) for x, y in pts]
    return make_face(Polyline(*pts2, close=True))


def link_plate(A, B, rA, rB, mat):
    """A stamped steel link: the plate with its edges rolled, and the pressed stiffening rib along the middle."""
    z0, z1 = PLATE
    p = extrude(Plane.XY.offset(z0) * hull_link(A, B, rA, rB), z1 - z0)
    p = soft(fillet, p, p.edges().filter_by(Axis.Z, reverse=True), 0.9)
    u = (B - A) / np.linalg.norm(B - A)
    a, b = A + u * (rA + 6.0), B - u * (rB + 4.0)
    rib = extrude(Plane.XY.offset(z1 - 0.4) * hull_link(a, b, 4.8, 3.6), 1.1)
    rib = soft(fillet, rib, rib.edges().group_by(Axis.Z)[-1], 0.5)
    return [(p + rib, mat)]


def ball_stud(at):
    """A ball stud: hex collar on the plate, the neck, the ball, and the riveted head under the plate."""
    x, y = at[0], at[1]
    z1 = PLATE[1]
    out = []
    collar = Pos(x, y, 0) * fs.hex_prism(11.0, 3.5, z0=z1)
    out.append((collar, 'steel_zinc'))
    neck = rev([(0, z1 + 3.4), (4.2, z1 + 3.4), (3.6, 7.5), (3.2, BALL_Z - 4.0), (0, BALL_Z - 4.0)], (x, y, 0))
    ball = Pos(x, y, BALL_Z) * Solid.make_sphere(6.2)
    out += [(neck, 'steel_bare'), (ball, 'steel_chrome')]
    rivet = rev([(0, PLATE[0]), (5.4, PLATE[0]), (5.0, PLATE[0] - 0.8), (3.6, PLATE[0] - 1.5), (0, PLATE[0] - 1.7)], (x, y, 0))
    out.append((rivet, 'steel_bare'))
    return out


def socket(at):
    """A coupler's ball socket: the moulded cup over the ball and the rubber boot under it."""
    x, y = at[0], at[1]
    cup = rev([(0, 5.8), (8.2, 5.8), (10.2, 7.6), (10.6, 11.8), (10.2, 16.8), (8.4, 19.2), (0, 19.6)], (x, y, 0))
    boot = rev([(4.4, PLATE[1] + 3.6), (6.2, PLATE[1] + 4.0), (7.2, 5.0), (6.6, 5.4), (8.0, 5.9), (4.0, 5.9)], (x, y, 0))
    return [(cup, 'plastic_black'), (boot, 'rubber')]


def flange_nut(at, z0, down=False):
    """An M8 serrated flange nut (ISO 4161): flange 17.9 mm, 13 mm hex, 8 mm high, on z0 going up (or down)."""
    n = fs.hex_prism(13.0, 6.2, z0=1.8)
    fl = rev([(0, 0), (8.95, 0), (8.95, 0.6), (6.6, 1.9), (0, 1.9)])
    nut = n + fl - Pos(0, 0, -1) * Solid.make_cylinder(3.4, 10)
    if down:
        nut = Rot(180, 0, 0) * nut
    return Pos(at[0], at[1], z0) * nut


def threaded_tip(at, z0, length, r=4.0):
    """A shaft's threaded end standing from z0, rings at 1.25 mm pitch and a chamfer."""
    rod = fs.threaded_rod(2 * r, length, 1.25, 0.0, True)
    return Pos(at[0], at[1], z0 + length) * rod


def build_links(O, PC, PV, PR):
    models = {}
    # crank: stamped steel tinted by the motor's torque, on its collar and the shaft's taper under the nut
    crank = link_plate(O, PC, 13.5, 10.5, 'paint')
    crank.append((rev([(0, PLATE[1] - 0.2), (10.5, PLATE[1] - 0.2), (10.5, PLATE[1] + 2.2), (9.4, PLATE[1] + 3.3), (0, PLATE[1] + 3.3)], (O[0], O[1], 0)), 'paint'))
    crank.append((Pos(O[0], O[1], 4.3) * fs.washer('M8', outer=17.6, thickness=0.9), 'steel_zinc'))
    crank.append((flange_nut(O, 5.2), 'steel_zinc'))
    crank.append((threaded_tip(O, 11.2, 2.4), 'steel_bare'))
    models['crank'] = crank
    models['coupler pin'] = ball_stud(PC)

    # coupler: a pressed tube, round in the middle and flattened into paddles riveted over the two sockets
    u = (PR - PC) / np.linalg.norm(PR - PC)
    L = float(np.linalg.norm(PR - PC))
    ang = math.degrees(math.atan2(u[1], u[0]))
    secs = []
    over = 11.5
    for t, (w, h, zc) in ((over + 10.0, (12.0, 3.4, 21.0)), (over + 22.0, (12.4, 7.0, 19.0)), (over + 36.0, (12.4, 9.2, 17.8)),
                          (L - over - 36.0, (12.4, 9.2, 17.8)), (L - over - 22.0, (12.4, 7.0, 19.0)), (L - over - 10.0, (12.0, 3.4, 21.0))):
        sk = RectangleRounded(h, w, min(h, w) * 0.49)
        secs.append(Plane(origin=(t, 0, zc), x_dir=(0, 0, 1), z_dir=(1, 0, 0)) * sk)
    rod = loft(secs)
    rod = Pos(PC[0], PC[1], 0) * Rot(0, 0, ang) * rod
    parts = [(rod, 'paint')]
    for at, sgn in ((PC, 1), (PR, -1)):
        pad = extrude(Plane.XY.offset(19.6) * (Pos(at[0], at[1]) * Circle(10.8) +
                      Pos(*(at[:2] + u[:2] * sgn * 9.0)) * Rot(0, 0, ang) * Rectangle(20.0, 12.0)), 3.2)
        pad = soft(fillet, pad, pad.edges().filter_by(Axis.Z, reverse=True), 1.0)
        parts.append((pad, 'paint'))
        parts += socket(at)
        parts.append((rev([(0, 22.6), (5.0, 22.6), (4.6, 23.6), (3.0, 24.4), (0, 24.6)], (at[0], at[1], 0)), 'steel_bare'))
    models['coupler'] = parts

    # rocker: stamped, splined onto the spindle with a nut underneath, its own ball stud at the far end
    rocker = link_plate(PV, PR, 16.0, 10.5, 'paint_black')
    rocker.append((flange_nut(PV, PLATE[0], down=True), 'steel_zinc'))
    rocker.append((Pos(PV[0], PV[1], PLATE[0] - 8.0) * Rot(180, 0, 0) * fs.threaded_rod(8.0, 2.4, 1.25, 0.0, True), 'steel_bare'))
    rocker += ball_stud(PR)
    models['rocker'] = rocker

    # the spindle's splined boss and thrust washer between the rocker and the tower
    boss = rev([(0, PLATE[1] - 0.2), (9.2, PLATE[1] - 0.2), (9.2, PLATE[1] + 1.6), (8.0, PLATE[1] + 2.4), (6.8, PLATE[1] + 2.4), (6.8, 4.6), (0, 4.6)], (PV[0], PV[1], 0))
    spl = Pos(PV[0], PV[1], 0) * spline_ring(6.8, 7.4, 24, PLATE[1] + 2.4, 4.6)
    models['rocker pin'] = [(boss, 'steel_bare'), (spl, 'steel_bare')]

    # output shaft between the gearbox and the crank, with its seal
    shaft = rev([(0, -11.2), (6.0, -11.2), (6.0, -3.5), (5.4, -3.0), (0, -3.0)], (O[0], O[1], 0))
    seal = rev([(5.8, -11.2), (9.5, -11.2), (9.5, -9.8), (7.2, -9.0), (5.8, -9.0)], (O[0], O[1], 0))
    models['crank shaft'] = [(shaft, 'steel_bare'), (seal, 'rubber')]
    return models


def spline_ring(r_in, r_out, teeth, z0, z1):
    """The knurled spline band where a boss meets the rocker's hub."""
    pts = []
    for i in range(teeth * 2):
        a = math.pi * i / teeth
        r = r_out if i % 2 == 0 else r_in
        pts.append((r * math.cos(a), r * math.sin(a)))
    return extrude(Plane.XY.offset(z0) * make_face(Polyline(*pts, close=True)), z1 - z0)


def build_motor(O):
    """The wiper motor: die cast gearbox round the worm wheel with a ribbed cover, the worm tube, the painted can
    with its crimped end cap and brush housing, the connector and three rubber mounted ears."""
    x0, y0 = O[0], O[1]
    parts = []
    gb = rev([(0, -44.5), (30.0, -44.5), (34.0, -41.0), (34.2, -16.5), (31.2, -13.2), (15.0, -12.8), (13.8, -11.2), (0, -11.2)], (x0, y0, 0))
    seam = rev([(33.0, -30.0), (35.4, -30.0), (35.6, -29.0), (35.4, -28.0), (33.0, -28.0)], (x0, y0, 0))
    parts += [(gb, 'aluminum_cast'), (seam, 'aluminum_cast')]
    # the cover's casting ribs, radial on its top face
    for k in range(8):
        a = 22.5 + 45 * k
        rib = Pos(x0, y0, -13.4) * Rot(0, 0, a) * Pos(23.5, 0, 0) * Box(14, 1.6, 1.6)
        parts.append((rib, 'aluminum_cast'))
    for k in range(5):
        a = math.radians(30 + k * 72)
        sx, sy = x0 + 26.2 * math.cos(a), y0 + 26.2 * math.sin(a)
        boss = Pos(sx, sy, -16.5) * Solid.make_cylinder(3.6, 3.3)
        parts.append((boss, 'aluminum_cast'))
        parts.append((Pos(sx, sy, -13.2) * fs.pan_phillips(4.0, 3.0, head_d=7.0), 'steel_zinc'))
    WY, WZ = y0 - 20.5, -30.0
    tube = rev([(0, 4.0), (12.5, 4.0), (14.0, -2.0), (14.0, -46.0), (16.5, -49.0), (16.5, -53.0), (0, -53.0)], (x0, WY, WZ), 'X')
    parts.append((tube, 'aluminum_cast'))
    can = rev([(0, -52.0), (27.0, -52.0), (27.5, -56.0), (27.5, -58.5), (26.2, -60.0), (26.2, -118.0), (24.0, -123.5), (15.0, -126.5),
               (8.5, -127.0), (8.0, -130.5), (5.0, -131.5), (0, -131.5)], (x0, WY, WZ), 'X')
    parts.append((can, 'paint'))
    for xk in (-74.0, -98.0):
        rib = rev([(25.8, xk - 1.2), (26.6, xk - 0.6), (26.6, xk + 0.6), (25.8, xk + 1.2)], (x0, WY, WZ), 'X')
        parts.append((rib, 'paint'))
    # the crimp tabs that hold the can on the flange
    for k in range(4):
        a = math.radians(45 + 90 * k)
        tab = Pos(x0 - 57.2, WY + 27.2 * math.cos(a), WZ + 27.2 * math.sin(a)) * Rot(math.degrees(a), 0, 0) * Box(3.0, 0.9, 4.0)
        parts.append((tab, 'steel_zinc'))
    # brush housing bump on the end cap and the ground strap
    bump = Pos(x0 - 120.0, WY, WZ + 18.0) * Box(10, 12, 7)
    parts.append((soft(fillet, bump, bump.edges(), 2.0), 'paint'))
    for k, ang in enumerate((25, 125, 265)):
        a = math.radians(ang)
        ex_, ey = x0 + 50.0 * math.cos(a), y0 + 50.0 * math.sin(a)
        ear = extrude(Plane.XY.offset(-42.5) * hull_link(np.array([x0 + 22.0 * math.cos(a), y0 + 22.0 * math.sin(a)]), np.array([ex_, ey]), 11.5, 9.8), 4.5)
        ear = ear - Pos(ex_, ey, -50) * Solid.make_cylinder(5.2, 20)
        ear = soft(fillet, ear, ear.edges().filter_by(Axis.Z, reverse=True), 0.6)
        gr = rev([(4.2, -47.8), (9.0, -47.8), (9.2, -44.0), (6.6, -43.0), (6.6, -37.5), (9.0, -36.5), (8.8, -33.0), (4.2, -33.0)], (ex_, ey, 0))
        sl = rev([(3.0, -48.4), (5.0, -48.4), (5.0, -32.4), (3.0, -32.4)], (ex_, ey, 0))
        parts += [(ear, 'aluminum_cast'), (gr, 'rubber'), (sl, 'steel_zinc')]
    con = Pos(x0, y0 - 39.25, -29.25) * Box(24, 16.5, 17.5)
    con = soft(fillet, con, con.edges(), 1.2)
    latch = Pos(x0, y0 - 48.1, -25.75) * Box(8, 2.2, 8.5)
    latch = soft(fillet, latch, latch.edges(), 0.6)
    parts += [(con, 'plastic_black'), (latch, 'plastic_black')]
    return parts


def build_tower(PV, PR):
    """The cast pivot tower round the spindle, bolted through two rubber mounted ears."""
    x, y = PV[0], PV[1]
    parts = []
    tw = rev([(6.8, 5.0), (12.5, 5.0), (13.5, 6.5), (12.8, 12.0), (11.2, 27.0), (10.0, 30.0), (7.0, 30.6)], (x, y, 0))
    parts.append((tw, 'aluminum_cast'))
    away = (PV - PR) / np.linalg.norm(PV - PR)
    side = np.array([-away[1], away[0], 0])
    for s in (1, -1):
        e = PV + side * s * 34.0 + away * 6.0
        ear = extrude(Plane.XY.offset(9.0) * hull_link(PV[:2], e[:2], 12.0, 8.5), 4.5)
        ear = ear - Pos(e[0], e[1], 0) * Solid.make_cylinder(4.5, 30)
        ear = soft(fillet, ear, ear.edges().filter_by(Axis.Z, reverse=True), 0.6)
        gr = rev([(3.6, 6.0), (7.8, 6.0), (8.0, 8.8), (6.0, 9.4), (6.0, 13.2), (8.0, 13.8), (7.8, 16.5), (3.6, 16.5)], (e[0], e[1], 0))
        bolt = Pos(e[0], e[1], 16.5) * fs.hex_prism(10.0, 4.0) + Pos(e[0], e[1], 16.5) * Solid.make_cylinder(6.8, 1.2)
        parts += [(ear, 'aluminum_cast'), (gr, 'rubber'), (bolt, 'steel_zinc')]
    spin = rev([(0, 3.0), (6.0, 3.0), (6.0, 33.0), (5.2, 34.0), (0, 34.0)], (x, y, 0))
    parts.append((spin, 'steel_bare'))
    return parts


def build_arm():
    """The wiper arm in its own frame (x along the arm from the spindle, z up from the arm's seat): the cast head
    under its cap, the hinge, the pressed channel arm and the hook the blade clips to."""
    parts = []
    head = extrude(Plane.XY.offset(1.0) * hull_link(np.array([0.0, 0.0]), np.array([58.0, 0.0]), 14.8, 8.2), 12.5)
    head = soft(fillet, head, head.edges().group_by(Axis.Z)[-1], 2.2)
    head = soft(fillet, head, head.edges().group_by(Axis.Z)[0], 1.0)
    parts.append((head, 'paint_black'))
    cap = rev([(0, 12.8), (12.8, 12.8), (12.8, 15.0), (11.0, 17.2), (6.0, 18.4), (0, 18.6)])
    parts.append((cap, 'plastic_black'))
    hinge = Pos(52.0, 0, 8.5) * Rot(90, 0, 0) * Pos(0, 0, -11.5) * Solid.make_cylinder(2.6, 23.0)
    parts.append((hinge, 'steel_bare'))
    secs = []
    for t, (a, b, zc) in ((50.0, (13.6, 7.6, 9.0)), (180.0, (11.6, 6.6, 10.4)), (310.0, (10.4, 5.8, 10.0)), (442.0, (9.2, 5.2, 8.8))):
        secs.append(Plane(origin=(t, 0, zc), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * RectangleRounded(a, b, min(a, b) * 0.35))
    bar = loft(secs)
    # the channel is open underneath: the pressed section's hollow
    parts.append((bar, 'paint_black'))
    hook = Pos(445.0, 0, 9.4) * Box(30.0, 8.4, 2.2)
    hook = soft(fillet, hook, hook.edges().filter_by(Axis.Y), 0.8)
    parts.append((hook, 'steel_bare'))
    # the arm spring's end, seen through the head's slot
    for k in range(4):
        coil = Pos(22.0 + k * 3.2, 0, 7.0) * Rot(0, 90, 0) * (Solid.make_torus(3.0, 0.7))
        parts.append((coil, 'steel_bare'))
    return parts


def build_blade():
    """A beam blade in its own frame: rubber squeegee, spoiler, centre adapter with its button and end caps."""
    half = 236.0
    parts = []

    def strip(w, h, zc, mat, y=0.0, n_end=6.0):
        body = Pos(0, y, zc) * Box(2 * half - 2 * n_end, w, h)
        body = soft(fillet, body, body.edges().filter_by(Axis.X), min(w, h) * 0.45)
        ends = []
        for s in (-1, 1):
            e = Pos(s * (half - n_end), y, zc) * Rot(0, 90 * s, 0) * Solid.make_cone(min(w, h) / 2, min(w, h) / 4, n_end)
            ends.append(e)
        return [(body, mat)]
    parts += strip(1.8, 4.4, 2.2, 'rubber')
    parts += strip(5.2, 2.6, 5.2, 'rubber')
    spo = Pos(0, 0, 9.0) * Box(2 * half - 4, 8.8, 5.6)
    spo = soft(fillet, spo, spo.edges().filter_by(Axis.X), 2.4)
    fin = Pos(0, 1.2, 12.4) * Box(2 * half - 12, 2.2, 4.8)
    fin = soft(fillet, fin, fin.edges().filter_by(Axis.X), 0.9)
    parts += [(spo, 'plastic_black'), (fin, 'plastic_black')]
    adapter = Pos(0, 0, 12.1) * Box(52, 10.4, 9.8)
    adapter = soft(fillet, adapter, adapter.edges(), 1.8)
    button = Pos(0, 0, 17.05) * Box(16, 7.6, 1.1)
    button = soft(fillet, button, button.edges().filter_by(Axis.Z), 1.0)
    parts += [(adapter, 'paint_black'), (button, 'plastic_black')]
    for s in (-1, 1):
        c = Pos(s * (half - 1.0), 0, 7.6) * Box(10, 8, 7.2)
        c = soft(fillet, c, c.edges(), 1.6)
        parts.append((c, 'plastic_black'))
    return parts


HARDWARE = {'steel_zinc', 'steel_bare', 'steel_chrome', 'rubber'}


def nodes_for(slug, parts, fine=False):
    body = plain([(s, m) for s, m in parts if m not in HARDWARE], ('steel_zinc', 'steel_bare', 'steel_chrome'))
    hw = plain([(s, m) for s, m in parts if m in HARDWARE], ('steel_zinc', 'steel_bare', 'steel_chrome'))
    kids = []
    if body:
        kids.append(Node(f'{slug}-body', body, tol=0.07 if fine else 0.12, ang=0.3))
    if hw:
        kids.append(Node(f'{slug}-hardware', hw, tol=0.08, ang=0.45))
    return [Node(slug, children=kids)]


def main():
    sc = ex.scene(f'{KEY}/output')
    idx = ex.index(sc)
    home = sc.home()

    def end(name, sign):
        o = sc.objects[idx[name]]
        M = ex.pose_mm(sc, idx[name], home)
        p = M @ np.array([0, 0, sign * o['size'][1] * 1000.0, 1.0])
        return np.array([p[0], p[1], 0.0])
    O, PC, PV, PR = end('crank', 1), end('crank', -1), end('rocker', 1), end('rocker', -1)
    links = build_links(O, PC, PV, PR)
    models = dict(links)
    models['crank motor'] = build_motor(O)
    models['wiper spindle'] = build_tower(PV, PR)
    files = {'crank': 'crank', 'coupler': 'coupler', 'rocker': 'rocker', 'coupler pin': 'ball-stud', 'rocker pin': 'spindle-boss',
             'crank shaft': 'output-shaft', 'crank motor': 'motor', 'wiper spindle': 'pivot-tower', 'wiper arm': 'arm',
             'wiper blade': 'blade'}
    entries, total = [], 0
    for name in ['crank', 'coupler pin', 'coupler', 'rocker', 'rocker pin', 'crank shaft', 'crank motor', 'wiper spindle']:
        local = ex.into(models[name], ex.pose_mm(sc, idx[name], home))
        rel = f'{KEY}/{files[name]}.glb'
        total += export(rel, nodes_for(files[name], local, fine=name in ('crank', 'coupler', 'rocker')), unit=UNIT)
        e = {'object': name, 'file': rel}
        if name in ('crank', 'crank motor'):
            e['tint'] = ['paint']
        if name == 'crank':
            e['anchors'] = {'hub': [0, 0, 0.04], 'pin': [0, 0, -0.04]}
        if name == 'coupler':
            e['anchors'] = {'crank_end': [0, 0, 0.15], 'rocker_end': [0, 0, -0.15]}
        if name == 'rocker':
            e['anchors'] = {'pivot': [0, 0, 0.055], 'coupler_end': [0, 0, -0.055]}
        entries.append(e)
    # the arm and the blade are built in their own frames
    for name, parts in (('wiper arm', build_arm()), ('wiper blade', build_blade())):
        rel = f'{KEY}/{files[name]}.glb'
        total += export(rel, nodes_for(files[name], parts), unit=UNIT)
        entries.append({'object': name, 'file': rel})
    entries.append({'object': 'windshield', 'file': f'{KEY}/glass.glb'})
    order = ['crank', 'coupler pin', 'coupler', 'rocker', 'rocker pin', 'crank shaft', 'crank motor', 'wiper spindle', 'wiper arm', 'wiper blade', 'windshield']
    entries.sort(key=lambda e: order.index(e['object']))
    write_manifest(KEY, {'replace': entries})
    print(f'[cad] {KEY} {total / 1024:.0f} KB')


if __name__ == '__main__':
    main()
