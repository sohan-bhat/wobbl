"""A loudspeaker cabinet's parts, from driver datasheet drawings: a 6.5 inch woofer with its stamped steel basket
flange, half roll rubber surround, polypropylene cone and dust cap, a soft dome tweeter on its faceplate, the
veneered cabinet with rounded edges, cutouts, terminal cup and feet. Builders return [(shape, material)] Z up in
millimetres; a driver looks along -y from its mounting face at y = 0.
"""
import math

from build123d import (Align, Axis, Box, Compound, Cylinder, Pos, Rot, Sphere, Torus, chamfer, fillet, revolve,
                       Polyline, make_face, Plane)

import electronics  # noqa: F401  (ferrite)
import looks
from battery import torus_mesh

looks.add('cone_pp', color=(0.025, 0.025, 0.028), metal=0.0, rough=0.55)
looks.add('surround', color=(0.02, 0.02, 0.02), metal=0.0, rough=0.75)
looks.add('basket', color=(0.03, 0.03, 0.033), metal=0.3, rough=0.4)
looks.add('dome_silk', color=(0.05, 0.05, 0.05), metal=0.0, rough=0.85)


def _lathe(pts):
    """A solid of revolution about z from a (r, z) profile."""
    face = make_face(Polyline(*[(r, 0, z) for r, z in pts], close=True))
    return revolve(face, Axis.Z)


def _facing(parts):
    """A driver built along +z, turned to look along -y from y = 0."""
    from mech.util import place
    return place(parts, Rot(90, 0, 0))


def woofer(r_frame=82.0):
    """A 6.5 inch woofer: the flange with its gasket ring and four screws, the half roll surround, the cone and the
    dust cap, built looking along -y with the flange's back face at y = 0."""
    out = []
    rf = r_frame
    flange = _lathe([(rf * 0.86, 0), (rf, 0), (rf, 4.0), (rf * 0.86, 4.0)])
    flange = flange - Compound([Rot(0, 0, 45 + 90 * k) * Pos(rf * 0.93, 0, 0) * Cylinder(2.2, 20) for k in range(4)])
    out.append((flange, 'basket'))
    from mech import fasteners as fs
    for k in range(4):
        a = math.radians(45 + 90 * k)
        s = fs.button_head('M4', 12)
        out.append((Pos(math.cos(a) * rf * 0.93, math.sin(a) * rf * 0.93, 4.0) * s, 'steel_black_oxide'))
    rs = rf * 0.78
    out.append((torus_mesh(rs, rf * 0.075, 4.0, seg=96, ring=12, arc=None), 'surround'))
    cone = _lathe([(rs - rf * 0.075, 4.0), (rf * 0.3, -26.0), (rf * 0.3, -27.5), (rs - rf * 0.075, 2.5)])
    out.append((cone, 'cone_pp'))
    rc, hc, zb = rf * 0.3, 8.0, -27.5
    R = (rc * rc + hc * hc) / (2 * hc)
    cap = Pos(0, 0, zb + hc - R) * Sphere(R) & Pos(0, 0, zb + 100) * Box(400, 400, 200)
    out.append((cap, 'cone_pp'))
    # behind the cone: the basket's four legs, the ferrite ring between its plates, and the vented pole
    for k in range(4):
        a = 90 * k
        leg = Rot(0, 0, a) * Pos(rf * 0.67, 0, -22.0) * Rot(0, 34, 0) * Box(4.0, 9.0, 50.0)
        out.append((leg, 'basket'))
    ring = Pos(0, 0, -44.0) * Cylinder(rf * 0.56, 4.0)
    out.append((ring, 'basket'))
    magnet = Pos(0, 0, -55.0) * (Cylinder(rf * 0.55, 16.0) - Cylinder(rf * 0.22, 20))
    out.append((chamfer(magnet.edges(), 1.0), 'ferrite'))
    back = Pos(0, 0, -65.0) * (Cylinder(rf * 0.52, 4.0) - Cylinder(5.0, 6))
    out.append((back, 'steel_zinc'))
    return _facing(out)


def tweeter(r_plate=48.0):
    """A soft dome tweeter: a round faceplate with a waveguide, the silk dome and two screws."""
    out = []
    plate = _lathe([(0, 0), (r_plate, 0), (r_plate, 3.0), (r_plate - 3.0, 6.0), (18.0, 6.0), (14.0, 2.0), (0, 2.0)])
    out.append((plate, 'basket'))
    dome = Pos(0, 0, -4.0) * Sphere(14.0) & Pos(0, 0, 102.0) * Box(400, 400, 200)
    out.append((dome, 'dome_silk'))
    can = Pos(0, 0, -12.0) * Cylinder(24.0, 24.0)
    out.append((chamfer(can.edges().group_by(Axis.Z)[0], 2.0), 'basket'))
    from mech import fasteners as fs
    for k in range(2):
        a = math.radians(90 + 180 * k)
        s = fs.button_head('M3', 10)
        out.append((Pos(math.cos(a) * (r_plate - 7), math.sin(a) * (r_plate - 7), 6.0 - 1.5) * s, 'steel_black_oxide'))
    return _facing(out)


def cabinet(x0, y0, z0, x1, y1, z1, wall, cutouts=(), brace=None, open_sides=('+x', '+z'), gap=3.0):
    """The cabinet's shell outside an inner cavity of wall thickness: rounded vertical and top edges, the driver
    cutouts [(x, z, r)] through the front (y = y0) baffle, the cross brace (z0, z1, hole radius) with its hole,
    a terminal cup on the back and four rubber feet. The walls in open_sides are cut away, as a cutaway drawing
    does, so the air inside and the standing wave slice stay in view. The inner faces stand gap back from the
    cavity's, since the stage draws a see through cavity's far faces giving way to anything they touch."""
    w, d, h = x1 - x0, y1 - y0, z1 - z0
    outer = Pos(x0 + w / 2, y0 + d / 2, z0 + h / 2) * Box(w, d, h)
    outer = fillet(outer.edges().filter_by(Axis.Z), 6.0)
    outer = fillet(outer.edges().group_by(Axis.Z)[-1], 4.0)
    inner = Pos(x0 + w / 2, y0 + d / 2, z0 + h / 2) * Box(w - 2 * wall + 2 * gap, d - 2 * wall + 2 * gap, h - 2 * wall + 2 * gap)
    shell = outer - inner
    holes = [Pos(x, y0 + wall / 2, z) * Cylinder(r, wall * 3, rotation=(90, 0, 0)) for x, z, r in cutouts]
    cup = Pos(x0 + w / 2, y1 - wall / 2, z0 + h * 0.2) * Box(70, wall * 3, 50)
    shell = shell - Compound(holes + [cup])
    cut = []
    if '+x' in open_sides:
        cut.append(Pos(x1 - wall / 2, y0 + d / 2, z0 + h / 2 + wall) * Box(wall * 3, d - 2 * wall, h))
    if '+z' in open_sides:
        cut.append(Pos(x0 + w / 2 + wall, y0 + d / 2, z1 - wall / 2) * Box(w, d - 2 * wall, wall * 3))
    if cut:
        shell = shell - Compound(cut)
    out = [(shell, 'wood')]
    if brace:
        bz0, bz1, br = brace
        b = Pos(x0 + w / 2, y0 + d / 2, (bz0 + bz1) / 2) * Box(w - 2 * wall + 2 * gap, d - 2 * wall + 2 * gap, bz1 - bz0 - 2 * gap)
        b = b - Pos(x0 + w / 2, y0 + d / 2, (bz0 + bz1) / 2) * Cylinder(br + gap, bz1 - bz0 + 2)
        out.append((b, 'wood'))
    plate = Pos(x0 + w / 2, y1 - 1.5, z0 + h * 0.2) * Box(78, 3, 58)
    plate = plate - Pos(x0 + w / 2, y1 - 6, z0 + h * 0.2) * Box(66, 10, 46)
    out.append((plate, 'plastic_black'))
    for k, dx in enumerate((-12, 12)):
        post = Pos(x0 + w / 2 + dx, y1 - wall + 2, z0 + h * 0.2) * Cylinder(5.0, 14, rotation=(90, 0, 0))
        out.append((post, 'paint_red' if k == 0 else 'plastic_black'))
    for sx in (x0 + 25, x1 - 25):
        for sy in (y0 + 25, y1 - 25):
            foot = Pos(sx, sy, z0) * Cylinder(10.0, 6.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
            out.append((chamfer(foot.edges().group_by(Axis.Z)[0], 1.5), 'rubber'))
    return out
