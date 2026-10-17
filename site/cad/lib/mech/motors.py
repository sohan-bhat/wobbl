"""Motors and the pulleys on their shafts, at catalogue dimensions.

NEMA 17 hybrid stepper (17HS4401 class): 42.3 mm square, corners chamfered to 50 mm across, M3 holes 4.5 mm deep on
a 31 mm square, a 22 mm pilot 2 mm high, a 5 mm shaft 24 mm long with a 0.5 mm deep D flat over 15 mm, black
end caps either side of the laminated stack, the rear cap's four screws and the JST PH connector.
80 mm frame AC servo (750 W class): 80 mm flange, four 6.6 mm holes on a 90 mm circle, 70 mm pilot 3 mm high,
a 19 mm keyed shaft, the encoder housing and its two connectors.
GT2 timing pulley: 2 mm pitch, 20 teeth (12.73 mm pitch diameter, 12.22 mm outside), 16 mm flanges, 7 mm tooth
width for a 6 mm belt, a hub with two M3 set screws at 90 degrees.

Frame: the shaft is +z, the motor's mounting face is z = 0 (the pilot stands above it) and the body runs to -z.
Functions return [(shape, material)] lists, a pulley's with its hub face at z = 0 rising along +z.
"""
import math

from build123d import (Axis, Box, Circle, Plane, Polyline, Pos, Rectangle, RegularPolygon, Rot, Solid, chamfer,
                       extrude, fillet, make_face)

from . import fasteners as fs

import looks

looks.add('label_white', color=(0.7, 0.7, 0.68), metal=0.0, rough=0.45)
looks.add('connector_white', color=(0.78, 0.76, 0.7), metal=0.0, rough=0.4)
looks.add('servo_paint', color=(0.045, 0.047, 0.05), metal=0.0, rough=0.38)


def _square(s, c, z0, h):
    """A square of side s with corners chamfered by c, extruded from z0 by h."""
    a = s / 2
    pts = [(a - c, -a), (a, -a + c), (a, a - c), (a - c, a), (-a + c, a), (-a, a - c), (-a, -a + c), (-a + c, -a)]
    return extrude(Plane.XY.offset(z0) * make_face(Polyline(*pts, close=True)), h)


def nema17(length=40.0, shaft_len=24.0, flat_len=15.0, connector=True):
    """A NEMA 17 stepper: front cap, laminated stack, rear cap, pilot, D shaft, rear screws, connector, label."""
    s, c = 42.3, 5.6
    front, rear = 8.5, 9.0
    out = []
    fc = _square(s, c, -front, front)
    fc = chamfer(fc.edges().group_by(Axis.Z)[-1], 0.6)
    for sx in (-1, 1):
        for sy in (-1, 1):
            fc = fc - Pos(sx * 15.5, sy * 15.5, -4.5) * Solid.make_cylinder(1.25, 4.6)
            fc = fc - Pos(sx * 15.5, sy * 15.5, -0.3) * Solid.make_cone(1.25, 1.65, 0.31)
    pilot = Solid.make_cylinder(11.0, 2.0)
    pilot = chamfer(pilot.edges().group_by(Axis.Z)[-1], 0.3)
    pilot = pilot - Pos(0, 0, 1.0) * Solid.make_cylinder(3.6, 1.1)
    out.append((fc + pilot, 'aluminum_anodized'))
    z0, z1 = -length + rear, -front
    stack = _square(s - 0.3, c, z0, z1 - z0)
    for k in range(1, 12):
        z = z0 + (z1 - z0) * k / 12
        ring = _square(s + 1, c, z - 0.15, 0.3) - _square(s - 0.7, c, z - 1, 2)
        stack = stack - ring
    out.append((stack, 'steel_stainless'))
    rc = _square(s, c, -length, rear)
    rc = chamfer(rc.edges().group_by(Axis.Z)[0], 0.8)
    for sx in (-1, 1):
        for sy in (-1, 1):
            rc = rc - Pos(sx * 15.5, sy * 15.5, -length - 0.1) * Solid.make_cylinder(3.1, 3.1)
    out.append((rc, 'aluminum_anodized'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((Pos(sx * 15.5, sy * 15.5, -length + 3.0) * Rot(180, 0, 0) * fs.socket_cap('M3', 2, threads=False, lite=True), 'steel_black_oxide'))
    # the shaft with its D flat, ground steel
    shaft = Solid.make_cylinder(2.5, shaft_len + 1.0, Plane.XY.offset(-1.0))
    shaft = chamfer(shaft.edges().group_by(Axis.Z)[-1], 0.3)
    shaft = shaft - Pos(2.0 + 2.0, 0, shaft_len - flat_len / 2) * Box(4.0, 6, flat_len + 0.01)
    out.append((shaft, 'steel_bare'))
    if connector:
        con = Pos(0, -s / 2 - 2.3, -length + rear / 2) * Box(12.4, 4.6, 6.0)
        con = con - Pos(0, -s / 2 - 4.7, -length + rear / 2) * Box(11.0, 2.0, 4.6)
        out.append((con, 'connector_white'))
    lab = Pos(0, s / 2 - 0.12, (z0 + z1) / 2) * Box(30, 0.4, (z1 - z0) * 0.8)
    out.append((lab, 'label_white'))
    return out


def servo80(length=150.0, shaft_len=35.0, shaft_d=19.0):
    """An 80 mm frame AC servo: the cast flange with its pilot and four holes, the painted body, the encoder cap,
    the power and encoder connectors and the keyed shaft."""
    out = []
    fl = _square(80.0, 9.0, -11.0, 11.0)
    fl = fillet(fl.edges().filter_by(Axis.Z), 3.0)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        fl = fl - Pos(45 * math.cos(a), 45 * math.sin(a), -12) * Solid.make_cylinder(3.3, 14)
        fl = fl - Pos(45 * math.cos(a), 45 * math.sin(a), -12) * Solid.make_cylinder(6.0, 5.5)
    pilot = Solid.make_cylinder(35.0, 3.0)
    pilot = chamfer(pilot.edges().group_by(Axis.Z)[-1], 0.5)
    seal = Pos(0, 0, 3.0) * Solid.make_cylinder(12.0, 0.8)
    out += [(fl + pilot, 'aluminum_cast'), (seal, 'rubber')]
    body = Pos(0, 0, -11.0 - (length - 60) / 2) * Box(74, 74, length - 60)
    body = fillet(body.edges().filter_by(Axis.Z), 6.0)
    # cooling flutes along the body
    for s_ in (-1, 1):
        for k in range(-2, 3):
            body = body - Pos(s_ * 37.6, k * 11.0, -11.0 - (length - 60) / 2) * Box(2.4, 3.2, length - 70)
            body = body - Pos(k * 11.0, s_ * 37.6, -11.0 - (length - 60) / 2) * Box(3.2, 2.4, length - 70)
    out.append((body, 'servo_paint'))
    enc = Pos(0, 0, -length + 49 - 24) * Box(72, 72, 48)
    enc = fillet(enc.edges().filter_by(Axis.Z), 8.0)
    enc = fillet(enc.edges().group_by(Axis.Z)[0], 4.0)
    out.append((enc, 'servo_paint'))
    for k, (w, d) in enumerate(((24.0, 18.0), (18.0, 14.0))):
        x = -12 + k * 26
        con = Pos(x, 36 + 6, -length + 30) * Box(w, 12, d)
        con = fillet(con.edges().filter_by(Axis.Y), 2.0)
        nut_ = Pos(x, 36 + 12.5, -length + 30) * Rot(-90, 0, 0) * Solid.make_cylinder(d * 0.42, 6)
        out += [(con, 'aluminum_cast'), (nut_, 'steel_bare')]
    shaft = Solid.make_cylinder(shaft_d / 2, shaft_len + 3.0)
    shaft = chamfer(shaft.edges().group_by(Axis.Z)[-1], 0.8)
    shaft = shaft - Pos(shaft_d / 2 - 1.75, 0, shaft_len - 13) * Box(6.0, 6.0, 22.0)
    key = Pos(shaft_d / 2 - 1.0, 0, shaft_len - 13) * Box(6.0, 6.0, 22.0)
    key = fillet(key.edges().filter_by(Axis.X), 2.9)
    out += [(shaft, 'steel_bare'), (key, 'steel_bare')]
    return out


def gt2_pulley(teeth=20, bore=5.0, width=7.0, flange_d=16.0, hub_d=13.0, hub_len=7.0, set_screws=2):
    """A GT2 timing pulley: hub on z = 0 with its set screws, then the toothed band between two flanges."""
    pd = 2.0 * teeth / math.pi
    od = pd - 0.508
    out = []
    hub = Solid.make_cylinder(hub_d / 2, hub_len)
    hub = chamfer(hub.edges().group_by(Axis.Z)[0], 0.5)
    fl_t = 1.0
    z = hub_len
    f1 = Pos(0, 0, z) * Solid.make_cylinder(flange_d / 2, fl_t)
    band = Pos(0, 0, z + fl_t) * Solid.make_cylinder(od / 2, width)
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        band = band - Pos(od / 2 * math.cos(a), od / 2 * math.sin(a), z + fl_t - 0.1) * Solid.make_cylinder(0.56, width + 0.2)
    f2 = Pos(0, 0, z + fl_t + width) * Solid.make_cylinder(flange_d / 2, fl_t)
    f2 = chamfer(f2.edges().group_by(Axis.Z)[-1], 0.3)
    body = hub + f1 + band + f2
    total = z + 2 * fl_t + width
    body = body - Pos(0, 0, -1) * Solid.make_cylinder(bore / 2, total + 2)
    for k in range(set_screws):
        a = math.radians(90 * k)
        body = body - Pos(0, 0, hub_len / 2) * Rot(0, 0, math.degrees(a)) * Rot(0, 90, 0) * Solid.make_cylinder(1.25, hub_d)
    out.append((body, 'aluminum_machined'))
    for k in range(set_screws):
        a = math.degrees(math.radians(90 * k))
        ss = Pos(0, 0, hub_len / 2) * Rot(0, 0, a) * Pos(hub_d / 2 - 0.4, 0, 0) * Rot(0, 90, 0) * Rot(180, 0, 0) * fs.set_screw('M3', 3)
        out.append((ss, 'steel_black_oxide'))
    return out, total


def htd_pulley(teeth=20, bore=19.0, belt=15.0, pitch=5.0, hub_d=None, hub_len=12.0, keyway=6.0):
    """An HTD 5M timing pulley for a 15 mm belt, keyed to its shaft, with two set screws over the key."""
    pd = pitch * teeth / math.pi
    od = pd - 1.144
    hub_d = hub_d or min(od - 4, bore + 14)
    out = []
    hub = Solid.make_cylinder(hub_d / 2, hub_len)
    hub = chamfer(hub.edges().group_by(Axis.Z)[0], 0.8)
    fw = belt + 2.0
    fl_t = 1.5
    f1 = Pos(0, 0, hub_len) * Solid.make_cylinder(od / 2 + 3.0, fl_t)
    band = Pos(0, 0, hub_len + fl_t) * Solid.make_cylinder(od / 2, fw)
    for i in range(teeth):
        a = 2 * math.pi * i / teeth
        band = band - Pos(od / 2 * math.cos(a), od / 2 * math.sin(a), hub_len + fl_t - 0.1) * Solid.make_cylinder(1.6, fw + 0.2)
    f2 = Pos(0, 0, hub_len + fl_t + fw) * Solid.make_cylinder(od / 2 + 3.0, fl_t)
    total = hub_len + 2 * fl_t + fw
    body = hub + f1 + band + f2
    body = body - Pos(0, 0, -1) * Solid.make_cylinder(bore / 2, total + 2)
    body = body - Pos(bore / 2, 0, total / 2) * Box(keyway * 0.9, keyway, total + 2)
    out.append((body, 'aluminum_machined'))
    for a in (0, 90):
        ss = Pos(0, 0, hub_len / 2) * Rot(0, 0, a) * Pos(hub_d / 2 - 0.5, 0, 0) * Rot(0, 90, 0) * Rot(180, 0, 0) * fs.set_screw('M6', 6)
        out.append((ss, 'steel_black_oxide'))
    return out, total


def belt_strand(width, length, thickness=1.4, pitch=2.0, tooth=0.75):
    """A straight run of toothed belt along +x from the origin, across y: the back on z = 0, teeth below it."""
    pts = [(0, 0), (length, 0), (length, -thickness)]
    n = int(length / pitch)
    for i in range(n - 1, -1, -1):
        x = pitch * (i + 0.5)
        pts += [(x + pitch * 0.3, -thickness), (x + pitch * 0.22, -thickness - tooth), (x - pitch * 0.22, -thickness - tooth),
                (x - pitch * 0.3, -thickness)]
    pts.append((0, -thickness))
    face = make_face(Plane.XZ * Polyline(*pts, close=True))
    return Pos(0, width / 2, 0) * extrude(face, width)


looks.add('motor_bell', color=(0.12, 0.02, 0.02), metal=1.0, rough=0.32)
looks.add('copper_winding', color=(0.72, 0.36, 0.18), metal=1.0, rough=0.35)


def bldc(stator_d=22.0, stator_h=7.0, can_d=28.0, mount=16.0):
    """An outrunner drone motor (2207 class): the base with its 16 mm M3 pattern, copper windings seen through the
    bell's windows, the anodised bell, the 5 mm shaft with its M5 thread. Base on z = 0, shaft up +z.
    Returns (parts, shaft top z)."""
    out = []
    base = Solid.make_cylinder(can_d / 2 - 0.3, 4.0)
    base = chamfer(base.edges().group_by(Axis.Z)[0], 0.5)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        base = base - Pos(mount / 2 * math.sqrt(2) * math.cos(a), mount / 2 * math.sqrt(2) * math.sin(a), -1) * Solid.make_cylinder(1.6, 6)
    for k in range(4):
        a = math.radians(90 * k)
        base = base - Pos(9.0 * math.cos(a), 9.0 * math.sin(a), -1) * Solid.make_cylinder(3.0, 6)
    out.append((base, 'aluminum_anodized'))
    wind = Pos(0, 0, 4.0) * Solid.make_cylinder(stator_d / 2, stator_h + 4.0)
    out.append((wind, 'copper_winding'))
    h = stator_h + 9.0
    bell = Pos(0, 0, 4.6) * (Solid.make_cylinder(can_d / 2, h) - Solid.make_cylinder(can_d / 2 - 1.4, h - 1.6))
    bell = chamfer(bell.edges().group_by(Axis.Z)[-1], 0.8)
    for k in range(6):
        a = 60 * k + 30
        win = Box(7.0, 5.0, 3.0)
        win = fillet(win.edges().filter_by(Axis.Z), 1.5)
        bell = bell - Pos(0, 0, 4.6 + h - 0.8) * Rot(0, 0, a) * Pos(8.0, 0, 0) * win
        slot = Box(3.0, 5.5, h * 0.5)
        slot = fillet(slot.edges().filter_by(Axis.X), 1.2)
        bell = bell - Pos(0, 0, 4.6 + h * 0.45) * Rot(0, 0, a) * Pos(can_d / 2, 0, 0) * slot
    out.append((bell, 'motor_bell'))
    z = 4.6 + h
    shaft = Pos(0, 0, z - 2) * Solid.make_cylinder(2.5, 12.0)
    shaft = chamfer(shaft.edges().group_by(Axis.Z)[-1], 0.4)
    out.append((shaft, 'steel_bare'))
    return out, z


def propeller(diameter=178.0, hub_d=13.0, hub_h=7.0, pitch_deg=(24.0, 9.0), chord=(16.0, 21.0, 9.0), blades=2):
    """A two blade propeller: the hub with its 5 mm bore, each blade lofted through twisted, tapering airfoil
    sections from the hub to the rounded tip. Hub's base on z = 0."""
    from build123d import Ellipse, Plane, loft
    out = []
    hub = Solid.make_cylinder(hub_d / 2, hub_h)
    hub = chamfer(hub.edges(), 0.8)
    hub = hub - Pos(0, 0, -1) * Solid.make_cylinder(2.55, hub_h + 2)
    R = diameter / 2
    blades_s = None
    for b in range(blades):
        secs = []
        stations = [hub_d / 2 - 1.0, R * 0.25, R * 0.5, R * 0.75, R * 0.97]
        for i, r in enumerate(stations):
            t = (r - stations[0]) / (stations[-1] - stations[0])
            c = chord[0] + (chord[1] - chord[0]) * min(t / 0.35, 1.0) if t < 0.35 else chord[1] + (chord[2] - chord[1]) * (t - 0.35) / 0.65
            th = max(0.9, 3.2 * (1 - t) + 0.9 * t)
            pitch = math.radians(pitch_deg[0] + (pitch_deg[1] - pitch_deg[0]) * t)
            pl = Plane(origin=(0, r, hub_h / 2), x_dir=(math.cos(pitch), 0, math.sin(pitch)), z_dir=(0, 1, 0))
            secs.append(pl * Ellipse(c / 2, th / 2))
        bl = loft(secs)
        bl = Rot(0, 0, 360.0 * b / blades) * bl
        blades_s = bl if blades_s is None else blades_s + bl
    out.append((hub + blades_s, 'plastic_black'))
    return out
