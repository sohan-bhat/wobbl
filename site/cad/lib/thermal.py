"""Parts for the thermal scenes, from datasheet drawings: an axial fan with its twisted blades, frame, struts, screws
and leads, a duct adapter, a flip chip BGA package, a motherboard patch, barbed hose fittings with worm drive
clamps and hoses, and a 62 mm half bridge IGBT module. Builders return [(shape, material)] Z up in millimetres.
"""
import math

import numpy as np
from build123d import (Align, Axis, Box, Compound, Cone, Cylinder, Plane, Pos, Rectangle, RectangleRounded, Rot,
                       Sphere, chamfer, extrude, fillet, loft)

import looks
from electronics import text

looks.add('fan_plastic', color=(0.018, 0.018, 0.02), metal=0.0, rough=0.5, rvar=(0.42, 0.58), tiles=80)
looks.add('fan_label', color=(0.22, 0.23, 0.25), metal=0.0, rough=0.35)
looks.add('wire_red', color=(0.5, 0.02, 0.02), metal=0.0, rough=0.4)
looks.add('wire_black', color=(0.012, 0.012, 0.012), metal=0.0, rough=0.4)
looks.add('ihs_nickel', color=(0.7, 0.7, 0.68), metal=1.0, rough=0.22, normal='brushed_n', nstr=0.15,
          rvar=(0.16, 0.28), tiles=60)
looks.add('substrate', color=(0.03, 0.09, 0.04), metal=0.0, rough=0.35)
looks.add('solder_ball', color=(0.72, 0.72, 0.7), metal=1.0, rough=0.3)
looks.add('mlcc', color=(0.32, 0.24, 0.14), metal=0.0, rough=0.5)
looks.add('module_grey', color=(0.035, 0.037, 0.04), metal=0.0, rough=0.55, rvar=(0.48, 0.62), tiles=40)
looks.add('copper_nickel', color=(0.72, 0.68, 0.62), metal=1.0, rough=0.3)
looks.add('hose', color=(0.015, 0.015, 0.016), metal=0.0, rough=0.62, rvar=(0.55, 0.72), tiles=40)


# ---------------------------------------------------------------- fan

def fan(size=40.0, depth=20.0, blades=7, hole_pitch=32.0, screws=True, leads=None):
    """An axial fan blowing along +z: the moulded frame with corner flanges and screw holes, a round venturi, the
    hub with its label on the intake side, twisted blades, four struts on the outlet side and the motor leads
    leaving along strut one toward leads (a point in the fan's frame), if given."""
    out = []
    h = depth
    r_bore = size * 0.475
    r_hub = size * 0.3
    fl = min(3.2, depth * 0.17)             # flange thickness at each face
    frame = Box(size, size, h)
    frame = fillet(frame.edges().filter_by(Axis.Z), size * 0.08)
    frame = frame - Cylinder(r_bore, h + 2)
    # the frame's walls between the flanges are thinner than the flanges, as moulded
    web = Box(size + 2, size + 2, h - 2 * fl) - Cylinder(r_bore + 1.2, h)
    web = web - Compound([Pos(sx * hole_pitch / 2, sy * hole_pitch / 2, 0) * Cylinder(size * 0.11, h)
                          for sx in (-1, 1) for sy in (-1, 1)])
    frame = frame - web
    frame = frame - Compound([Pos(sx * hole_pitch / 2, sy * hole_pitch / 2, 0) * Cylinder(1.7, h + 2)
                              for sx in (-1, 1) for sy in (-1, 1)])
    frame = chamfer(frame.edges().filter_by(lambda e: e.geom_type.name == 'CIRCLE' and abs(e.radius - r_bore) < 0.01), 0.6)
    out.append((frame, 'fan_plastic'))
    # four struts and the motor cup on the outlet side (+z)
    zs = h / 2 - fl / 2
    for k in range(4):
        a = 45 + 90 * k
        strut = Rot(0, 0, a) * Pos((r_hub + r_bore) / 2, 0, zs) * Box(r_bore - r_hub + 1.0, 1.6, fl)
        out.append((strut, 'fan_plastic'))
    cup = Pos(0, 0, zs) * Cylinder(r_hub * 0.82, fl)
    out.append((cup, 'fan_plastic'))
    # the impeller: hub with a chamfered intake face, the label, and blades twisted from root to tip
    hub = Pos(0, 0, -fl / 2 + 0.3) * Cylinder(r_hub, h - fl - 1.2)
    hub = chamfer(hub.edges().group_by(Axis.Z)[0], 0.8)
    out.append((hub, 'fan_plastic'))
    z_in = -fl / 2 + 0.3 - (h - fl - 1.2) / 2
    out.append((Pos(0, 0, z_in - 0.01) * Cylinder(r_hub * 0.72, 0.02), 'fan_label'))
    span = h - fl - 2.4
    for k in range(blades):
        out.append((Rot(0, 0, k * 360 / blades) * _blade(r_hub - 0.4, r_bore - 0.6, span, -fl / 2 + 0.3), 'fan_plastic'))
    if screws:
        from mech import fasteners as fs
        for sx in (-1, 1):
            for sy in (-1, 1):
                sc = fs.pan_phillips(3.5, 4.0, head_d=6.4)
                out.append((Pos(sx * hole_pitch / 2, sy * hole_pitch / 2, -h / 2) * Rot(180, 0, 0) * sc, 'steel_zinc'))
    if leads is not None:
        from mech import cable
        a = math.radians(45)
        x0, y0 = math.cos(a) * r_bore, math.sin(a) * r_bore
        for off, mat in ((-0.45, 'wire_red'), (0.45, 'wire_black')):
            ox, oy = -math.sin(a) * off, math.cos(a) * off
            p0 = (math.cos(a) * r_hub + ox, math.sin(a) * r_hub + oy, zs + fl / 2 + 0.3)
            p1 = (x0 + ox, y0 + oy, zs + fl / 2 + 0.3)
            p2 = (size / 2 + 2 + ox, size / 2 + 2 + oy, zs + 2)
            pts = [p0, p1, p2, (leads[0] + ox, leads[1] + oy, leads[2])]
            out.append((cable.tube(pts, 0.45, seg=8), mat))
    return out


def _blade(r0, r1, span, z0, sweep=22.0):
    """One twisted fan blade between radii r0 and r1, its chord falling from the root's steep pitch to the tip's
    flatter one and swept back toward the tip."""
    secs = []
    n = 4
    for i in range(n):
        t = i / (n - 1)
        r = r0 + (r1 - r0) * t
        pitch = 58 - 24 * t                      # degrees from the fan's plane
        chord = span / math.sin(math.radians(pitch)) * (0.9 + 0.25 * t)
        chord = min(chord, 2 * math.pi * r / 7 * 1.15)
        plane = Plane(origin=(0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        sec = Rot(0, 0, sweep * t) * Pos(r, 0, z0) * Rot(0, 90, 0) * Rot(0, 0, 90 - pitch) * \
            RectangleRounded(chord, 0.7 if t < 0.5 else 0.55, 0.25)
        secs.append(sec)
    return loft(secs)


def duct(w0, h0, w1, h1, length):
    """A moulded duct adapter from a w0 x h0 opening at z=0 to a w1 x h1 one at z=length, walls 1.2 mm."""
    outer = loft([RectangleRounded(w0 + 2.4, h0 + 2.4, 1.5), Plane.XY.offset(length) * RectangleRounded(w1 + 2.4, h1 + 2.4, 1.5)])
    inner = loft([Plane.XY.offset(-0.01) * RectangleRounded(w0, h0, 0.4), Plane.XY.offset(length + 0.01) * RectangleRounded(w1, h1, 0.4)])
    return [(outer - inner, 'fan_plastic')]


# ---------------------------------------------------------------- packages and boards

def spheres(centres, r, seg=8, rings=5):
    """Many small balls as one ready mesh, a fixed few dozen triangles each (a CAD sphere costs hundreds)."""
    th = np.linspace(0, np.pi, rings + 1)
    ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    u = np.array([(math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)) for t in th for p in ph])
    tri = []
    for i in range(rings):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            c, d = a + seg, b + seg
            tri += [(a, c, b), (b, c, d)]
    tri = np.array(tri)
    P, N, T = [], [], []
    for k, c in enumerate(centres):
        P.append(np.asarray(c) + r * u)
        N.append(u)
        T.append(tri + k * len(u))
    return ('mesh', np.concatenate(P), np.concatenate(N), np.concatenate(T))


def fcbga(hx, hy, hz, lid=0.72, balls=True, flip=False):
    """A flip chip BGA in the heat source's box (half sizes, centred): the nickel plated lid against the heat sink
    at +z (or -z when flip), the organic substrate with its land side capacitors, and the ball grid below."""
    s = -1 if flip else 1
    t_lid, t_sub = hz * 2 * 0.42, hz * 2 * 0.4
    lw = 2 * hx * lid
    lid_s = Pos(0, 0, s * (hz - t_lid / 2)) * Box(lw, 2 * hy * lid, t_lid)
    lid_s = fillet(lid_s.edges().filter_by(Axis.Z), 1.2)
    lid_s = chamfer(lid_s.edges().group_by(Axis.Z)[-1 if s > 0 else 0], 0.25)
    sub = Pos(0, 0, s * (hz - t_lid - t_sub / 2)) * Box(2 * hx, 2 * hy, t_sub)
    out = [(lid_s, 'ihs_nickel'), (sub, 'substrate')]
    zc = s * (hz - t_lid + 0.25)
    for k in range(10):
        a = k / 10
        for (x, y) in ((-hx + 1.4, -hy * 0.7 + 1.4 * hy * a), (hx - 1.4, -hy * 0.7 + 1.4 * hy * a)):
            out.append((Pos(x, y, zc) * Box(0.5, 1.0, 0.5), 'mlcc'))
    if balls:
        zb = s * (hz - t_lid - t_sub) - s * 0.2
        n = 10
        pts = [(-hx + 1.8 + (2 * hx - 3.6) * i / (n - 1), -hy + 1.8 + (2 * hy - 3.6) * j / (n - 1), zb)
               for i in range(n) for j in range(n)]
        out.append((spheres(pts, 0.45), 'solder_ball'))
    return out


def board_patch(x0, y0, x1, y1, z_top, t=1.6, parts=()):
    """A patch of a multilayer board, top at z_top, with surface parts [(x, y, hx, hy, hz, material)]."""
    core = Pos((x0 + x1) / 2, (y0 + y1) / 2, z_top - t / 2) * Box(x1 - x0, y1 - y0, t)
    core = fillet(core.edges().filter_by(Axis.Z), 2.0)
    out = [(core, 'solder_mask')]
    for x, y, hx, hy, hz, m in parts:
        out.append((Pos(x, y, z_top + hz) * Box(2 * hx, 2 * hy, 2 * hz), m))
    return out


# ---------------------------------------------------------------- coolant plumbing

def barb_fitting(port_r, hex_af=17.0, barb_r=5.2, length=26.0):
    """A G1/4 male to hose barb fitting along -x from a port face at x=0: the thread in the port, an O ring, the
    hex, and a barb with three ridges."""
    out = []
    thread = Pos(-0.01, 0, 0) * Rot(0, 90, 0) * Cylinder(port_r - 0.1, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    out.append((thread, 'brass'))
    hexb = Rot(0, -90, 0) * extrude(_hexagon(hex_af), 6.0)
    hexb = chamfer(hexb.edges().group_by(Axis.X)[0], 0.8)
    out.append((hexb, 'brass'))
    ring = Pos(-0.3, 0, 0) * Rot(0, 90, 0) * (Cylinder(port_r + 2.0, 0.6) - Cylinder(port_r, 1))
    out.append((ring, 'rubber'))
    stem = Pos(-6, 0, 0) * Rot(0, -90, 0) * Cylinder(barb_r * 0.82, length - 6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    out.append((stem, 'brass'))
    for k in range(3):
        x = -9 - k * 5.0
        cone = Pos(x, 0, 0) * Rot(0, -90, 0) * Cone(barb_r, barb_r * 0.82, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        out.append((cone, 'brass'))
    bore = Pos(-length - 0.02, 0, 0) * Rot(0, 90, 0) * Cylinder(barb_r * 0.5, 0.04)
    out.append((bore, 'decal'))
    return out


def _hexagon(af):
    from build123d import RegularPolygon
    return RegularPolygon(af / math.sqrt(3), 6)


def hose_clamp(r, width=8.0):
    """A worm drive hose clamp round a hose of radius r along x at x=0: the perforated band and its screw housing."""
    band = Rot(0, 90, 0) * (Cylinder(r + 0.6, width) - Cylinder(r, width + 1))
    housing = Pos(0, 0, r + 1.6) * Box(width + 1.2, 6.0, 3.0)
    housing = fillet(housing.edges().filter_by(Axis.X), 0.8)
    screw = Pos(0, 0, r + 1.6) * Rot(90, 0, 0) * Cylinder(1.6, 8.0)
    head = Pos(0, -4.6, r + 1.6) * Rot(90, 0, 0) * extrude(_hexagon(5.5), 1.5, both=True)
    slots = [Rot(k * 12, 0, 0) * Pos(0, 0, r + 0.3) * Box(width * 0.55, 1.0, 1.0) for k in range(-12, 13) if abs(k) > 2]
    band = band - Compound(slots)
    return [(band, 'steel_stainless'), (housing, 'steel_stainless'), (screw, 'steel_zinc'), (head, 'steel_zinc')]


def hose(points, r, wall=1.5):
    """A rubber coolant hose along a path, radius r outside."""
    from mech import cable
    return [(cable.tube(points, r, seg=18, n=60), 'hose')]


def igbt_module(base_w=106.0, base_d=61.4, x0=0.0, y0=0.0, z_face=0.0, down=True):
    """A 62 mm half bridge IGBT module bolted under a cold plate, its copper base against the face at z_face and its
    housing hanging below (down): the nickel plated base with four mounting slots, the grey housing with three
    power terminals and their M5 screws, and the gate and emitter pins."""
    s = -1 if down else 1
    out = []
    base = Pos(x0, y0, z_face + s * 1.5) * Box(base_w, base_d, 3.0)
    base = fillet(base.edges().filter_by(Axis.Z), 3.0)
    base = base - Compound([Pos(x0 + sx * (base_w / 2 - 4.5), y0 + sy * (base_d / 2 - 7), z_face) * Cylinder(3.2, 10)
                            for sx in (-1, 1) for sy in (-1, 1)])
    out.append((base, 'copper_nickel'))
    hw, hd, hh = base_w - 14, base_d - 1.4, 27.0
    housing = Pos(x0, y0, z_face + s * (3.0 + hh / 2)) * Box(hw, hd, hh)
    housing = fillet(housing.edges().filter_by(Axis.Z), 3.0)
    housing = chamfer(housing.edges().group_by(Axis.Z)[0 if down else -1], 1.5)
    face = z_face + s * (3.0 + hh)
    tz = []
    for k, dx in enumerate((-hw * 0.33, 0.0, hw * 0.33)):
        boss = Pos(x0 + dx, y0 + hd * 0.18, face) * Box(17.0, 17.0, 6.0)
        housing = housing - boss
        tz.append(dx)
    out.append((housing, 'module_grey'))
    from mech import fasteners as fs
    for dx in tz:
        tab = Pos(x0 + dx, y0 + hd * 0.18, face - s * 3.0 + s * 0.4) * Box(15.0, 15.0, 0.8)
        out.append((tab, 'copper_nickel'))
        sh = fs.hex_bolt('M5', 10, threads=False, marks=False)
        out.append((Pos(x0 + dx, y0 + hd * 0.18, face - s * 3.0 + s * 0.8) * (Rot(180, 0, 0) if down else Rot(0, 0, 0)) * sh, 'steel_zinc'))
    for k in range(4):
        pin = Pos(x0 - hw * 0.4 + k * 3.0, y0 - hd * 0.32, face + s * 4.0) * Box(0.8, 2.8, 8.0)
        out.append((pin, 'tin'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            sh = fs.socket_cap('M6', 6, threads=False)
            out.append((Pos(x0 + sx * (base_w / 2 - 4.5), y0 + sy * (base_d / 2 - 7), z_face + s * 3.0) *
                        (Rot(180, 0, 0) if down else Rot(0, 0, 0)) * sh, 'steel_zinc'))
    return out
