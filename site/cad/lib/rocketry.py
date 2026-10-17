"""High power rocket hardware, from vendor drawings (1010 rail buttons, a screw on motor retainer, a 75 mm motor's
aft closure and graphite nozzle): the wound airframe with its coupler joint, avionics switch band, vents, shear
pins and roll pattern, the ogive nose with its aluminium tip, fins with rounded edges and epoxy root fillets, rail
buttons, a launch lug, the open aft end, the launch pad with its blast deflector and tilting rail mount, and the
1010 T slot rail. Lengths in millimetres, the rocket along +z, nose up.
"""
import math

import numpy as np
from build123d import (Align, Axis, Box, Compound, Cylinder, Plane, Polygon, Polyline, Pos, Rot, Sphere, chamfer,
                       extrude, fillet, make_face, revolve)

import looks

looks.add('nylon_black', color=(0.02, 0.02, 0.022), metal=0.0, rough=0.35)
looks.add('anodized_red', color=(0.42, 0.02, 0.02), metal=1.0, rough=0.3)
looks.add('switch_band', color=(0.02, 0.02, 0.022), metal=0.0, rough=0.3, normal='peel_n', nstr=0.1, rvar=(0.2, 0.32), tiles=30)


def lathe(pts):
    """A solid of revolution about z from a closed (r, z) profile."""
    return revolve(make_face(Polyline(*[(r, 0, z) for r, z in pts], close=True)), Axis.Z)


def band(r0, r1, z0, z1, a0=0.0, a1=360.0):
    """A ring sector between radii r0 and r1 from z0 to z1, from angle a0 to a1 in degrees: a paint panel."""
    ring = Pos(0, 0, z0) * (Cylinder(r1, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN)) -
                            Cylinder(r0, 3 * (z1 - z0) + 2))
    if a1 - a0 >= 360:
        return ring
    big = r1 * 3
    wedge = Polygon((0, 0), (big * math.cos(math.radians(a0)), big * math.sin(math.radians(a0))),
                    (big * math.cos(math.radians((a0 + a1) / 2)) * 1.5, big * math.sin(math.radians((a0 + a1) / 2)) * 1.5),
                    (big * math.cos(math.radians(a1)), big * math.sin(math.radians(a1))), align=None)
    return ring & Pos(0, 0, z0 - 1) * extrude(wedge, z1 - z0 + 2)


def airframe(R, z_aft, z_top, wall=1.8, seam=-450.0, bay=(-330.0, -270.0)):
    """The airframe tube: paint white with a black roll pattern and stripes, a coupler joint, the avionics switch
    band with its vent holes, key switch and shear pins, the motor tube and centring ring at the open aft end."""
    out = []
    tube = lathe([(R - wall, z_aft), (R - 0.4, z_aft), (R, z_aft + 0.4), (R, z_top - 0.4), (R - 0.4, z_top), (R - wall, z_top)])
    joint = Pos(0, 0, seam) * (Cylinder(R + 1, 1.4) - Cylinder(R - 0.45, 4))
    tube = tube - joint
    out.append((tube, 'paint_white'))
    lift = R + 0.3
    out.append((band(R - 0.5, lift + 0.05, bay[0], bay[1]), 'switch_band'))
    for k in range(3):
        a = math.radians(30 + 120 * k)
        out.append((Pos(math.cos(a) * lift, math.sin(a) * lift, (bay[0] + bay[1]) / 2) * Rot(0, 90, math.degrees(a)) *
                    Cylinder(1.6, 1.0), 'decal'))
    key = math.radians(150)
    out.append((Pos(math.cos(key) * (lift + 0.3), math.sin(key) * (lift + 0.3), bay[0] + 12) * Rot(0, 90, 150) *
                Cylinder(4.5, 1.6), 'steel_stainless'))
    for k in range(3):
        a = math.radians(90 + 120 * k)
        for z in (bay[1] + 18, seam + 14):
            out.append((Pos(math.cos(a) * lift, math.sin(a) * lift, z) * Rot(0, 90, math.degrees(a)) * Cylinder(1.1, 0.6), 'nylon'))
    for z0, z1 in ((z_top - 24, z_top - 20), (-1130, -1110), (-700, -660)):
        out.append((band(R - 0.5, lift, z0, z1), 'decal'))
    for k in range(0, 4, 2):
        out.append((band(R - 0.5, lift, -230, -130, k * 90, (k + 1) * 90), 'decal'))
    mmt = Pos(0, 0, z_aft + 0.4) * (Cylinder(39.5, 140, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(38.0, 400))
    ring = Pos(0, 0, z_aft + 1.6) * (Cylinder(R - wall, 6, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(39.5, 20))
    out += [(mmt, 'plastic_black'), (ring, 'plastic_black')]
    return out


def rail_button(R, z, ang):
    """A 1010 rail button: an acetal spool on a pan head screw standing off the tube at angle ang (degrees)."""
    spool = lathe([(0, 0), (8.2, 0), (8.2, 1.6), (5.6, 3.0), (5.6, 7.4), (8.2, 8.2), (8.2, 9.8), (0, 9.8)])
    spool = fillet(spool.edges().filter_by(lambda e: e.geom_type.name == 'CIRCLE' and e.radius > 8.0), 0.5)
    head = lathe([(0, 9.8), (3.4, 9.8), (3.0, 11.0), (0, 11.3)])
    head = head - Pos(0, 0, 11.3) * Box(5.0, 0.8, 1.2) - Pos(0, 0, 11.3) * Box(0.8, 5.0, 1.2)
    a = math.radians(ang)
    M = Pos(math.cos(a) * (R - 0.4), math.sin(a) * (R - 0.4), z) * Rot(0, 0, ang) * Rot(0, 90, 0)
    return [(M * spool, 'nylon_black'), (M * head, 'steel_zinc')]


def launch_lug(R, z0, z1, ang, r=6.6):
    """A launch lug: a phenolic tube glued along the body at angle ang, its ends cut square, with epoxy fillets."""
    a = math.radians(ang)
    c = (math.cos(a) * (R + r - 0.3), math.sin(a) * (R + r - 0.3))
    tube = Pos(c[0], c[1], z0) * (Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN)) -
                                  Cylinder(r - 0.9, 3 * (z1 - z0)))
    fillets = []
    for s in (-1, 1):
        t = (-math.sin(a) * s * r * 0.75, math.cos(a) * s * r * 0.75)
        f = Pos(math.cos(a) * (R + 0.6) + t[0], math.sin(a) * (R + 0.6) + t[1], (z0 + z1) / 2) * \
            Cylinder(1.4, z1 - z0 - 4)
        fillets.append((f, 'paint_white'))
    return [(tube, 'paint_white')] + fillets


def ogive_nose(profile, tip_r=3.5):
    """The run's nose profile [(r, z)] from its base up, with an aluminium tip and a seam at the shoulder."""
    pts = sorted(profile, key=lambda p: p[1])
    z_tip = pts[-1][1]
    body = [(0, pts[0][1] + 0.3), (pts[0][0] - 0.4, pts[0][1] + 0.3), (pts[0][0], pts[0][1] + 0.9)]
    body += [(r, z) for r, z in pts[1:] if z < z_tip - 18]
    zc = max(z for r, z in body)
    rc = max(r for r, z in body if z == zc)
    body += [(0, zc)]
    nose = lathe(body)
    tip = lathe([(0, zc - 0.01), (rc, zc - 0.01)] + [(rc * math.cos(math.pi / 2 * k / 8) * (1 - k / 9), zc + (z_tip - zc) * math.sin(math.pi / 2 * k / 8)) for k in range(1, 8)] + [(0, z_tip)])
    return [(nose, 'paint_red'), (tip, 'aluminum_machined')]


def fins(R, root, tip, half, count=3, start=90.0):
    """Fins of thickness 2*half: root (z aft, z fore) at the tube and tip (r, z aft, z fore), rounded on every
    edge, with an epoxy fillet along each side of each root."""
    za0, zf0 = root
    r1, za1, zf1 = tip
    out = []
    for k in range(count):
        ang = start + k * 360 / count
        prof = Polygon((R - 3, za0), (r1, za1), (r1, zf1), (R - 3, zf0), align=None)
        f = Plane.XZ * prof
        f = extrude(f, half, both=True)
        f = fillet(f.edges().filter_by(Axis.Y, reverse=True).filter_by(lambda e: e.center().X > R - 2.5), half * 0.9)
        out.append((Rot(0, 0, ang) * f, 'paint_gloss_black'))
        for s in (-1, 1):
            fl = Pos(R + 0.4, s * (half + 0.5), (za0 + zf0) / 2) * Cylinder(2.2, zf0 - za0 - 10)
            fl = chamfer(fl.edges(), 2.0)
            out.append((Rot(0, 0, ang) * fl, 'paint_gloss_black'))
    return out


def motor_tail(z0, z1):
    """The aft end below the airframe: a knurled screw on retainer, the motor's aft closure and the graphite
    nozzle with its throat and exit cone, from z0 (the airframe's aft face) down to z1 (the nozzle exit)."""
    out = []
    ret = lathe([(26.2, z0 + 0.5), (31.8, z0 + 0.5), (32.2, z0 - 1.0), (32.2, z0 - 11.8), (31.2, z0 - 12.8), (25.5, z0 - 12.8), (25.5, z0 + 0.5)])
    knurl = [Rot(0, 0, k * 10) * Pos(32.3, 0, z0 - 6.4) * Rot(0, 0, 45) * Box(0.9, 0.9, 9.6) for k in range(36)]
    out.append((ret - Compound(knurl), 'anodized_red'))
    closure = lathe([(0, z0 - 12.8), (24.8, z0 - 12.8), (24.8, z0 - 17.0), (24.2, z0 - 17.6), (24.2, z0 - 18.4), (23.6, z0 - 19.0), (0, z0 - 19.0)])
    out.append((closure, 'aluminum_machined'))
    nz = lathe([(0, z0 - 15.7), (14.7, z0 - 19.6), (11.8, z0 - 21.6), (18.6, z1), (21.4, z1), (21.9, z1 + 1.0), (22.6, z0 - 19.0), (0, z0 - 18.9)])
    nz = nz - lathe([(0, z0 - 30), (6.0, z0 - 21.6), (17.6, z1 - 0.01), (0, z1 - 0.01)])
    out.append((nz, 'graphite'))
    return out


def rail_1010(w, L):
    """A 1010 style T slot extrusion of half width w, length 2L along z, centred on the origin."""
    slot, lip, deep = w * 0.30, w * 0.26, w * 0.62
    pts = []
    for k in range(4):
        rot = k * math.pi / 2
        side = [(w, -w * 0.82), (w, -slot), (w - lip, -slot), (w - lip, -w * 0.52), (w - deep, -w * 0.36), (w - deep, w * 0.36),
                (w - lip, w * 0.52), (w - lip, slot), (w, slot), (w, w * 0.82), (w * 0.82, w)]
        for x, y in side:
            pts.append((x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)))
    prof = Polygon(*pts, align=None)
    rail = Pos(0, 0, -L) * extrude(prof, 2 * L)
    rail = rail - Pos(0, 0, -L) * Cylinder(w * 0.2, 4 * L)
    return [(rail, 'aluminum_machined')]


def pad(hx, hy, hz, rail_foot=None):
    """The launch pad table (the scene's box, centred): a tread plate top on a painted frame with rubber feet, a
    blast deflector plate, and at rail_foot (x, y in the pad's frame, tilt in degrees about y) the rail's hinged
    mount with its pivot bolt and stop block."""
    out = []
    top = Pos(0, 0, hz - 4) * Box(2 * hx, 2 * hy, 8)
    out.append((chamfer(top.edges().group_by(Axis.Z)[-1], 1.5), 'tread_steel'))
    for s in (-1, 1):
        out.append((Pos(0, s * (hy - 50), -4) * Box(2 * hx - 20, 40, 2 * hz - 20), 'paint_black'))
        out.append((Pos(s * (hx - 50), 0, -4) * Box(40, 2 * hy - 20, 2 * hz - 20), 'paint_black'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = lathe([(0, -hz), (35, -hz), (35, -hz + 6), (8, -hz + 10), (8, -hz + 13), (0, -hz + 13)])
            out.append((Pos(sx * (hx - 50), sy * (hy - 50), 0) * foot, 'rubber'))
    if rail_foot is not None:
        x, y, tilt = rail_foot
        defl = Pos(x - 40, y, hz + 1.5) * Box(220, 220, 3)
        out.append((fillet(defl.edges().filter_by(Axis.Z), 10), 'steel_stainless'))
        for s in (-1, 1):
            cheek = Pos(x, y + s * 18, hz + 3 + 30) * Box(50, 6, 60)
            out.append((fillet(cheek.edges().filter_by(Axis.Y), 6) - Pos(x, y + s * 18, hz + 3 + 40) * Cylinder(5.5, 20, rotation=(90, 0, 0)), 'paint_black'))
        bolt = Pos(x, y, hz + 43) * Cylinder(5.0, 52, rotation=(90, 0, 0))
        out.append((bolt, 'steel_zinc'))
        for s in (-1, 1):
            nut = Pos(x, y + s * 23, hz + 43) * Rot(90, 0, 0) * extrude(_hexagon(16), 4, both=True)
            out.append((nut, 'steel_zinc'))
    return out


def _hexagon(af):
    from build123d import RegularPolygon
    return RegularPolygon(af / math.sqrt(3), 6)
