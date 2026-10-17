"""Involute spur gears: true involute flanks generated from the module, tooth count and pressure angle, with root
fillets approximated by the root circle, a bore and optional lightening holes.

A gear stands on z = 0, its axis on z, and rises to its face width. Two gears mesh when their centres are
module * (z1 + z2) / 2 apart; rotate one by half a tooth pitch (180 / z degrees) so the teeth interleave.

    from mech import gears
    g = gears.spur(module=1.0, teeth=20, width=4, bore=3)
"""
import math

from build123d import Polyline, Pos, Solid, extrude, make_face


def _involute(rb, r):
    """The involute's roll angle at radius r from a base circle rb."""
    return math.sqrt(max(r * r / (rb * rb) - 1, 0.0))


def outline(module, teeth, pressure=20.0, steps=6, backlash=0.05):
    """The gear's outline in the xy plane as a list of points, one tooth after another."""
    m, z = module, teeth
    rp = m * z / 2
    rb = rp * math.cos(math.radians(pressure))
    ra = rp + m
    rf = max(rp - 1.25 * m, 0.2)
    # half the tooth's angular thickness at the pitch circle, less half the backlash
    inv = lambda a: math.tan(a) - a
    alpha = math.radians(pressure)
    half = (math.pi / (2 * z)) + inv(alpha) - backlash / (2 * rp)
    pts = []
    r0 = max(rb, rf)
    rs = [r0 + (ra - r0) * (i / steps) ** 1.2 for i in range(steps + 1)]
    for k in range(z):
        c = 2 * math.pi * k / z
        flank = []
        for r in rs:
            t = _involute(rb, r)
            phi = t - math.atan(t)          # the polar angle of the involute at radius r
            flank.append((r, half - phi))
        left = [(r, c - a) for r, a in flank]
        right = [(r, c + a) for r, a in reversed(flank)]
        if rf < r0:
            left.insert(0, (rf, c - flank[0][1]))
            right.append((rf, c + flank[0][1]))
        tooth = left + right
        # the root arc to the next tooth
        a1 = c + flank[0][1]
        a2 = c + 2 * math.pi / z - flank[0][1]
        root = [(rf, a1 + (a2 - a1) * j / 3) for j in range(1, 3)]
        for r, a in tooth + root:
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def spur(module, teeth, width, bore=0.0, pressure=20.0, hub=None, holes=0, hole_r=None, chamfer=0.0):
    """A spur gear: the involute outline extruded to width, a bore, a hub (diameter, height above the face) and
    holes lightening holes on the web."""
    pts = outline(module, teeth, pressure)
    g = extrude(make_face(Polyline(*pts, close=True)), width)
    rp = module * teeth / 2
    if hub:
        hd, hh = hub
        g = g + Pos(0, 0, width) * Solid.make_cylinder(hd / 2, hh)
    if holes:
        rr = hole_r or (rp - 1.25 * module - (hub[0] / 2 if hub else bore)) * 0.32
        rc = ((hub[0] / 2 if hub else bore / 2) + rp - 1.25 * module) / 2
        for i in range(holes):
            a = 2 * math.pi * i / holes
            g = g - Pos(rc * math.cos(a), rc * math.sin(a), -1) * Solid.make_cylinder(rr, width + 2)
    if bore:
        g = g - Pos(0, 0, -1) * Solid.make_cylinder(bore / 2, width + (hub[1] if hub else 0) + 2)
    return g


def pitch_radius(module, teeth):
    return module * teeth / 2
