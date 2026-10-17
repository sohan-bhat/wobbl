"""Hobby servos at their datasheet dimensions, with the horns that bolt to them.

TowerPro MG996R (datasheet drawing): case 40.3 x 20 mm, 36.6 mm from the bottom to the top of the case, mounting
tabs 53.6 mm across with their top face 26.6 mm above the bottom, 47.6 mm to the top of a fitted horn and 42.9 mm to
the top of the 25 tooth brass spline. Tab holes 4.6 mm on 49.5 x 10 mm.
TowerPro MG90S: case 22.8 x 12.2 mm, 22.7 mm to the top of the case and 28.5 mm to the top of the spline, tabs
32.2 mm across with their top 15.9 mm above the bottom, holes 2 mm on 27.6 mm.

Frame: the output axis is +z through the origin, the case's top face is z = 0 and the case runs along -x from the
axis (the spline sits near one end, as on the real part), centred on y. Each function returns a dict of
[(shape, material)] lists keyed 'case', 'output' (spline and collar, fixed to the case) and 'horn' (which turns with
the output, so it belongs to whatever the horn is screwed to). The case is in the material case_material, so a page
can tint it, for instance with how hard the servo is working.
"""
import math

from build123d import (Axis, Box, Cylinder, Plane, Polyline, Pos, Rot, Solid, Spline, Wire, Circle, Rectangle, Sketch,
                       chamfer, extrude, fillet, make_face, revolve, sweep, Locations)

from . import cable as cables
from . import fasteners as fs
import looks

looks.add('servo_case', color=(0.03, 0.03, 0.032), metal=0.0, rough=0.42)
looks.add('servo_label', color=(0.07, 0.06, 0.2), metal=0.0, rough=0.35)
looks.add('wire_brown', color=(0.12, 0.05, 0.02), metal=0.0, rough=0.45)
looks.add('wire_red', color=(0.55, 0.03, 0.02), metal=0.0, rough=0.45)
looks.add('wire_orange', color=(0.85, 0.3, 0.02), metal=0.0, rough=0.45)

# name: case length, width, height, spline offset from the front end, tab span, tab top above bottom, tab thickness,
# hole diameter, hole spacing along x, across y (0 for one hole a tab), spline outer diameter, teeth, spline top
# above the case top, collar diameter
SPECS = {
    'MG996R': dict(L=40.3, W=20.0, H=36.6, off=10.0, span=53.6, tab_top=26.6, tab_t=2.5, hole=4.6, hx=49.5, hy=10.0,
                   spline_d=5.9, teeth=25, spline_top=6.3, collar=13.0, label=True),
    'MG90S': dict(L=22.8, W=12.2, H=22.7, off=5.9, span=32.2, tab_top=15.9, tab_t=2.5, hole=2.0, hx=27.6, hy=0.0,
                  spline_d=4.8, teeth=20, spline_top=5.8, collar=11.8, label=False),
}


def _box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def spline_shaft(d, teeth, z0, z1):
    """A splined output shaft: a star of teeth extruded from z0 to z1, the top chamfered, an M3 or M2 tapped centre."""
    ro, ri = d / 2, d / 2 * 0.9
    pts = []
    for i in range(teeth * 2):
        a = math.pi * i / teeth
        r = ro if i % 2 == 0 else ri
        pts.append((r * math.cos(a), r * math.sin(a)))
    s = extrude(Plane.XY.offset(z0) * make_face(Polyline(*pts, close=True)), z1 - z0)
    s = s - Solid.make_cone(ro * 1.3, ri * 0.75, ro * 1.3 - ri * 0.75, Plane.XY.offset(z1 - (ro - ri) * 0.9))
    s = s - Solid.make_cylinder(d * 0.26, (z1 - z0) * 0.8, Plane.XY.offset(z1 - (z1 - z0) * 0.8))
    return s


def _case(sp, case_material, cable=True):
    L, W, H, off = sp['L'], sp['W'], sp['H'], sp['off']
    x0, x1 = off - L, off
    out = []
    small = L < 30
    r = 0.8 if small else 1.2
    body = _box(x0, x1, -W / 2, W / 2, -H, 0)
    body = fillet(body.edges().filter_by(Axis.Z), r)
    body = chamfer(body.edges().group_by(Axis.Z)[-1], 0.5 if small else 0.7)
    body = chamfer(body.edges().group_by(Axis.Z)[0], 0.4 if small else 0.6)
    # the seams where the bottom cover and the top cover meet the middle of the case
    for zs in ((-H + 4.6, -2.6) if not small else (-H + 3.4, -1.8)):
        ring = _box(x0 - 1, x1 + 1, -W / 2 - 1, W / 2 + 1, zs - 0.18, zs + 0.18) - _box(x0 + 0.25, x1 - 0.25, -W / 2 + 0.25, W / 2 - 0.25, zs - 1, zs + 1)
        body = body - ring
    # mounting tabs with their holes, the slot to the end of each tab and the ribs under them
    tz1 = -H + sp['tab_top']
    tz0 = tz1 - sp['tab_t']
    ext = (sp['span'] - L) / 2
    tw = W * (0.9 if not small else 0.95)
    for s in (-1, 1):
        xa, xb = (x1, x1 + ext) if s > 0 else (x0 - ext, x0)
        tab = _box(xa - s * 1.0, xb, -tw / 2, tw / 2, tz0, tz1)
        tab = fillet(tab.edges().filter_by(Axis.Z).group_by(Axis.X)[-1 if s > 0 else 0], min(1.2, tw * 0.2))
        xc = (x0 + x1) / 2 + s * sp['hx'] / 2
        ys = (-sp['hy'] / 2, sp['hy'] / 2) if sp['hy'] else (0.0,)
        for y in ys:
            tab = tab - Pos(xc, y, tz0 - 1) * Solid.make_cylinder(sp['hole'] / 2, sp['tab_t'] + 2)
        slot_w = sp['hole'] * (0.3 if sp['hy'] else 0.55)
        tab = tab - _box(min(xc, xc + s * ext), max(xc, xc + s * ext), -slot_w / 2, slot_w / 2, tz0 - 1, tz1 + 1)
        body = body + tab
        for y in ((-tw * 0.32, tw * 0.32) if not small else (0.0,)):
            rib = Pos(0, y, 0) * extrude(Plane.XZ.offset(0.6) * make_face(Polyline(
                (xa if s > 0 else xb, tz0), (xa + s * ext * 0.55 if s > 0 else xb - ext * 0.55, tz0),
                (xa if s > 0 else xb, tz0 - ext * 0.9), close=True)), 1.2)
            body = body + rib
    out.append((body, case_material))
    # the brand sticker on the side the cable leaves from
    if sp['label']:
        lab = _box(x0 + 3.5, x1 - 3.5, W / 2 - 0.02, W / 2 + 0.12, -H + 7.5, tz0 - 2.5)
        out.append((lab, 'servo_label'))
        lab2 = _box(x0 + 3.5, x1 - 3.5, -W / 2 - 0.12, -W / 2 + 0.02, -H + 7.5, tz0 - 2.5)
        out.append((lab2, 'servo_label'))
    # the cable's strain relief and the three wires leaving the back end of the case
    if cable:
        cz = -H + (5.5 if not small else 3.8)
        boss = _box(x0 - 2.2, x0 + 0.5, -3.2 if not small else -2.4, 3.2 if not small else 2.4, cz - 1.6, cz + 1.6)
        out.append((fillet(boss.edges().filter_by(Axis.X), 0.6), case_material))
        wd = 0.6 if not small else 0.45
        path = [(x0 - 1.5, 0, cz), (x0 - 6.0, 0, cz - 0.8), (x0 - 9.0, 0, cz - 5.0), (x0 - 9.6, 0, cz - 12.0)]
        out += cables.ribbon(path, wd, ['wire_brown', 'wire_red', 'wire_orange'], side=(0, 1, 0))
        # the lead ends in its three way receptacle
        px, _, pz = path[-1]
        cw = 3 * 2 * wd + 1.2
        plug = Pos(px, 0, pz - 6.5) * Box(2.6 + wd, cw, 13.0)
        plug = chamfer(plug.edges().filter_by(Axis.Z), 0.3)
        out.append((plug, 'plastic_black'))
    return out


def _output(sp):
    d = sp['spline_d']
    top = sp['spline_top']
    c = sp['collar']
    small = d < 5.5
    collar = Solid.make_cylinder(c / 2, 1.4 if not small else 3.2)
    collar = chamfer(collar.edges().group_by(Axis.Z)[-1], 0.4)
    z = 1.4 if not small else 3.2
    bush = Pos(0, 0, z) * Solid.make_cylinder(d * 0.75, 0.8)
    return [(collar, 'servo_case'), (bush, 'brass'), (spline_shaft(d, sp['teeth'], z + 0.8 - 0.01, top), 'brass')]


def disc_horn(d=25.0, hub_d=10.0, t=2.5, hub_h=6.5, holes=((7.0, 4, 45), (10.0, 4, 0)), hole_d=3.0, screw='M3', z0=None):
    """A machined aluminium round servo horn: the disc with its bolt circles, the clamped hub and the centre screw.
    It sits on the spline with the disc on top, so the face it bolts to is its top face. Returns [(shape, material)]
    with the horn's top face at z = 0."""
    disc = Pos(0, 0, -t) * Solid.make_cylinder(d / 2, t)
    disc = chamfer(disc.edges().group_by(Axis.Z)[-1], 0.35)
    disc = chamfer(disc.edges().group_by(Axis.Z)[0], 0.25)
    hub = Pos(0, 0, -hub_h) * Solid.make_cylinder(hub_d / 2, hub_h - t + 0.01)
    hub = chamfer(hub.edges().group_by(Axis.Z)[0], 0.4)
    horn = disc + hub
    for r, n, a0 in holes:
        for i in range(n):
            a = math.radians(a0 + 360 * i / n)
            horn = horn - Pos(r * math.cos(a), r * math.sin(a), -t - 1) * Solid.make_cylinder(hole_d / 2, t + 2)
    # the clamp slot across the hub and the bore with its counterbore for the centre screw
    horn = horn - Pos(0, hub_d * 0.25, -hub_h + (hub_h - t) / 2) * Box(0.6, hub_d * 0.5 + 0.1, hub_h - t + 0.02)
    horn = horn - Pos(0, 0, -t - 1) * Solid.make_cylinder(3.2 if screw == 'M3' else 2.2, t + 1.02)
    out = [(horn, 'aluminum_machined')]
    sc = fs.socket_cap(screw, 2, threads=False, lite=True)
    out.append((Pos(0, 0, -t + 0.4) * sc, 'steel_black_oxide'))
    cl = fs.socket_cap('M2', hub_d * 0.9, threads=False, lite=True)
    out.append((Pos(hub_d * 0.5, hub_d * 0.25, -hub_h + (hub_h - t) / 2) * Rot(0, 90, 0) * cl, 'steel_black_oxide'))
    return out


def arm_horn(length=16.0, d_hub=7.0, t=1.6, hub_h=4.0, holes=5):
    """A moulded plastic single arm horn with its row of holes and the hub, top face at z = 0."""
    tip = length
    prof = Sketch() + Circle(d_hub / 2)
    arm = extrude(Plane.XY.offset(-t) * make_face(Polyline((0, d_hub / 2 * 0.95), (tip, 1.6), (tip, -1.6), (0, -d_hub / 2 * 0.95), close=True)), t)
    arm = arm + Pos(tip, 0, -t) * Solid.make_cylinder(1.6, t)
    hub = Pos(0, 0, -hub_h) * Solid.make_cylinder(d_hub / 2, hub_h)
    h = arm + hub
    for i in range(holes):
        x = d_hub / 2 + 1.5 + i * (tip - d_hub / 2 - 1.5) / max(1, holes - 1)
        h = h - Pos(x, 0, -t - 1) * Solid.make_cylinder(0.5, t + 2)
    return [(h, 'plastic_white')]


def servo(model='MG996R', case_material='servo_case', horn='disc', cable=True, horn_top=None):
    """A servo by name: {'case': ..., 'output': ..., 'horn': ...}. The horn's top face is at the datasheet's height
    with a horn fitted (47.6 mm above the bottom for an MG996R), the spline fully engaged in its hub."""
    sp = SPECS[model]
    out = {'case': _case(sp, case_material, cable), 'output': _output(sp)}
    out['output'][0] = (out['output'][0][0], case_material)
    if horn == 'disc':
        if model == 'MG996R':
            top = horn_top or 47.6 - sp['H']
            parts = disc_horn(hub_h=min(6.5, top - 2.9))
        else:
            top = horn_top or sp['spline_top'] + 1.6
            parts = disc_horn(d=16.0, hub_d=7.0, t=2.0, hub_h=3.6, holes=((5.0, 4, 45),), hole_d=2.0, screw='M2')
        out['horn'] = [(Pos(0, 0, top) * s, m) for s, m in parts]
    elif horn == 'arm':
        out['horn'] = [(Pos(0, 0, sp['spline_top'] + 1.6) * s, m) for s, m in arm_horn()]
    else:
        out['horn'] = []
    return out


def horn_top(model='MG996R'):
    """The height of the fitted horn's top face above the case top, the face a bracket bolts to."""
    sp = SPECS[model]
    return 47.6 - sp['H'] if model == 'MG996R' else sp['spline_top'] + 1.6


def tab_plane(model='MG996R'):
    """(z of the tab underside, z of the tab top, x of each hole column, y of the holes) for screwing the case down."""
    sp = SPECS[model]
    tz1 = -sp['H'] + sp['tab_top']
    xs = [(sp['off'] - sp['L'] / 2) + s * sp['hx'] / 2 for s in (-1, 1)]
    ys = [-sp['hy'] / 2, sp['hy'] / 2] if sp['hy'] else [0.0]
    return tz1 - sp['tab_t'], tz1, xs, ys
