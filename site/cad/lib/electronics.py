"""Electronics parts for the circuit and board scenes, modelled from datasheet drawings: a solderless breadboard with
its clip rows, a bench supply with five way binding posts and knobs, through hole and surface mount packages (DIP,
SOIC, SOT-23, TO-220, DO-41, radial electrolytics with their vents, toroid inductors with their windings), screw
terminals, LED leads and 18650 cell ends. Each builder returns a list of (build123d shape, material name) in the
part's own frame, Z up, in millimetres, ready for gltf.Node.
"""
import math

import numpy as np
from build123d import (Align, Axis, Box, Compound, Cylinder, Plane, Pos, Rectangle, Rot, Sphere, Text, Torus,
                       chamfer, extrude, fillet, loft)

import looks

# colors of parts that carry no live field, from the parts themselves
looks.add('print_grey', color=(0.09, 0.09, 0.09), metal=0.0, rough=0.5)
looks.add('rail_red', color=(0.62, 0.05, 0.04), metal=0.0, rough=0.45)
looks.add('rail_blue', color=(0.03, 0.12, 0.5), metal=0.0, rough=0.45)
looks.add('tin', color=(0.78, 0.78, 0.76), metal=1.0, rough=0.32)
looks.add('epoxy_black', color=(0.02, 0.02, 0.022), metal=0.0, rough=0.55)
looks.add('cap_plastic', color=(0.05, 0.05, 0.05), metal=0.0, rough=0.35)
looks.add('lcd_bezel', color=(0.012, 0.012, 0.013), metal=0.0, rough=0.2)
looks.add('panel_print', color=(0.7, 0.7, 0.68), metal=0.0, rough=0.4)
looks.add('copper_wire', color=(0.93, 0.5, 0.3), metal=1.0, rough=0.25)
looks.add('ferrite', color=(0.06, 0.065, 0.06), metal=0.0, rough=0.7)
looks.add('solder_mask', color=(0.02, 0.17, 0.07), metal=0.0, rough=0.28, rvar=(0.22, 0.34), tiles=40)
looks.add('silkscreen', color=(0.85, 0.85, 0.82), metal=0.0, rough=0.5)
looks.add('gold', color=(0.95, 0.72, 0.38), metal=1.0, rough=0.22)
looks.add('terminal_blue', color=(0.03, 0.18, 0.55), metal=0.0, rough=0.4)
looks.add('cap_sleeve_top', color=(0.62, 0.63, 0.65), metal=1.0, rough=0.3)

FONT = 'Helvetica'


def text(s, h, x, y, z, mat, lift=0.015, rot=0, align=(Align.CENTER, Align.CENTER), plane=None):
    """Printed lettering: the glyphs as flat faces a hair above a face at height z (a print has no walls, and the
    walls would cost five times the triangles). plane, when given, places it on another face instead."""
    t = Rot(0, 0, rot) * Text(s, h, font=FONT, align=align)
    if plane is not None:
        return plane * t, mat
    return Pos(x, y, z + lift) * t, mat


# ---------------------------------------------------------------- breadboard

def breadboard(hx, hy, hz, cols, rows, channel_y, rails, stripes, number_y=(), letters=(), funnel=True):
    """A solderless breadboard the size of the scene's box (half sizes hx, hy, hz, centred on its origin).

    cols and rows are the hole centres in x and y (local), channel_y the DIP ravine, rails the y of the power rail
    rows (their clips run the board's length), stripes [(y, 'red' or 'blue')] the printed rail lines, number_y the
    y of the column numbers and letters [(x, [(y, 'a'), ...])] the row letters."""
    top = hz
    body = Box(2 * hx, 2 * hy, 2 * hz)
    body = fillet(body.edges().filter_by(Axis.Z), 1.0)
    body = chamfer(body.edges().group_by(Axis.Z)[-1], 0.25)
    cuts = [Pos(0, channel_y, top) * Box(2 * hx + 2, 2.4, 2.2, align=(Align.CENTER, Align.CENTER, Align.MAX))]
    # the shallow V where a power rail strip meets the terminal block
    for y in _rail_seams(rows, rails):
        cuts.append(Pos(0, y, top + 0.15) * Rot(45, 0, 0) * Box(2 * hx + 2, 0.4, 0.4))
    # square holes with a pyramid lead in, down to the clip
    if funnel:
        lead_in = loft([Plane.XY.offset(0.02) * Rectangle(1.5, 1.5), Plane.XY.offset(-0.35) * Rectangle(0.86, 0.86)])
        hole = lead_in + Pos(0, 0, -0.35) * Box(0.86, 0.86, 1.25, align=(Align.CENTER, Align.CENTER, Align.MAX))
    else:
        # a long board seen from further off: plain square holes, half the triangles
        hole = Pos(0, 0, 0.02) * Box(1.1, 1.1, 1.62, align=(Align.CENTER, Align.CENTER, Align.MAX))
    cuts += [Pos(x, y, top) * hole for x in cols for y in rows]
    # dovetails: tabs on the far long side, matching slots on the near one, so boards lock together
    xs = [x for x in _spread(-hx + 6, hx - 6, 12.0)]
    for x in xs:
        cuts.append(Pos(x, -hy, -hz + 1.6) * _dovetail(1.0, 3.4))
    body = body - Compound(cuts)
    tabs = [Pos(x, hy, -hz + 1.6) * _dovetail(0.9, 3.0) for x in xs]
    out = [(body, 'plastic_white')] + [(t, 'plastic_white') for t in tabs]
    # clips: a tin plated strip under each run of five holes and under each rail, seen through the holes
    clip_z = top - 1.58
    terminal = [y for y in rows if y not in rails]
    for group in _runs(terminal, 2.6):
        y0, y1 = min(group), max(group)
        for x in cols:
            out.append((Pos(x, (y0 + y1) / 2, clip_z) * Box(1.0, y1 - y0 + 1.2, 0.1), 'tin'))
    for y in rails:
        out.append((Pos((cols[0] + cols[-1]) / 2, y, clip_z) * Box(cols[-1] - cols[0] + 1.2, 1.0, 0.1), 'tin'))
    # printed rail lines with their polarity marks at both ends
    x0, x1 = cols[0] - 1.2, cols[-1] + 1.2
    for y, color in stripes:
        out.append((Pos((x0 + x1) / 2, y, top) * Box(x1 - x0, 0.3, 0.04), f'rail_{color}'))
        mark = '+' if color == 'red' else '–'
        for xm in (x0 - 1.4, x1 + 1.4):
            out.append(text(mark, 1.5, xm, y, top, f'rail_{color}'))
    # column numbers every five and the row letters, moulded and printed grey
    for y in number_y:
        for k, x in enumerate(cols):
            n = k + 1
            if n == 1 or n % 5 == 0:
                out.append(text(str(n), 0.95, x, y, top, 'print_grey'))
    for x, labels in letters:
        for y, ch in labels:
            out.append(text(ch, 1.0, x, y, top, 'print_grey'))
    return out


def _dovetail(depth, width):
    """A dovetail prism standing out from a side wall along +y (or cut into one), wider at its outer end."""
    from build123d import Polygon
    w0, w1 = width * 0.7, width
    sk = Polygon((-w0 / 2, -depth), (w0 / 2, -depth), (w1 / 2, depth), (-w1 / 2, depth), align=None)
    return extrude(sk, 1.6, both=True)


def _spread(a, b, step):
    n = max(1, int((b - a) // step))
    return [a + (b - a) * (i + 0.5) / n for i in range(n)]


def _runs(vals, gap):
    vals = sorted(vals)
    out, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] > gap:
            out.append(cur)
            cur = []
        cur.append(v)
    out.append(cur)
    return out


def _rail_seams(rows, rails):
    """Midway between each pair of power rails and the terminal row next to it."""
    terminal = sorted(y for y in rows if y not in rails)
    seams = []
    for group in _runs(sorted(rails), 2.6):
        near = min(terminal, key=lambda t: min(abs(t - g) for g in group))
        edge = min(group, key=lambda g: abs(g - near))
        seams.append((near + edge) / 2)
    return seams


# ---------------------------------------------------------------- bench supply

def front_plane(x, y, z):
    """A plane on a front face that looks along -y, its x along the panel."""
    return Plane(origin=(x, y, z), x_dir=(1, 0, 0), z_dir=(0, -1, 0))


def knob(r, h, flutes=16):
    """A fluted instrument knob looking along -z (built along +z, front at z=h): a graduated skirt, a fluted
    grip with a chamfered face and a white pointer line."""
    skirt = Cylinder(r * 1.25, h * 0.18, align=(Align.CENTER, Align.CENTER, Align.MIN))
    skirt = chamfer(skirt.edges().group_by(Axis.Z)[-1], h * 0.06)
    grip = Pos(0, 0, h * 0.18) * Cylinder(r, h * 0.82, align=(Align.CENTER, Align.CENTER, Align.MIN))
    grip = chamfer(grip.edges().group_by(Axis.Z)[-1], r * 0.14)
    cuts = [Rot(0, 0, k * 360 / flutes) * Pos(r * 1.02, 0, h * 0.18 + h * 0.41) * Box(r * 0.2, r * 0.2, h * 0.9)
            for k in range(flutes)]
    grip = grip - Compound(cuts)
    face = Pos(0, 0, h) * Cylinder(r * 0.7, 0.02, align=(Align.CENTER, Align.CENTER, Align.MIN))
    pointer = Pos(0, r * 0.42, h + 0.02) * Box(r * 0.09, r * 0.55, 0.02)
    ticks = [Rot(0, 0, a) * Pos(0, r * 1.15, h * 0.18) * Box(r * 0.05, r * 0.16, 0.02)
             for a in range(-135, 136, 27)]
    return [(skirt + grip, 'plastic_black'), (face, 'aluminum_machined'), (pointer, 'panel_print')] + \
        [(t, 'panel_print') for t in ticks]


def binding_post(r, hz):
    """A five way binding post on a panel, standing along +z from z=-hz (the panel) to z=+hz: an insulated body,
    the brass stud with its cross hole showing, and a knurled cap with a banana socket. 'post_plastic' is the
    material the runtime tints with the post's own color."""
    z0 = -hz
    body = Pos(0, 0, z0) * Cylinder(r * 1.12, hz * 0.55, align=(Align.CENTER, Align.CENTER, Align.MIN))
    body = chamfer(body.edges().group_by(Axis.Z)[-1], r * 0.12)
    nut = Pos(0, 0, z0) * Cylinder(r * 1.35, hz * 0.12, align=(Align.CENTER, Align.CENTER, Align.MIN))
    stud_z = z0 + hz * 0.55
    stud = Pos(0, 0, stud_z) * Cylinder(r * 0.42, hz * 0.45, align=(Align.CENTER, Align.CENTER, Align.MIN))
    stud = stud - Pos(0, 0, stud_z + hz * 0.22) * Rot(0, 90, 0) * Cylinder(r * 0.2, r * 2)
    cap_z = stud_z + hz * 0.4
    cap = Pos(0, 0, cap_z) * Cylinder(r, hz - cap_z, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = fillet(cap.edges().group_by(Axis.Z)[-1], r * 0.18)
    knurl = [Rot(0, 0, k * 20) * Pos(r * 1.0, 0, cap_z + (hz - cap_z) * 0.42) * Rot(0, 0, 45) * Box(r * 0.14, r * 0.14, (hz - cap_z) * 0.6)
             for k in range(18)]
    cap = cap - Compound(knurl) - Pos(0, 0, hz) * Cylinder(r * 0.3, r * 1.6)
    socket = Pos(0, 0, hz - r * 0.8) * Cylinder(r * 0.3, r * 0.1)
    return [(body, 'post_plastic'), (nut, 'steel_zinc'), (stud, 'brass'), (cap, 'post_plastic'),
            (socket, 'brass')]


looks.add('post_plastic', color=(0.5, 0.5, 0.5), metal=0.0, rough=0.3)


def supply_case(hx, hy, hz, riser=0.0, cord=None):
    """The supply's sheet steel cover over a black chassis: rounded top edges, side louvres and top vents, a
    rear panel with fan grille, rubber feet, an instrument riser below when the bench is riser lower than the
    case, and the mains cord out of the back toward cord (a point in the case's frame), if given."""
    out = []
    t = 0.5
    shell = Box(2 * hx, 2 * hy, 2 * hz)
    shell = fillet(shell.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1], 2.0)
    inner = Pos(0, 0, -t) * Box(2 * hx - 2 * t, 2 * hy + 2, 2 * hz)
    cover = shell - inner
    cuts = []
    for side in (-1, 1):
        for k in range(7):
            y = hy * 0.15 + k * hy * 0.11
            cuts.append(Pos(side * hx, y, 0) * Box(2 * t + 0.4, 0.55, hz * 1.0))
    for k in range(9):
        x = -hx * 0.6 + k * hx * 0.15
        cuts.append(Pos(x, hy * 0.55, hz) * Box(0.5, hy * 0.55, 2 * t + 0.4))
    cover = cover - Compound(cuts)
    out.append((cover, 'paint'))
    chassis = Pos(0, 0.3, -hz) * Box(2 * hx - 2 * t - 0.05, 2 * hy - 0.6, 2 * hz - t - 0.05,
                                      align=(Align.CENTER, Align.CENTER, Align.MIN))
    grille = [Pos(hx * 0.45 + dx, hy + 0.0, hz * 0.1 + dz) * Cylinder(0.35, 1.0, rotation=(90, 0, 0))
              for dx in (-2.4, -1.2, 0, 1.2, 2.4) for dz in (-2.4, -1.2, 0, 1.2, 2.4) if dx * dx + dz * dz <= 7.3]
    chassis = chassis - Compound(grille)
    out.append((chassis, 'paint_black'))
    iec = Pos(-hx * 0.45, hy, -hz * 0.35) * Box(5.0, 0.8, 3.6)
    out.append((iec, 'plastic_black'))
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = Pos(sx * (hx - 3), sy * (hy - 3), -hz) * Cylinder(2.0, 1.2, align=(Align.CENTER, Align.CENTER, Align.MAX))
            out.append((chamfer(foot.edges().group_by(Axis.Z)[0], 0.3), 'rubber'))
    if riser > 1.2:
        h = riser - 1.2
        z1 = -hz - 1.2
        top = Pos(0, 0, z1) * Box(2 * hx + 4, 2 * hy + 2, 0.8, align=(Align.CENTER, Align.CENTER, Align.MAX))
        legs = [Pos(s * (hx + 2 - 0.4), 0, z1) * Box(0.8, 2 * hy + 2, h, align=(Align.CENTER, Align.CENTER, Align.MAX))
                for s in (-1, 1)]
        stand = top + legs[0] + legs[1]
        stand = fillet(stand.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1], 0.6)
        out.append((stand, 'aluminum_anodized'))
    if cord is not None:
        from mech import cable
        a = (-hx * 0.45, hy + 0.4, -hz * 0.35)
        pts = [a, (a[0], a[1] + 3, a[2] - 0.3), (a[0] * 0.5 + cord[0] * 0.5, a[1] + 6, cord[2] + 1.5),
               (cord[0], cord[1] - 2, cord[2]), cord]
        out.append((cable.tube(pts, 0.9, seg=14), 'rubber'))
        out.append((Pos(*a) * Cylinder(1.3, 2.4, rotation=(90, 0, 0)), 'rubber'))
    return out


def supply_panel(hx, hy, hz, display, posts, knobs, labels=True):
    """The supply's moulded front bezel (the scene's panel box, centred on its origin, looking along -y): a
    window frame round the display (display = (cx, cz, half w, half h) in the panel's frame), two knobs at
    knobs [(x, z)], recesses round the binding posts at posts [(x, z)], and the panel legend."""
    front = -hy
    plate = Box(2 * hx, 2 * hy, 2 * hz)
    plate = fillet(plate.edges().filter_by(Axis.Y), 1.0)
    plate = chamfer(plate.edges().group_by(Axis.Y)[0], 0.2)
    cx, cz, dw, dh = display
    win = Pos(cx, front, cz) * Box(2 * dw + 0.2, 4 * hy, 2 * dh + 0.2)
    plate = plate - win
    frame_o = Pos(cx, front - 0.35, cz) * Box(2 * dw + 1.6, 0.7, 2 * dh + 1.6)
    frame_o = fillet(frame_o.edges().filter_by(Axis.Y), 0.5)
    frame = frame_o - Pos(cx, front - 0.35, cz) * Box(2 * dw + 0.2, 2, 2 * dh + 0.2)
    frame = chamfer(frame.edges().group_by(Axis.Y)[0], 0.25)
    out = [(plate, 'plastic_black'), (frame, 'lcd_bezel')]
    for x, z in posts:
        ring = Pos(x, front, z) * Cylinder(2.5, 0.3, rotation=(90, 0, 0))
        out.append((ring, 'steel_dark'))
    for x, z in knobs:
        for s, m in knob(1.9, 2.4):
            out.append((Pos(x, front, z) * Rot(90, 0, 0) * s, m))
    if labels:
        for (x, z), word in zip(knobs, ('V', 'A')):
            out.append(text(word, 1.1, 0, 0, 0, 'panel_print', plane=front_plane(x + (3.3 if x > 0 else -3.3), front - 0.015, z)))
        for (x, z), mark in zip(posts, ('+', '\u2013')):
            out.append(text(mark, 1.3, 0, 0, 0, 'panel_print', plane=front_plane(x + (3.2 if x > 0 else -3.2), front - 0.015, z)))
    return out


# ---------------------------------------------------------------- through hole parts

def cap_top(r, hh, vent='+'):
    """The pressed aluminium top of a radial electrolytic can (the scene's top disc, radius r, half height hh):
    a domed rim and the scored safety vent, a cross or, on larger cans, a K."""
    disc = Cylinder(r, 2 * hh)
    disc = fillet(disc.edges().group_by(Axis.Z)[-1], min(hh * 1.6, r * 0.1))
    cuts = []
    L = r * 0.62
    w, d = max(0.08, r * 0.035), hh * 0.9
    arms = [0, 90] if vent == '+' else [90, 30, -30]
    for a in arms:
        if vent == 'K' and a != 90:
            cuts.append(Pos(0, 0, hh) * Rot(0, 0, a) * Pos(L / 2, 0, 0) * Box(L, w, d))
        else:
            cuts.append(Pos(0, 0, hh) * Rot(0, 0, a) * Box(2 * L, w, d))
    return [(disc - Compound(cuts), 'cap_sleeve_top')]


def toroid_core(R, r):
    """A powder iron toroid core with its coated, rounded section (the scene's torus: radius R, tube r)."""
    return [(Torus(R, r), 'ferrite')]


def toroid_winding(R, r, wire, turns=18, gap_at=-90.0, gap=60.0, leads=()):
    """Enamelled copper turns wound round a toroid (major radius R, core tube r), each turn a loop of wire
    tilted a little along the core so the winding reads as a helix, with a gap where the ends leave toward
    leads [(x, y, z)] in the torus's frame (axis z)."""
    from mech import cable
    out = []
    rr = r + wire
    span = 360.0 - gap
    a0 = gap_at + gap / 2
    pitch = span / turns
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    lean = math.radians(pitch) * 0.5
    for k in range(turns):
        a = math.radians(a0 + pitch * (k + 0.5))
        # a loop round the core's section at angle a, leaned along the core so the turns read as a helix
        ca, sa = math.cos(a), math.sin(a)
        ang = a + lean * np.sin(t)
        rad = R + rr * np.cos(t)
        ring = np.c_[rad * np.cos(ang), rad * np.sin(ang), rr * np.sin(t)]
        out.append((cable.tube(np.vstack([ring, ring[:1]]), wire, seg=8, smooth_path=False, caps=False), 'copper_wire'))
    for end, lead in zip((a0, a0 + span), leads):
        a = math.radians(end)
        start = (math.cos(a) * (R + rr), math.sin(a) * (R + rr), 0.0)
        mid = (start[0] * 0.6 + lead[0] * 0.4, start[1] * 0.6 + lead[1] * 0.4, lead[2] * 0.3)
        out.append((cable.tube([start, mid, lead], wire, seg=10), 'copper_wire'))
    return out


def do41(r, hh):
    """A DO-41 axial rectifier's moulded body (the scene's cylinder, radius r, half length hh) with rounded ends."""
    b = Cylinder(r, 2 * hh)
    return [(fillet(b.edges(), r * 0.35), 'epoxy_black')]


def to220_tab(hx, hy, hz, hole_r):
    """A TO-220's nickel plated copper tab (the scene's box) with its mounting hole, side notches and the
    chamfered top corners."""
    tab = Box(2 * hx, 2 * hy, 2 * hz)
    tab = chamfer(tab.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1], hx * 0.12)
    cuts = [Cylinder(hole_r, 4 * hy, rotation=(90, 0, 0))]
    for s in (-1, 1):
        cuts.append(Pos(s * hx, 0, -hz * 0.15) * Box(0.7, 4 * hy, 1.2))
    return [(tab - Compound(cuts), 'tin')]


def to220_lead(r, hh):
    """One TO-220 lead (the scene's capsule, half length hh): a flat 0.8 mm strip, wide at the body."""
    t = 0.5
    tip = Box(0.8, t, 2 * hh)
    tip = chamfer(tip.edges().group_by(Axis.Z)[0], 0.12)
    shoulder = Pos(0, 0, hh) * Box(1.4, t, 0.9 * hh, align=(Align.CENTER, Align.CENTER, Align.MAX))
    return [(tip + shoulder, 'tin')]


def dip_leads(pins, board_z, body_hy, body_bottom):
    """The stamped leads of a DIP package in the body's frame: pins [(x, y)] where each enters the board,
    board_z the board's surface. Each lead leaves the body's side, bends down, and narrows at the seating
    shoulder into the hole."""
    out = []
    t = 0.25
    for x, y in pins:
        s = 1 if y > 0 else -1
        z_exit = body_bottom + 0.9
        arm = Pos(x, s * (body_hy + (abs(y) - body_hy) / 2), z_exit) * Box(1.5, abs(y) - body_hy + t, t)
        drop = Pos(x, y, (z_exit + board_z + 1.0) / 2) * Box(1.5, t, z_exit - board_z - 1.0 + t / 2)
        shoulder = loft([Plane.XY.offset(board_z + 1.0) * Rectangle(1.5, t), Plane.XY.offset(board_z + 0.6) * Rectangle(0.5, t)])
        shoulder = Pos(x, y, 0) * shoulder
        tip = Pos(x, y, board_z + 0.6) * Box(0.5, t, 1.4, align=(Align.CENTER, Align.CENTER, Align.MAX))
        out += [(arm, 'tin'), (drop, 'tin'), (shoulder, 'tin'), (tip, 'tin')]
    return out


def led_lead(hh, inward, kink=False):
    """A 5 mm LED's lead in its capsule's frame (axis z, half length hh, the board at z=-hh): straight up out of
    the hole, then splayed in by inward (a vector in the lead's frame) to meet the epoxy, the longer anode
    with the crimp a lead forming tool leaves."""
    from mech import cable
    ix, iy = inward
    n = math.hypot(ix, iy) or 1.0
    px, py = -iy / n, ix / n           # across the splay, where the crimp bulges
    up = -hh + 2 * hh * 0.45
    pts = [(0, 0, -hh - 0.6), (0, 0, -hh + 0.3)]
    if kink:
        pts += [(px * 0.5, py * 0.5, -hh + 0.75), (0, 0, -hh + 1.2)]
    pts += [(0, 0, up), (ix * 0.5, iy * 0.5, up + (hh - up) * 0.6), (ix, iy, hh + 0.3)]
    sq = cable.tube(pts, 0.23, seg=8, n=40)
    return [(sq, 'tin')]


# ---------------------------------------------------------------- board parts

looks.add('cap_sleeve', color=(0.012, 0.04, 0.13), metal=0.0, rough=0.3)
looks.add('bung', color=(0.03, 0.03, 0.03), metal=0.0, rough=0.8)


def radial_can(r, hh):
    """A radial electrolytic's can in its heat shrunk sleeve (the scene's cylinder, radius r, half height hh): the
    sleeve rolled over the top edge, the crimped groove that holds the bung, and the rubber bung below."""
    can = Cylinder(r, 2 * hh)
    can = fillet(can.edges().group_by(Axis.Z)[-1], r * 0.12)
    can = fillet(can.edges().group_by(Axis.Z)[0], r * 0.06)
    groove = Pos(0, 0, -hh + 2 * hh * 0.16) * Torus(r + 0.02, r * 0.06)
    can = can - groove
    bung = Pos(0, 0, -hh) * Cylinder(r * 0.82, 0.25, align=(Align.CENTER, Align.CENTER, Align.MAX))
    return [(can, 'cap_sleeve'), (bung, 'bung')]


def screw_terminal(hx, hy, hz, n, entry=1):
    """A PCB screw terminal block of n ways along x (the scene's box): screw wells in the top, square wire entries
    with their rising cage clamps on the +y (entry=1) or -y face, and the moulded ribs between ways."""
    body = Box(2 * hx, 2 * hy, 2 * hz)
    body = chamfer(body.edges().filter_by(Axis.X).group_by(Axis.Y)[-1 if entry < 0 else 0].group_by(Axis.Z)[-1], hz * 0.35)
    pitch = 2 * hx / n
    xs = [-hx + pitch * (k + 0.5) for k in range(n)]
    cuts = []
    for x in xs:
        cuts.append(Pos(x, 0, hz) * Cylinder(pitch * 0.36, hz * 0.7))
        cuts.append(Pos(x, entry * hy, -hz * 0.15) * Box(pitch * 0.55, hy * 1.4, hz * 0.7))
    for k in range(1, n):
        cuts.append(Pos(-hx + pitch * k, entry * hy, -hz * 0.15) * Box(0.12, 0.5, hz * 1.2))
    body = body - Compound(cuts)
    out = [(body, 'terminal_blue')]
    for x in xs:
        out.append((Pos(x, entry * hy * 0.55, -hz * 0.15) * Box(pitch * 0.5, hy * 0.5, hz * 0.62), 'tin'))
    pins = [Pos(x, 0, -hz) * Box(0.5, 0.5, 0.8, align=(Align.CENTER, Align.CENTER, Align.MAX)) for x in xs]
    return out + [(p, 'tin') for p in pins]


def slotted_screw(r, hh):
    """A captive slotted cheese head screw (the scene's cylinder), its slot across the head."""
    head = Cylinder(r, 2 * hh)
    head = chamfer(head.edges().group_by(Axis.Z)[-1], hh * 0.4)
    head = head - Pos(0, 0, hh) * Box(2.2 * r, r * 0.28, hh * 1.4)
    return [(head, 'steel_zinc')]


def soic(hx, hy, hz, board_z, n=8, pitch=1.27):
    """A SOIC package (the scene's box, half sizes, centred) with its gull wing leads down to the board at
    board_z (in the body's frame), the pin one dimple and the moulded edge draft."""
    body = Box(2 * hx, 2 * hy, 2 * hz)
    body = chamfer(body.edges().group_by(Axis.Z)[-1], hz * 0.25)
    body = body - Pos(-hx + 0.7, -hy + 0.7, hz) * Sphere(0.32)
    out = [(body, 'epoxy_black')]
    per = n // 2
    t, w = 0.2, 0.42
    for side in (-1, 1):
        for k in range(per):
            x = (k - (per - 1) / 2) * pitch
            y0 = side * hy
            z0 = 0.0
            # out of the body's side, down the knee, and along the pad
            arm = Pos(x, y0 + side * 0.25, z0) * Box(w, 0.5, t)
            knee = Pos(x, y0 + side * 0.55, (z0 + board_z) / 2) * Box(w, t, abs(z0 - board_z) + t)
            foot = Pos(x, y0 + side * 0.85, board_z + t / 2) * Box(w, 0.7, t)
            out += [(arm, 'tin'), (knee, 'tin'), (foot, 'tin')]
    return out
