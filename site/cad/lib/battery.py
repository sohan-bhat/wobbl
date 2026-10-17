"""18650 cells and pack hardware, from cell and connector datasheets: a cell's PVC wrap folded over the shoulders,
the positive cap with its vent holes and insulator ring, the flat negative can, nickel strips with spot weld
dimples, silicone leads to an XT60, a JST-XH balance lead, a taped type K thermocouple and a ceramic cell stand.
Builders return [(shape, material)] Z up in millimetres, a cell standing along z centred on the origin.
"""
import math

import numpy as np
from build123d import Align, Axis, Box, Compound, Cylinder, Pos, Rot, Sphere, Torus, chamfer, fillet

import electronics  # noqa: F401  (registers the shared metals)
import looks

looks.add('wrap_blue', color=(0.02, 0.07, 0.3), metal=0.0, rough=0.25)
looks.add('fish_paper', color=(0.62, 0.55, 0.42), metal=0.0, rough=0.7)
looks.add('nickel', color=(0.68, 0.67, 0.64), metal=1.0, rough=0.28)
looks.add('silicone_red', color=(0.45, 0.02, 0.02), metal=0.0, rough=0.55)
looks.add('silicone_black', color=(0.012, 0.012, 0.012), metal=0.0, rough=0.55)
looks.add('xt60_yellow', color=(0.75, 0.5, 0.02), metal=0.0, rough=0.42)
looks.add('kapton', color=(0.5, 0.22, 0.02), metal=0.0, rough=0.18)
looks.add('alumina', color=(0.8, 0.79, 0.74), metal=0.0, rough=0.6)
looks.add('tc_yellow', color=(0.7, 0.55, 0.03), metal=0.0, rough=0.4)
looks.add('wire_white', color=(0.75, 0.75, 0.73), metal=0.0, rough=0.45)


def torus_mesh(R, r, z=0.0, seg=56, ring=10):
    """A thin torus about z as a ready mesh: a CAD torus with a thin tube costs thousands of triangles."""
    u = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    v = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')
    N = np.stack([np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)], -1).reshape(-1, 3)
    P = np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), z + r * np.sin(V)], -1).reshape(-1, 3)
    T = []
    for i in range(seg):
        for j in range(ring):
            a, b = i * ring + j, i * ring + (j + 1) % ring
            c, d = ((i + 1) % seg) * ring + j, ((i + 1) % seg) * ring + (j + 1) % ring
            T += [(a, c, b), (b, c, d)]
    return ('mesh', P, N, np.array(T))


def cell_ends(r, hh, wrap='wrap_blue'):
    """The parts of a cell outside its can's plain cylinder (radius r, half length hh): the wrap's folded lip
    over each shoulder, the insulator ring and button cap with its vent holes on top, the negative can face."""
    out = []
    for s in (-1, 1):
        out.append((torus_mesh(r - 0.55, 0.62, s * hh), wrap))
    ring = Pos(0, 0, hh + 0.05) * (Cylinder(r - 1.1, 0.25) - Cylinder(r * 0.42, 1))
    out.append((ring, 'fish_paper'))
    cap = Pos(0, 0, hh + 0.2) * Cylinder(r * 0.4, 0.9, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = fillet(cap.edges().group_by(Axis.Z)[-1], 0.3)
    vents = [Rot(0, 0, k * 90 + 45) * Pos(r * 0.4, 0, hh + 0.65) * Cylinder(0.45, 1.2, rotation=(0, 90, 0)) for k in range(4)]
    cap = cap - Compound(vents)
    out.append((cap, 'nickel'))
    shoulder = Pos(0, 0, hh) * (Cylinder(r * 0.62, 0.3, align=(Align.CENTER, Align.CENTER, Align.MIN)) - Cylinder(r * 0.42, 1))
    out.append((shoulder, 'nickel'))
    base = Pos(0, 0, -hh - 0.05) * Cylinder(r - 1.0, 0.12)
    out.append((base, 'nickel'))
    return out


def cell(r, hh, wrap='wrap_blue'):
    """A whole 18650: the wrapped can and its ends."""
    can = Cylinder(r, 2 * hh)
    return [(can, wrap)] + cell_ends(r, hh, wrap)


def strip(a, b, z, w=7.0, t=0.15, welds=((0, 0),)):
    """A nickel strip from cell centre a to cell centre b (x, y) at height z, with a pair of spot weld dimples on
    each cell it crosses."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    L = float(np.linalg.norm(d))
    ang = math.degrees(math.atan2(d[1], d[0]))
    m = (a + b) / 2
    body = Pos(m[0], m[1], z) * Rot(0, 0, ang) * Box(L + w, w, t)
    body = fillet(body.edges().filter_by(Axis.Z), w * 0.2)
    dents = []
    for c in (a, b):
        for dx in (-1.6, 1.6):
            for dy in (-1.2, 1.2):
                ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                p = c + np.array([ux * dx - uy * dy, uy * dx + ux * dy])
                dents.append(Pos(p[0], p[1], z + t / 2 + 0.25) * Sphere(0.38))
    return [(body - Compound(dents), 'nickel')]


def xt60(x, y, z, axis='+x'):
    """An XT60 plug, its mating face along axis from (x, y, z): the yellow nylon shell with the chamfered
    polarity corners and two gold plated bullets."""
    shell = Box(16.0, 8.0, 15.8)
    shell = chamfer(shell.edges().filter_by(Axis.X).group_by(Axis.Y)[-1], 2.2)
    shell = shell - Compound([Pos(s * 3.6, 0, 7.9) * Cylinder(2.1, 8.0) for s in (-1, 1)])
    pins = [Pos(s * 3.6, 0, 6.0) * Cylinder(1.75, 6.0) for s in (-1, 1)]
    rot = {'+x': Rot(0, 90, 0), '-x': Rot(0, -90, 0), '+y': Rot(-90, 0, 0), '-y': Rot(90, 0, 0), '+z': Rot(0, 0, 0), '-z': Rot(180, 0, 0)}[axis]
    out = [(Pos(x, y, z) * rot * shell, 'xt60_yellow')] + [(Pos(x, y, z) * rot * p, 'gold') for p in pins]
    return out


def lead(points, r, mat):
    from mech import cable
    return [(cable.tube(points, r, seg=10, n=max(24, len(points) * 10)), mat)]


def jst_xh(x, y, z, n=3, axis='+x'):
    """A JST-XH balance plug: the white housing with its locking ramp, n ways at 2.5 mm."""
    w = 2.5 * n + 2.4
    shell = Box(w, 5.7, 7.0)
    ramp = Pos(0, 2.85 + 0.4, 1.2) * Box(w * 0.4, 0.8, 2.0)
    rot = {'+x': Rot(0, 90, 0), '-x': Rot(0, -90, 0), '+y': Rot(-90, 0, 0), '-y': Rot(90, 0, 0), '+z': Rot(0, 0, 0)}[axis]
    return [(Pos(x, y, z) * rot * (shell + ramp), 'plastic_white')]


def thermocouple(r, z, angle=0.0, wire_to=None):
    """A type K thermocouple bead held to a cell's side (radius r) at height z by a band of Kapton tape, its
    twisted pair running off toward wire_to (a point)."""
    out = []
    band = Pos(0, 0, z) * (Cylinder(r + 0.09, 8.0) - Cylinder(r - 0.01, 9.0))
    out.append((band, 'kapton'))
    a = math.radians(angle)
    bead = Pos(math.cos(a) * (r + 0.55), math.sin(a) * (r + 0.55), z) * Sphere(0.6)
    out.append((bead, 'nickel'))
    if wire_to is not None:
        from mech import cable
        p0 = np.array([math.cos(a) * (r + 0.6), math.sin(a) * (r + 0.6), z])
        out_dir = np.array([math.cos(a), math.sin(a), 0.0])
        for k, mat in enumerate(('tc_yellow', 'silicone_red')):
            off = np.array([-math.sin(a), math.cos(a), 0]) * (0.35 if k else -0.35)
            pts = [p0 + off, p0 + off + out_dir * 3 + np.array([0, 0, 2]), (np.asarray(wire_to) + p0) / 2 + off + np.array([0, 0, 6]),
                   np.asarray(wire_to) + off]
            out.append((cable.tube(pts, 0.32, seg=8, n=48), mat))
    return out


def cell_stand(r, h, z_top):
    """A ceramic cell stand: an alumina disc with a shallow seat for the cell's negative end."""
    disc = Pos(0, 0, z_top - h / 2) * Cylinder(r * 1.45, h)
    disc = chamfer(disc.edges().group_by(Axis.Z)[-1], 0.8)
    disc = disc - Pos(0, 0, z_top) * Cylinder(r + 0.3, 1.6)
    return [(disc, 'alumina')]
