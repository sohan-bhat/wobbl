"""Rolling bearings at catalogue dimensions: shielded deep groove ball bearings (ISO 15), the flanged miniature kind
that idler pivots use, an open thrust ball bearing and a four point slewing ring for turntables, with their balls.

Each bearing stands on z = 0 with its bore on the z axis and rises to its width. Functions return
[(shape, material)] lists: rings in bearing steel, shields in zinc plate, balls in chrome.

    from mech import bearings as br
    parts = br.deep_groove('608')
"""
import math

from build123d import Axis, Plane, Polyline, Pos, Rot, Solid, make_face, revolve

# bore, outer diameter, width
SIZES = {
    '603': (3, 9, 5), '623': (3, 10, 4), '624': (4, 13, 5), '625': (5, 16, 5), '626': (6, 19, 6), '608': (8, 22, 7),
    '609': (9, 24, 7), '6000': (10, 26, 8), '6001': (12, 28, 8), '6200': (10, 30, 9), '6201': (12, 32, 10),
    '6800': (10, 19, 5), '6802': (15, 24, 5), '6804': (20, 32, 7), '6805': (25, 37, 7), '6806': (30, 42, 7),
    '6808': (40, 52, 7), 'MR63': (3, 6, 2.5), 'MR85': (5, 8, 2.5), 'MR105': (5, 10, 4), '688': (8, 16, 5),
}
# flanged: flange diameter and thickness
FLANGE = {'F623': ('623', 11.5, 1.0), 'F624': ('624', 15.0, 1.0), 'F625': ('625', 18.0, 1.2), 'F608': ('608', 25.0, 1.5),
          'MF105': ('MR105', 11.5, 0.8), 'F688': ('688', 18.0, 1.1)}


def _rev(profile):
    pts = []
    for p in profile:
        if not pts or (abs(pts[-1][0] - p[0]) > 1e-7 or abs(pts[-1][1] - p[1]) > 1e-7):
            pts.append(p)
    return revolve(make_face(Plane.XZ * Polyline(*pts, close=True)), Axis.Z)


def _ring(r0, r1, z0, z1, c):
    """A ring section from radius r0 to r1 and z0 to z1 with chamfered corners of c."""
    return _rev([(r0 + c, z0), (r1 - c, z0), (r1, z0 + c), (r1, z1 - c), (r1 - c, z1), (r0 + c, z1), (r0, z1 - c), (r0, z0 + c)])


def deep_groove(code, shields=True, flange=None):
    """A deep groove ball bearing: inner and outer rings with chamfered edges, the shields set just below the faces
    with their circlip groove line, or the open cage and balls when shields is False. flange (diameter, thickness)
    adds an outer flange at the top face."""
    d, D, B = SIZES[code]
    c = max(0.15, min(0.6, B * 0.06))
    pitch_r = (d + D) / 4
    t_out = (D - d) * 0.17
    out = []
    outer = _ring(D / 2 - t_out, D / 2, 0, B, c)
    if flange:
        fd, ft = flange
        outer = outer + _ring(D / 2 - 0.2, fd / 2, B - ft, B, c * 0.6)
    inner = _ring(d / 2, d / 2 + t_out, 0, B, c)
    out += [(outer, 'steel_bare'), (inner, 'steel_bare')]
    recess = B * 0.08
    if shields:
        for z0 in (recess, B - recess - 0.25):
            sh = _rev([(d / 2 + t_out * 0.98, z0), (D / 2 - t_out * 0.98, z0), (D / 2 - t_out * 0.98, z0 + 0.25),
                       (d / 2 + t_out * 0.98, z0 + 0.25)])
            out.append((sh, 'steel_zinc'))
    else:
        rb = (D - d) * 0.15
        n = max(6, int(2 * math.pi * pitch_r / (2.6 * rb)))
        for i in range(n):
            a = 2 * math.pi * i / n
            out.append((Pos(pitch_r * math.cos(a), pitch_r * math.sin(a), B / 2) * Solid.make_sphere(rb), 'steel_chrome'))
        cage = _ring(pitch_r - rb * 0.45, pitch_r + rb * 0.45, B / 2 - rb * 0.55, B / 2 + rb * 0.55, 0.05)
        for i in range(n):
            a = 2 * math.pi * i / n
            cage = cage - Pos(pitch_r * math.cos(a), pitch_r * math.sin(a), B / 2) * Solid.make_sphere(rb * 1.05)
        out.append((cage, 'steel_zinc'))
    return out


def flanged(code, shields=True):
    """A flanged miniature bearing such as an F623 idler, the flange at its top face."""
    base, fd, ft = FLANGE[code]
    return deep_groove(base, shields, flange=(fd, ft))


def thrust_ring(d, D, h, balls=True):
    """An open thrust ball bearing: two flat washers with the ball ring between, as a lazy susan turntable uses."""
    t = h * 0.3
    c = 0.3
    out = [(_ring(d / 2, D / 2, 0, t, c), 'steel_bare'), (_ring(d / 2, D / 2, h - t, h, c), 'steel_bare')]
    if balls:
        pr = (d + D) / 4
        rb = (h - 2 * t) / 2 + t * 0.35
        cage = _ring(pr - rb * 1.2, pr + rb * 1.2, h / 2 - rb * 0.4, h / 2 + rb * 0.4, 0.05)
        n = int(2 * math.pi * pr / (2.5 * rb))
        for i in range(n):
            a = 2 * math.pi * i / n
            p = Pos(pr * math.cos(a), pr * math.sin(a), h / 2)
            out.append((p * Solid.make_sphere(rb), 'steel_chrome'))
            cage = cage - p * Solid.make_sphere(rb * 1.04)
        out.append((cage, 'brass'))
    return out


def slewing_ring(d, D, h, bolts=8):
    """A four point contact slewing ring for a small turntable: an outer ring fixed to the base and an inner ring that
    turns, each drilled for its bolt circle, with the seal gap between them. Returns (outer, inner) lists."""
    mid = (d + D) / 4
    c = 0.5
    outer = _ring(mid + 0.4, D / 2, 0, h * 0.92, c)
    inner = _ring(d / 2, mid - 0.4, h * 0.08, h, c)
    hr = (D / 2 - mid) * 0.22
    for i in range(bolts):
        a = 2 * math.pi * (i + 0.5) / bolts
        ro = (mid + D / 2) / 2 + 0.2
        outer = outer - Pos(ro * math.cos(a), ro * math.sin(a), -h) * Solid.make_cylinder(hr, h * 3)
        a2 = 2 * math.pi * i / bolts
        ri = (mid + d / 2) / 2 - 0.2
        inner = inner - Pos(ri * math.cos(a2), ri * math.sin(a2), -h) * Solid.make_cylinder(hr, h * 3)
    seal = _ring(mid - 0.4, mid + 0.4, h * 0.45, h * 0.92 - 0.25, 0.05)
    return [(outer, 'steel_bare'), (seal, 'rubber')], [(inner, 'steel_bare')]
