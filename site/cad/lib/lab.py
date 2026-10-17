"""Laboratory and field props, from catalogue drawings: a Bunsen burner with its cast base, needle valve, brass
barrel and slotted air collar, a hose barb with a rubber gas hose, a clay graphite crucible set in a sand bed with
crucible tongs and a magnesium ribbon fuse, a Griffin beaker with rolled rim, spout and printed graduations and a
glass stirring rod, and angular talus boulders. Builders return [(shape, material)] Z up in millimetres.
"""
import math

import numpy as np
from build123d import Align, Axis, Box, Compound, Cylinder, Polygon, Pos, Rot, chamfer, extrude, fillet

import looks
from electronics import front_plane, text
from rocketry import lathe

looks.add('crucible_clay', color=(0.09, 0.085, 0.08), metal=0.0, rough=0.75, normal='cast_n', nstr=0.6,
          rvar=(0.65, 0.85), tiles=60)
looks.add('crucible_glaze', color=(0.16, 0.12, 0.08), metal=0.0, rough=0.4, normal='cast_n', nstr=0.3,
          rvar=(0.32, 0.5), tiles=60)
looks.add('sand', color=(0.25, 0.2, 0.13), metal=0.0, rough=0.92, normal='fine_n', nstr=1.6, tiles=60)
looks.add('magnesium', color=(0.72, 0.72, 0.7), metal=1.0, rough=0.35)
looks.add('gas_hose', color=(0.55, 0.12, 0.04), metal=0.0, rough=0.55)
looks.add('brass_turned', color=(0.86, 0.66, 0.36), metal=1.0, rough=0.26)
looks.add('print_white', color=(0.9, 0.9, 0.88), metal=0.0, rough=0.45)


# ---------------------------------------------------------------- Bunsen burner

def bunsen(barrel_r=6.0, base_r=16.0, base_h=6.0, top=90.0, collar=(24.0, 34.0)):
    """A Bunsen burner on the scene's profile: a round cast iron base with its needle valve knob, a brass barrel
    and a knurled air collar whose two slots line up with the barrel's air holes."""
    out = []
    base = lathe([(0, 0), (base_r, 0), (base_r, 1.2), (base_r - 0.8, 2.5), (base_r * 0.69, 5.4), (barrel_r + 0.6, base_h + 0.2),
                  (barrel_r + 0.4, base_h + 2.0), (0, base_h + 2.0)])
    out.append((base, 'cast_iron'))
    felt = Pos(0, 0, 0.01) * Cylinder(base_r - 1.5, 0.4, align=(Align.CENTER, Align.CENTER, Align.MAX))
    out.append((felt, 'rubber'))
    barrel = lathe([(barrel_r - 1.5, base_h + 2), (barrel_r, base_h + 2), (barrel_r, top - 1.4), (barrel_r - 0.4, top),
                    (barrel_r - 1.5, top)])
    holes = [Rot(0, 0, a) * Pos(barrel_r, 0, (collar[0] + collar[1]) / 2) * Box(4, 3.2, 5.0) for a in (0, 180)]
    out.append((barrel - Compound(holes), 'brass_turned'))
    z0, z1 = collar
    sleeve = lathe([(barrel_r + 0.02, z0), (barrel_r + 1.3, z0), (barrel_r + 1.6, z0 + 0.5), (barrel_r + 1.6, z1 - 0.5),
                    (barrel_r + 1.3, z1), (barrel_r + 0.02, z1)])
    slots = [Rot(0, 0, a + 20) * Pos(barrel_r + 1, 0, (z0 + z1) / 2) * Box(4, 3.4, 5.4) for a in (0, 180)]
    knurl = [Rot(0, 0, k * 10) * Pos(barrel_r + 1.65, 0, (z0 + z1) / 2) * Rot(0, 0, 45) * Box(0.35, 0.35, z1 - z0 - 2.4)
             for k in range(36)]
    out.append((sleeve - Compound(slots + knurl), 'brass_turned'))
    # the needle valve: a knurled knob under the base's rim
    knob = Pos(0, -base_r + 1.0, 2.6) * Rot(90, 0, 0) * Cylinder(3.2, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    knob = chamfer(knob.edges(), 0.4)
    out.append((knob, 'plastic_black'))
    return out


def hose_barb(r, hl):
    """The gas inlet stub along the scene's cylinder (radius r, half length hl along z): a hex boss and a barb
    with three ridges."""
    prof = [(0, -hl), (r * 0.8, -hl)]
    for k in range(3):
        z = -hl + (k + 0.3) * (2 * hl / 3.2)
        prof += [(r * 0.72, z), (r, z + 2 * hl / 6.4), (r * 0.72, z + 2 * hl / 6.4 + 0.01)]
    prof += [(r * 0.8, hl), (0, hl)]
    return [(lathe(prof), 'brass_turned')]


def gas_hose(points, r=4.2):
    from mech import cable
    return [(cable.tube(points, r, seg=16, n=60), 'gas_hose')]


# ---------------------------------------------------------------- crucible

def crucible(outline, wall_top=None):
    """A clay graphite crucible on the scene's outline [(z, r inner, r outer)], with a glazed rim, a rounded foot
    and a pouring lip."""
    (zb, ri_b, ro_b), (z0, ri0, _), (zt, ri_t, ro_t) = outline
    prof = [(0, zb), (ro_b - 3, zb), (ro_b, zb + 3), (ro_t, zt - 2), (ro_t - 0.8, zt), (ri_t + 0.8, zt), (ri_t, zt - 1.5),
            (ri0, z0 + 4), (ri0 - 4, z0), (0, z0)]
    body = lathe(prof)
    lip = Pos(ro_t - 2, 0, zt - 2) * Rot(0, 30, 0) * Box(8, 16, 4)
    body = body - lip
    # the glaze runs down from the rim; the body is split there rather than coated, so no two faces overlap
    split = Pos(0, 0, zt - 7 + 50) * Box(400, 400, 100)
    return [(body - split, 'crucible_clay'), (body & split, 'crucible_glaze')]


def sand_bed(r_in, r_out, z_floor, z_top):
    """A mound of dry sand round the crucible's foot, on a steel tray."""
    mound = lathe([(r_in, z_floor), (r_out, z_floor), (r_out * 0.8, z_floor + (z_top - z_floor) * 0.45), (r_in + 4, z_top),
                   (r_in, z_top)])
    tray = lathe([(0, z_floor - 2), (r_out + 12, z_floor - 2), (r_out + 12, z_floor + 10), (r_out + 10.5, z_floor + 10),
                  (r_out + 10.5, z_floor - 0.5), (0, z_floor - 0.5)])
    return [(mound, 'sand'), (tray, 'steel_mill')]


def tongs(length=260.0, at=(0, 0, 0), angle=0.0):
    """Crucible tongs lying on the ground: two forged bars crossed at a rivet, with curved bow jaws."""
    out = []
    for s in (-1, 1):
        bar = Box(length * 0.75, 4.0, 6.0)
        bar = fillet(bar.edges().filter_by(Axis.X), 1.5)
        out.append((Pos(length * 0.25, s * 3.5, 4) * Rot(0, 0, s * 3) * bar, 'steel_black_oxide'))
        jaw = Pos(-length * 0.18, s * 14, 4) * (Cylinder(18, 6) - Cylinder(14, 8)) & Pos(-length * 0.18, s * 26, 4) * Box(40, 26, 8)
        out.append((jaw, 'steel_black_oxide'))
    rivet = Pos(-length * 0.12, 0, 4) * Cylinder(4.0, 9.0)
    out.append((rivet, 'steel_zinc'))
    from mech.util import place
    return place(out, Pos(*at) * Rot(0, 0, angle))


def mg_ribbon(z0, height=46.0, width=3.0, t=0.25):
    """A magnesium ribbon fuse standing out of the charge, curling over at its free end."""
    from mech import cable
    C = cable.smooth([(0, 0, z0 - 6), (0, 0, z0 + 6), (2, 1, z0 + height * 0.5), (-1, 3, z0 + height * 0.85),
                      (3, 7, z0 + height), (6, 9, z0 + height - 6)], 40)
    side = np.array([1.0, 0, 0])
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    S = side - (T @ side)[:, None] * T
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    Nn = np.cross(T, S)
    P = np.concatenate([C + S * width / 2 + Nn * t / 2, C - S * width / 2 + Nn * t / 2,
                        C - S * width / 2 - Nn * t / 2, C + S * width / 2 - Nn * t / 2])
    N = np.concatenate([Nn, Nn, -Nn, -Nn])
    n = len(C)
    tris = []
    for i in range(n - 1):
        for a, b in ((0, 1), (2, 3)):
            p0, p1, q0, q1 = a * n + i, b * n + i, a * n + i + 1, b * n + i + 1
            tris += [(p0, q0, p1), (p1, q0, q1)]
    return [(('mesh', P, N, np.array(tris)), 'magnesium')]


# ---------------------------------------------------------------- beaker

def beaker(r_out, h, wall=1.8, volume=1000, step=50):
    """A Griffin beaker on the scene's outline: rolled rim, pouring spout, and printed white graduations every step
    millilitres with their figures, the capacity and the glass type."""
    t = wall
    prof = [(0, 0), (r_out - t * 2, 0), (r_out, t * 2), (r_out, h - t), (r_out + t * 0.6, h - t * 0.4), (r_out + t * 0.6, h),
            (r_out - t, h), (r_out - t, t * 3), (r_out - t * 3, t), (0, t)]
    glass = lathe(prof)
    spout = Pos(r_out, 0, h) * Rot(0, 20, 0) * Box(12, 22, 10)
    glass = glass - spout
    out = [(glass, 'glass')]
    ri = r_out - t
    ml_per_mm = math.pi * ri * ri / 1000.0
    ang0 = -100.0                                # the scale faces the page's view, off to one side of the spout
    lift = r_out + 0.08
    for v in range(step, volume, step):
        z = t + v / ml_per_mm
        if z > h - 12:
            break
        major = v % 100 == 0
        w = 9.0 if major else 5.0
        a = math.radians(ang0)
        tick = Pos(math.cos(a) * lift, math.sin(a) * lift, z) * Rot(0, 0, ang0) * Box(0.3, w, 0.5)
        out.append((tick, 'print_white'))
        if major and v % 200 == 0:
            ax = math.radians(ang0 - 9)
            pl = _tangent_plane(lift + 0.3, ax, z)
            out.append(text(str(v), 3.2, 0, 0, 0, 'print_white', plane=pl))
    a = math.radians(ang0 + 50)
    out.append(text(f'{volume} ml', 5.0, 0, 0, 0, 'print_white', plane=_tangent_plane(lift + 0.4, a, h * 0.75)))
    out.append(text('BORO 3.3', 3.0, 0, 0, 0, 'print_white', plane=_tangent_plane(lift + 0.3, a, h * 0.75 - 8)))
    return out


def _tangent_plane(r, a, z):
    """A plane on a cylinder of radius r at angle a and height z, looking out along the radius, x round it."""
    from build123d import Plane
    n = (math.cos(a), math.sin(a), 0)
    return Plane(origin=(n[0] * r, n[1] * r, z), x_dir=(-math.sin(a), math.cos(a), 0), z_dir=n)


def stirring_rod(base, top, r=3.0):
    """A glass stirring rod from base to top, its ends fire polished round."""
    from mech import cable
    return [(cable.tube([base, top], r, seg=14, n=2), 'glass')]


# ---------------------------------------------------------------- talus

def boulder(size, seed, at=(0, 0, 0), sink=0.25):
    """An angular talus boulder: the hull of random points on a squashed ellipsoid, sunk a little into the
    ground, as one ready mesh with flat faces."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(28, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.75, 1.0, size=(28, 1))
    v *= np.array([1.0, rng.uniform(0.7, 1.0), rng.uniform(0.45, 0.7)]) * size / 2
    hull = ConvexHull(v)
    tris = hull.simplices.copy()
    c = v.mean(0)
    for k, (a, b, d) in enumerate(tris):
        if np.dot(np.cross(v[b] - v[a], v[d] - v[a]), v[a] - c) < 0:
            tris[k] = (a, d, b)
    P = v[tris].reshape(-1, 3)
    fn = np.cross(P[1::3] - P[0::3], P[2::3] - P[0::3])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    N = np.repeat(fn, 3, axis=0)
    zmin = P[:, 2].min()
    P = P + np.array([at[0], at[1], at[2] - zmin - sink * (v[:, 2].max() - zmin)])
    return ('mesh', P, N, np.arange(len(P)).reshape(-1, 3))
