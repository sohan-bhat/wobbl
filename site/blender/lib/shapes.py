"""Shapes for the context models: turned parts, flat plates from outlines, lofted bars, hex nuts, booleans and
the bevels every real edge has, built with bmesh in Blender.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

def _obj(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def hull2d(pts):
    """Convex hull of 2D points, counter clockwise (monotone chain)."""
    P = sorted(set((round(x, 9), round(y, 9)) for x, y in pts))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def circle(cx, cy, r, n=48, a0=0.0, a1=2 * math.pi):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n)]


def plate(name, outline, z0, z1, holes=(), M=None):
    """A flat part from a 2D outline (counter clockwise), extruded from z0 to z1, with round holes (x, y, r)."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, z0)) for x, y in outline]
    f = bm.faces.new(vs)
    if f.normal.z > 0:
        f.normal_flip()
    r = bmesh.ops.extrude_face_region(bm, geom=[f])
    top = [e for e in r['geom'] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=top, vec=(0, 0, z1 - z0))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = _obj(name, bm)
    for i, (x, y, rr) in enumerate(holes):
        c = cylinder(f'{name}_h{i}', rr, z0 - 1, z1 + 1, at=(x, y), n=40)
        cut(ob, c)
    if M is not None:
        ob.matrix_world = M
    return ob


UNIT = 1.0   # model units per metre: 1 for a scene in metres, 1000 for one in millimetres


def lathe(name, prof, n=None, at=(0, 0, 0), axis='Z', smooth=True):
    """A turned part from a profile [(r, z), ...] in order, spun about z (or another axis). Without n, small
    parts get fewer sides, since a 6 mm ball covers a few pixels on the page."""
    if n is None:
        rmax = max(r for r, z in prof) / UNIT
        n = int(round(24 + 48 * min(1.0, rmax / 0.04) / 4)) * 4
    bm = bmesh.new()
    vs = []
    for r, z in prof:
        vs.append(bm.verts.new((max(r, 0.0), 0, z)))
    es = [bm.edges.new((vs[i], vs[i + 1])) for i in range(len(vs) - 1)]
    bmesh.ops.spin(bm, geom=vs + es, cent=(0, 0, 0), axis=(0, 0, 1), angle=2 * math.pi, steps=n, use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = _obj(name, bm)
    R = {'Z': Matrix.Identity(4), 'X': Matrix.Rotation(math.pi / 2, 4, 'Y'), 'Y': Matrix.Rotation(-math.pi / 2, 4, 'X')}[axis]
    ob.matrix_world = Matrix.Translation(at) @ R
    if smooth:
        smooth_by_angle(ob)
    return ob


def cylinder(name, r, z0, z1, at=(0, 0), n=48):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=n, radius1=r, radius2=r, depth=z1 - z0)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(at[0], at[1], (z0 + z1) / 2))
    return _obj(name, bm)


def box(name, lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector([lo[k] + (v.co[k] + 0.5) * (hi[k] - lo[k]) for k in range(3)])
    return _obj(name, bm)


def hexprism(name, across_flats, z0, z1, at=(0, 0), chamfer=True, rot=0.0):
    """A nut or bolt head: a hexagon with the 30 degree chamfer rolled onto its corners."""
    R = across_flats / math.sqrt(3)
    pts = [(at[0] + R * math.cos(rot + math.pi / 6 + i * math.pi / 3), at[1] + R * math.sin(rot + math.pi / 6 + i * math.pi / 3)) for i in range(6)]
    ob = plate(name, pts, z0, z1)
    if chamfer:
        h = z1 - z0
        c = across_flats / 2
        k = lathe(name + '_ch', [(0, z0 - 1e-6), (c * 0.98, z0 - 1e-6), (R * 1.15, z0 + (R * 1.15 - c * 0.98) * 0.577 + 1e-6),
                                 (R * 1.15, z1 - (R * 1.15 - c * 0.98) * 0.577 - 1e-6), (c * 0.98, z1 + 1e-6), (0, z1 + 1e-6)],
                  n=48, at=(at[0], at[1], 0), smooth=False)
        intersect(ob, k)
    return ob


def tube_along(name, pts, r, n=16, cap=True):
    """A round section swept along a polyline of 3D points."""
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    sp = curve.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1)
    curve.bevel_depth = r
    curve.bevel_resolution = max(2, n // 4)
    curve.use_fill_caps = cap
    ob = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(ob)
    return to_mesh(ob)


def loft(name, length, section, n=24, steps=48, cap=True):
    """A bar along x from 0 to length whose cross section at t (0 to 1) is section(t) -> (half width y,
    half height z, z offset, squareness 0 round to 1 boxy), made of rings bridged into one closed surface."""
    bm = bmesh.new()
    rings = []
    for i in range(steps + 1):
        t = i / steps
        a, b, zc, sq = section(t)
        ring = []
        for j in range(n):
            th = 2 * math.pi * j / n
            c, s_ = math.cos(th), math.sin(th)
            e = 2.0 / (2.0 + 8.0 * sq)  # superellipse exponent: 1 round, small values boxy
            y = a * math.copysign(abs(c) ** e, c)
            z = b * math.copysign(abs(s_) ** e, s_)
            ring.append(bm.verts.new((t * length, y, zc + z)))
        rings.append(ring)
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(n):
            bm.faces.new((r0[j], r0[(j + 1) % n], r1[(j + 1) % n], r1[j]))
    if cap:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = _obj(name, bm)
    smooth_by_angle(ob, 50)
    return ob


def to_mesh(ob):
    select(ob)
    bpy.ops.object.convert(target='MESH')
    return bpy.context.view_layer.objects.active


def select(*obs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]


def apply_mods(ob):
    select(ob)
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    return ob


def _bool(ob, other, op, solver='EXACT'):
    m = ob.modifiers.new('b', 'BOOLEAN')
    m.operation = op
    m.object = other
    m.solver = solver
    apply_mods(ob)
    bpy.data.objects.remove(other, do_unlink=True)
    return ob


def cut(ob, other):
    return _bool(ob, other, 'DIFFERENCE')


def intersect(ob, other):
    return _bool(ob, other, 'INTERSECT')


def union(ob, other):
    return _bool(ob, other, 'UNION')


def bevel(ob, width, segments=3, angle=35, apply=True, harden=True):
    """Round every sharp edge a little, as every machined, cast or stamped edge is, so it catches a highlight."""
    m = ob.modifiers.new('bevel', 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'ANGLE'
    m.angle_limit = math.radians(angle)
    m.harden_normals = False
    m.miter_outer = 'MITER_ARC'
    m.use_clamp_overlap = True
    if apply:
        apply_mods(ob)
        smooth_by_angle(ob, 40)
        if harden:
            weighted_normals(ob)
    return ob


def weighted_normals(ob):
    m = ob.modifiers.new('wn', 'WEIGHTED_NORMAL')
    m.keep_sharp = True
    m.weight = 50
    apply_mods(ob)


def smooth_by_angle(ob, deg=35):
    select(ob)
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(deg))
    except Exception:
        bpy.ops.object.shade_smooth()


def subdivide(ob, levels=2):
    m = ob.modifiers.new('sub', 'SUBSURF')
    m.levels = levels
    m.render_levels = levels
    apply_mods(ob)
    return ob


def transform(ob, M):
    ob.matrix_world = M @ ob.matrix_world
    return ob


def bake_transform(ob):
    select(ob)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return ob


def join(name, obs):
    obs = [o for o in obs if o is not None]
    for o in obs:
        bake_transform(o)
    select(*obs)
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    return ob


def frame_matrix(origin, x, y=None, z=None):
    """A 4x4 matrix whose columns are the axes given (missing ones made orthonormal), at origin."""
    x = Vector(x).normalized()
    if z is not None:
        z = Vector(z).normalized()
        y = z.cross(x).normalized()
        x = y.cross(z)
    else:
        y = Vector(y).normalized()
        z = x.cross(y).normalized()
        y = z.cross(x)
    M = Matrix.Identity(4)
    for k in range(3):
        M[k][0], M[k][1], M[k][2], M[k][3] = x[k], y[k], z[k], origin[k]
    return M
