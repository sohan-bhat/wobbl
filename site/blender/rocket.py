"""The rocket's context models (examples/rocket-launch): the airframe with its paint, roll pattern and rail buttons,
the ogive nose cone, the motor mount tail with its retainer and graphite nozzle, three fins at their true
thickness with rounded edges and root fillets, and the launch pad with its T slot rail.

The nose, airframe, nozzle and fins share the rocket's pose, so they are built straight in its own frame: z along
the axis, nose up, every length from the run's scene.wobbl.

    blender -b --factory-startup -P site/blender/rocket.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import kit  # noqa: E402
import numpy as np  # noqa: E402
import wobscene  # noqa: E402
from kit import assign, band, bevel, box, circle, frame_matrix, join, lathe, plate, sweep, text, wrap  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

EX = 'rocket-launch'
SC = wobscene.Scene(os.path.join(kit.REPO, 'examples', EX, 'output', 'scene.wobbl'))
F = SC.home()


def mesh_of(name):
    return SC.mesh(SC.find(name)[0])[0]


BODY = mesh_of('rocket')
R = float(np.hypot(BODY[:, 0], BODY[:, 1]).max())          # airframe radius
Z_AFT, Z_TOP = float(BODY[:, 2].min()), float(BODY[:, 2].max())
NOSE = mesh_of('nose')
FINS = mesh_of('fins')


def nose_profile():
    r = np.hypot(NOSE[:, 0], NOSE[:, 1])
    z = np.round(NOSE[:, 2], 5)
    out = []
    for zz in sorted(set(z)):
        out.append((float(r[z == zz].max()), float(zz)))
    return out


def fin_planform():
    """Root and tip of one fin from the scene's fins: (root r, root z aft, root z fore, tip r, tip z aft, tip z fore)."""
    P = FINS[:8]                  # the first fin, standing along +y
    ys, zs = P[:, 1], P[:, 2]
    root = P[np.isclose(ys, ys.min())]
    tip = P[np.isclose(ys, ys.max())]
    return float(ys.min()), float(root[:, 2].min()), float(root[:, 2].max()), float(ys.max()), float(tip[:, 2].min()), float(tip[:, 2].max()), \
        float(np.abs(P[:, 0]).max())


def build():
    kit.reset(1.0)
    M = kit.mats(1.0)
    white, red, gloss, decal = M['paint_white'](), M['paint_red'](), M['paint_gloss_black'](), M['decal']()
    graphite, anod, alum, nylon, zinc = M['graphite'](), M['aluminum_anodized'](), M['aluminum_machined'](), M['nylon'](), M['steel_zinc']()
    models = {}
    wall = 0.0018

    # ---- airframe: a wound tube with a coupler seam, paint lines, a roll pattern, the name and two rail buttons
    seam = -0.45
    prof = [(R - wall, Z_AFT), (R - 0.0004, Z_AFT), (R, Z_AFT + 0.0004), (R, seam - 0.0012), (R - 0.0005, seam - 0.0007), (R - 0.0005, seam + 0.0007),
            (R, seam + 0.0012), (R, Z_TOP - 0.0004), (R - 0.0004, Z_TOP), (R - wall, Z_TOP)]
    tube = lathe('airframe', prof + [prof[0]], n=96, smooth=True)
    assign(tube, white)
    parts = [tube]
    lift = R + 0.00006
    for z0, z1 in ((Z_TOP - 0.024, Z_TOP - 0.020), (-1.13, -1.11), (-0.70, -0.66)):
        b = band('stripe', lift, z0, z1, n=96)
        assign(b, decal)
        parts.append(b)
    for k in range(4):     # the black and white roll pattern, a quarter each
        if k % 2:
            continue
        b = band('roll', lift, -0.30, -0.20, k * math.pi / 2, (k + 1) * math.pi / 2, n=24)
        assign(b, decal)
        parts.append(b)
    for word, z, size, ang in (('WOBBL', -0.02, 0.034, math.radians(-30)), ('WOBBL', -0.02, 0.034, math.radians(150)),
                               ('W-1  98 MM', -0.95, 0.012, math.radians(-30))):
        t = text('name', word, size)
        t.rotation_euler = (0, 0, 0)
        kit.bake_transform(t)
        # lay the line along the body (x up the tube), then wrap it round the tube facing the given side
        for v in t.data.vertices:
            x, y, _ = v.co
            v.co = (z + x, -y + ang * lift, 0)
        sub = t.modifiers.new('s', 'SUBSURF')
        sub.subdivision_type = 'SIMPLE'
        sub.levels = 1
        kit.apply_mods(t)
        wrap(t, lift)
        assign(t, decal)
        parts.append(t)
    vent = band('vent', lift + 0.00002, 0.10, 0.104, math.radians(267), math.radians(273), n=6)
    assign(vent, decal)
    parts.append(vent)
    # rail buttons on the side between two fins, nylon spools on a pan head screw
    rb_ang = math.radians(270)
    for z in (-1.02, -0.18):
        spool = lathe('rail_button', [(0, 0), (0.0058, 0), (0.0058, 0.0015), (0.0040, 0.0025), (0.0040, 0.0060), (0.0062, 0.0068), (0.0062, 0.0082),
                                      (0, 0.0082)], n=32)
        bevel(spool, 0.0004, 2, harden=False)
        assign(spool, nylon)
        screw = lathe('rb_screw', [(0, 0.0081), (0.0026, 0.0081), (0.0022, 0.0090), (0, 0.0092)], n=20)
        assign(screw, zinc)
        rb = join('rb', [spool, screw])
        radial = Vector((math.cos(rb_ang), math.sin(rb_ang), 0))
        rb.matrix_world = frame_matrix(radial * (R - 0.0006) + Vector((0, 0, z)), Vector((0, 0, 1)), None, radial)
        parts.append(rb)
    # the centring ring and motor tube seen at the open aft end
    ring = lathe('centring_ring', [(0.0310, Z_AFT + 0.0016), (R - wall, Z_AFT + 0.0016), (R - wall, Z_AFT + 0.0076), (0.0310, Z_AFT + 0.0076), (0.0310, Z_AFT + 0.0016)], n=64)
    assign(ring, M['plastic_black']())
    mmt = lathe('motor_tube', [(0.0262, Z_AFT + 0.0004), (0.0310, Z_AFT + 0.0004), (0.0310, Z_AFT + 0.14), (0.0262, Z_AFT + 0.14), (0.0262, Z_AFT + 0.0004)], n=64)
    assign(mmt, M['plastic_black']())
    parts += [ring, mmt]
    models['rocket'] = join('airframe', parts)

    # ---- nose: the run's ogive, with a rounded tip and a thin seam where it meets the airframe
    np_ = nose_profile()
    zt = np_[-1][1]
    tip_r = 0.0035
    prof = [(0, np_[0][1] + 0.0003), (np_[0][0] - 0.0004, np_[0][1] + 0.0003), (np_[0][0], np_[0][1] + 0.0009)]
    prof += [(r, z) for r, z in np_[1:-2]]
    for k in range(1, 7):
        a = math.pi / 2 * k / 6
        prof.append((tip_r * math.cos(a) * 1.4, zt - tip_r + tip_r * math.sin(a)))
    nose = lathe('nose', prof, n=96)
    assign(nose, red)
    models['nose'] = nose

    # ---- the motor mount tail: anodised retainer, the motor's aft closure and its graphite nozzle
    noz = mesh_of('nozzle')
    zn0, zn1 = float(noz[:, 2].max()), float(noz[:, 2].min())   # -1.36 to -1.409
    ret = lathe('retainer', [(0.0262, zn0 + 0.0005), (0.0318, zn0 + 0.0005), (0.0322, zn0 - 0.0010), (0.0322, zn0 - 0.0118), (0.0312, zn0 - 0.0128),
                             (0.0255, zn0 - 0.0128), (0.0255, zn0 + 0.0005)], n=72)
    assign(ret, anod)
    knurl = []
    for k in range(36):
        a = 2 * math.pi * k / 36
        b = box('knurl', (-0.0005, -0.0006, zn0 - 0.0112), (0.0005, 0.0006, zn0 - 0.0016))
        b.matrix_world = Matrix.Rotation(a, 4, 'Z') @ Matrix.Translation((0.0322, 0, 0))
        assign(b, anod)
        knurl.append(b)
    closure = lathe('aft_closure', [(0, zn0 - 0.0128), (0.0248, zn0 - 0.0128), (0.0248, zn0 - 0.0170), (0.0242, zn0 - 0.0176), (0.0242, zn0 - 0.0184),
                                    (0.0236, zn0 - 0.0190), (0, zn0 - 0.0190)], n=72)
    assign(closure, alum)
    nz = lathe('nozzle', [(0, zn0 - 0.0157), (0.0147, zn0 - 0.0196), (0.0118, zn0 - 0.0216), (0.0186, zn1), (0.0214, zn1), (0.0219, zn1 + 0.0010),
                          (0.0226, zn0 - 0.0190), (0, zn0 - 0.0189)], n=72)
    assign(nz, graphite)
    models['nozzle'] = join('motor_tail', [ret, closure, nz] + knurl)

    # ---- three fins at the scene's thickness, rounded on every edge, with epoxy fillets along each root
    r0, za0, zf0, r1, za1, zf1, half = fin_planform()
    fins = []
    for k in range(3):
        phi = math.pi / 2 + k * 2 * math.pi / 3
        radial = Vector((math.cos(phi), math.sin(phi), 0))
        out = [(R - 0.004, za0), (r1, za1), (r1, zf1), (R - 0.004, zf0)]
        f = plate('fin', [(x, y) for x, y in out], -half, half)
        bevel(f, half * 0.85, 4, angle=30, harden=False)
        f.matrix_world = frame_matrix((0, 0, 0), radial, (0, 0, 1))
        assign(f, gloss)
        fins.append(f)
        for side in (1, -1):
            fr = 0.0065

            def prof(s, side=side):
                k_ = min(1.0, min(s, 1 - s) / 0.12)
                rr = fr * (0.25 + 0.75 * math.sqrt(k_))
                pts = [(half * 0.6, R - 0.0012), (half + rr, R - 0.0012 - (half + rr) ** 2 / (2 * R))]
                for j in range(1, 8):
                    a = math.pi / 2 * j / 8
                    pts.append((half + rr - rr * math.sin(a), R + rr - rr * math.cos(a)))
                pts.append((half * 0.6, R + rr))
                return [(side * x, y) for x, y in (pts if side > 0 else list(reversed(pts)))]
            fl = sweep('fillet', za0 + 0.004, zf0 - 0.004, prof, steps=30)
            # sweep lays (tangent, radial) in x, y along z: turn that frame onto this fin
            tang = Vector((0, 0, 1)).cross(radial)
            Mf = Matrix.Identity(4)
            for i in range(3):
                Mf[i][0], Mf[i][1], Mf[i][2] = tang[i], radial[i], (0, 0, 1)[i]
            fl.matrix_world = Mf
            kit.smooth_by_angle(fl, 60)
            assign(fl, gloss)
            fins.append(fl)
    models['fins'] = join('fins', fins)

    # ---- launch pad: a steel table with a tread plate top, a frame and feet; the rail is a T slot extrusion
    pi = SC.find('launch pad')[0]
    hx, hy, hz = SC.objects[pi]['size']
    c = SC.pose(pi, F)[0]
    top = box('pad_top', (c[0] - hx, c[1] - hy, c[2] + hz - 0.008), (c[0] + hx, c[1] + hy, c[2] + hz))
    bevel(top, 0.0015, 2)
    assign(top, M['tread_steel']())
    parts = [top]
    fr_mat = M['paint_black']()
    for s in (-1, 1):
        for axis in (0, 1):
            lo = [c[0] - hx + 0.01, c[1] - hy + 0.01, c[2] - hz + 0.012]
            hi = [c[0] + hx - 0.01, c[1] + hy - 0.01, c[2] + hz - 0.008]
            if axis == 0:
                lo[1], hi[1] = (c[1] + s * (hy - 0.05)) - 0.02, (c[1] + s * (hy - 0.05)) + 0.02
            else:
                lo[0], hi[0] = (c[0] + s * (hx - 0.05)) - 0.02, (c[0] + s * (hx - 0.05)) + 0.02
            b = box('frame', lo, hi)
            bevel(b, 0.003, 2)
            assign(b, fr_mat)
            parts.append(b)
    for sx in (-1, 1):
        for sy in (-1, 1):
            fx, fy = c[0] + sx * (hx - 0.05), c[1] + sy * (hy - 0.05)
            foot = lathe('foot', [(0, c[2] - hz), (0.035, c[2] - hz), (0.035, c[2] - hz + 0.006), (0.008, c[2] - hz + 0.010), (0.008, c[2] - hz + 0.013),
                                  (0, c[2] - hz + 0.013)], n=32, at=(fx, fy, 0))
            assign(foot, M['rubber']())
            parts.append(foot)
    models['launch pad'] = join('launch_pad', parts)

    ri = SC.find('launch rail')[0]
    rh = SC.objects[ri]['size']
    w, L = rh[0], rh[2]
    slot, lip, deep = w * 0.30, w * 0.26, w * 0.62

    def rail_profile(s):
        pts = []
        for k in range(4):
            rot = k * math.pi / 2
            side = [(w, -w * 0.82), (w, -slot), (w - lip, -slot), (w - lip, -w * 0.52), (w - deep, -w * 0.36), (w - deep, w * 0.36),
                    (w - lip, w * 0.52), (w - lip, slot), (w, slot), (w, w * 0.82), (w * 0.82, w)]
            for x, y in side:
                pts.append((x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)))
        return pts
    rail = sweep('rail', -L, L, rail_profile, steps=1)
    bevel(rail, w * 0.03, 2, angle=40, harden=False)
    assign(rail, M['aluminum_machined']())
    models['launch rail'] = rail
    return models


FILES = {'rocket': 'airframe', 'nose': 'nose-cone', 'nozzle': 'motor-tail', 'fins': 'fins', 'launch pad': 'launch-pad', 'launch rail': 'rail'}
LOCAL = {'launch pad'}          # built in the scene's world, moved into the object's frame on export


def main():
    models = build()
    if os.environ.get('WOBBL_PREVIEW'):
        return models
    entries = []
    for name, ob in models.items():
        kit.finish(ob, 1.0, ao=512 if name in ('rocket', 'fins') else 256, ao_distance=0.05 if name != 'launch pad' else 0.2)
        if name in LOCAL:
            kit.to_local(ob, SC.matrix(SC.find(name)[0], F))
        rel = f'{EX}/{FILES[name]}.glb'
        kit.export(ob, rel)
        entries.append({'object': name, 'file': rel})
    kit.write_manifest(EX, {'replace': entries})


if __name__ == '__main__':
    main()
