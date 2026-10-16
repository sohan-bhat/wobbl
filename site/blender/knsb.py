"""The static fire stand round the KNSB motor (examples/knsb-motor): a welded steel stand on a concrete pad, rubber
lined cradles under the casing, a floating thrust plate on guide rods against a button load cell and its
backstop, the machined forward closure and nozzle, and the radial bolts that hold them in the casing.

The casing itself keeps its FE mesh and its live stress colors. The closure and nozzle are cut open on the same
wedge the scene cuts the casing on, so the grains still show. Lengths in mm, as the scene stores them.

    blender -b --factory-startup -P site/blender/knsb.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import kit  # noqa: E402
import numpy as np  # noqa: E402
import wobscene  # noqa: E402
from kit import assign, bevel, box, circle, cut, frame_matrix, hexprism, join, lathe, plate, sweep  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

EX = 'knsb-motor'
SC = wobscene.Scene(os.path.join(kit.REPO, 'examples', EX, 'output', 'scene.wobbl'))
F = SC.home()
MOTOR = SC.matrix(SC.find('casing')[0], F)        # the motor's own frame: z along the axis from the forward end
AXIS_Z = float(MOTOR[2, 3])                        # height of the motor's axis above the ground


def profile_of(name):
    """Outer and inner radius of a turned mesh at each station along its axis."""
    P = SC.mesh(SC.find(name)[0])[0]
    r, z = np.hypot(P[:, 0], P[:, 1]), np.round(P[:, 2], 3)
    return [(float(zz), float(r[z == zz].min()), float(r[z == zz].max())) for zz in sorted(set(z))]


def cut_angle():
    """The wedge the scene cuts the casing open on, as a half angle about local -x."""
    P = SC.mesh(SC.find('casing')[0])[0]
    r = np.hypot(P[:, 0], P[:, 1])
    return math.acos(-P[:, 0].min() / r.max())


def wedge(name, half, z0, z1, R=60.0):
    pts = [(0, 0)] + [(R * math.cos(math.pi - half + 2 * half * i / 16), R * math.sin(math.pi - half + 2 * half * i / 16)) for i in range(17)]
    return plate(name, pts, z0, z1)


def build():
    kit.reset(1000.0)
    M = kit.mats(1000.0)
    mill, steel, zinc, alum = M['steel_mill'](), M['steel_bare'](), M['steel_zinc'](), M['aluminum_machined']()
    rubber, concrete, black, stain = M['rubber'](), M['concrete'](), M['paint_black'](), M['steel_stainless']()
    models, adds = {}, {}
    half = cut_angle()
    toW = Matrix(MOTOR.tolist())

    # ---- nozzle: the run's convergent divergent bore in a turned steel body with a flange and a chamfered exit
    st = profile_of('nozzle')
    (z0, ri0, ro0), (z1, ri1, ro1), (z2, ri2, ro2), (z3, ri3, ro3), (z4, ri4, ro4) = st
    outer = [(ri4, z4), (ro4 - 0.6, z4), (ro4, z4 - 0.6), (ro3, z3), (ro2, z2 + 1.2), (ro2 + 1.2, z2), (ro1 - 0.8, z1 + 0.6), (ro1, z1),
             (ro0, z0 + 0.8), (ro0 - 0.8, z0)]
    inner = [(ri0, z0), (ri1, z1), (ri2 + 0.8, z2 - 1.5), (ri2, z2), (ri2 + 0.4, z2 + 1.5), (ri3, z3), (ri4, z4)]
    nz = lathe('nozzle', outer + inner, n=96)
    cut(nz, wedge('nz_cut', half, z0 - 1, z4 + 1))
    bevel(nz, 0.35, 2, angle=40)
    assign(nz, steel)
    # graphite throat insert, as amateur nozzles are often built
    gi = lathe('throat', [(ri2 + 0.05, z2 - 6), (ri2 + 3.2, z2 - 6), (ri2 + 3.2, z2 + 6), (ri2 + 0.05, z2 + 6), (ri2 + 0.05, z2 - 6)], n=64)
    cut(gi, wedge('gi_cut', half, z2 - 7, z2 + 7))
    assign(gi, M['graphite']())
    models['nozzle'] = join('nozzle', [nz, gi])

    # ---- forward closure: a turned aluminium bulkhead with an o ring groove and a pressure port
    fc = profile_of('forward closure')
    fz0, fz1, fr = fc[0][0], fc[-1][0], fc[0][2]
    clo = lathe('closure', [(0, fz0), (fr - 0.6, fz0), (fr, fz0 + 0.6), (fr, fz1 - 3.6), (fr - 1.2, fz1 - 3.2), (fr - 1.2, fz1 - 1.6), (fr, fz1 - 1.2),
                            (fr, fz1), (0, fz1)], n=96)
    port = lathe('port', [(0, fz0 - 0.01), (3.2, fz0 - 0.01), (3.2, fz0 + 2.5), (0, fz0 + 2.5)], n=32)
    cut(clo, port)
    cut(clo, wedge('fc_cut', half, fz0 - 1, fz1 + 1))
    assign(clo, alum)
    models['forward closure'] = clo

    # ---- the radial bolts through the casing into the closure and the nozzle, clear of the cut
    bolts = []
    for zb in (fz1 + 6.5, z0 - 6.5):
        for k in range(8):
            a = 2 * math.pi * k / 8
            if abs((a - math.pi + math.pi) % (2 * math.pi) - math.pi) < half + 0.2:
                continue
            radial = Vector((math.cos(a), math.sin(a), 0))
            head = lathe('bhcs', [(0, 0), (4.6, 0), (4.6, 0.6), (4.0, 1.6), (2.8, 2.5), (0, 2.7)], n=32)
            sock = hexprism('sock', 2.5, 1.6, 3.0)
            cut(head, sock)
            assign(head, M['steel_black_oxide']())
            head.matrix_world = toW @ frame_matrix(radial * 26.95 + Vector((0, 0, zb)), Vector((0, 0, 1)), None, radial)
            bolts.append(head)
    adds['bolts'] = join('casing_bolts', bolts)

    # ---- the stand, in the scene's world: base rails and cross members on a concrete pad, anchored
    x_lo, x_hi = -300.0, 175.0
    parts = []
    for s in (-1, 1):
        rail = box('rail', (x_lo, s * 30 - 10, 0), (x_hi, s * 30 + 10, 15))
        bevel(rail, 1.2, 2)
        parts.append(rail)
    for x in (x_lo + 10, x_hi - 10):
        xm = box('cross', (x - 10, -48, 0), (x + 10, 48, 12))
        bevel(xm, 1.0, 2)
        parts.append(xm)
    base = join('base', parts)
    assign(base, mill)
    anchors = []
    for x in (x_lo + 10, x_hi - 10):
        for y in (-40, 40):
            w = lathe('washer', [(5.2, 12), (11, 12), (11, 14), (5.2, 14)], n=32, at=(x, y, 0))
            n_ = hexprism('nut', 13, 14, 20.5, at=(x, y))
            bevel(n_, 0.3, 2)
            th = lathe('stud', [(0, 12), (5, 12), (5, 23), (4.4, 23.6), (0, 23.6)], n=24, at=(x, y, 0))
            for o in (w, n_, th):
                assign(o, zinc)
            anchors += [w, n_, th]
    pad = box('pad', (-340, -150, -70), (260, 150, 0))
    bevel(pad, 6.0, 3, harden=False)
    assign(pad, concrete)
    adds['stand'] = join('stand', [base, pad] + anchors)

    # ---- cradles: a welded upright with a U cut to the casing, a rubber liner and a foot bolted to the rails
    sad = SC.find('stand saddle')
    rcase = st[0][2]
    for i, oi in enumerate(sad[:1]):
        c = SC.pose(oi, F)[0]
        sx, sy, sz = SC.objects[oi]['size']
        cx = c[0]
        pts = [(-36, 21), (36, 21), (36, AXIS_Z)]
        for k in range(33):
            a = 2 * math.pi - (math.pi * k / 32)
            pts.append(((rcase + 1.6) * math.cos(a), AXIS_Z + (rcase + 1.6) * math.sin(a)))
        pts.append((-36, AXIS_Z))
        up = plate('upright', pts, -4, 4)
        up.matrix_world = frame_matrix((cx, 0, 0), (0, 1, 0), (0, 0, 1))
        bevel(up, 0.8, 2)
        foot = box('foot', (cx - 18, -42, 15), (cx + 18, 42, 21))
        bevel(foot, 0.8, 2)
        gus = []
        for s in (-1, 1):
            g = plate('gusset', [(0, 21), (14, 21), (0, 40)], -2.5, 2.5)
            g.matrix_world = frame_matrix((cx, s * 30, 0), (1, 0, 0), (0, 0, 1))
            gus.append(g)
            g2 = plate('gusset', [(0, 21), (-14, 21), (0, 40)], -2.5, 2.5)
            g2.matrix_world = frame_matrix((cx, s * 30, 0), (1, 0, 0), (0, 0, 1))
            gus.append(g2)
        cradle = join('cradle', [up, foot] + gus)
        assign(cradle, mill)
        liner = []
        a0, a1 = math.radians(200), math.radians(340)

        def lin(s):
            pts = []
            for k in range(25):
                a = a0 + (a1 - a0) * k / 24
                pts.append(((rcase + 0.05) * math.cos(a), AXIS_Z + (rcase + 0.05) * math.sin(a)))
            for k in range(24, -1, -1):
                a = a0 + (a1 - a0) * k / 24
                pts.append(((rcase + 1.65) * math.cos(a), AXIS_Z + (rcase + 1.65) * math.sin(a)))
            return pts
        ln = sweep('liner', -9, 9, lin, steps=1)
        ln.matrix_world = Matrix(((0, 0, 1, cx), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
        assign(ln, rubber)
        liner.append(ln)
        hw = []
        for x in (cx - 11, cx + 11):
            for y in (-34, 34):
                h = hexprism('fb', 10, 21, 25.5, at=(x, y))
                bevel(h, 0.25, 2)
                w = lathe('fw', [(3.2, 21), (8.5, 21), (8.5, 22.2), (3.2, 22.2)], n=24, at=(x, y, 0))
                for o in (h, w):
                    assign(o, zinc)
                hw += [h, w]
        sm = join('saddle', [cradle] + liner + hw)
        models['stand saddle'] = sm

    # ---- thrust plate floating on two guide rods, pressing a button load cell against the welded backstop
    ti = SC.find('thrust plate')[0]
    c = SC.pose(ti, F)[0]
    hx, hy, hz = SC.objects[ti]['size']
    xf = c[0] + hx                   # the face the closure bears on
    xb = c[0] - hx
    tp = plate('thrust_plate', [(-hy, 17), (hy, 17), (hy, 2 * hz - 6), (hy - 6, 2 * hz), (-hy + 6, 2 * hz), (-hy, 2 * hz - 6)], xb, xf,
               holes=[(-32, 72, 6.5), (32, 72, 6.5)])
    tp.matrix_world = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    bevel(tp, 0.8, 2)
    assign(tp, mill)
    cell = lathe('load_cell', [(0, 0), (19, 0), (19.5, 1), (19.5, 17), (19, 18), (7, 18), (6, 20.5), (0, 20.5)], n=64, at=(0, 0, 0))
    cell.matrix_world = Matrix.Translation((xb, 0, AXIS_Z)) @ Matrix.Rotation(-math.pi / 2, 4, 'Y')
    assign(cell, stain)
    gland = lathe('gland', [(0, 0), (4.5, 0), (4.5, 6), (3.2, 7), (3.2, 9), (0, 9)], n=24)
    gland.matrix_world = Matrix.Translation((xb - 10, 19.2, AXIS_Z)) @ Matrix.Rotation(-math.pi / 2, 4, 'X')
    assign(gland, M['plastic_black']())
    cable = kit.tube_along('cable', [(xb - 10, 27, AXIS_Z), (xb - 10, 40, AXIS_Z - 4), (xb - 14, 52, AXIS_Z - 30), (xb - 22, 58, 18), (xb - 40, 62, 16)], 2.6)
    assign(cable, M['rubber']())
    xs = xb - 20.5                    # the backstop's face
    bs = plate('backstop', [(-48, 0), (48, 0), (48, 90), (42, 96), (-42, 96), (-48, 90)], xs - 14, xs)
    bs.matrix_world = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    bevel(bs, 0.8, 2)
    gs = []
    for y in (-40, 40):
        g = plate('bs_gusset', [(0, 15), (-46, 15), (0, 80)], -4, 4)
        g.matrix_world = frame_matrix((xs - 14, y, 0), (1, 0, 0), (0, 0, 1))
        gs.append(g)
    back = join('backstop', [bs] + gs)
    bevel(back, 0.6, 2)
    assign(back, mill)
    rods = []
    for y in (-32, 32):
        rd = lathe('rod', [(0, 0), (5.6, 0), (6, 0.5), (6, 74.5), (5.6, 75), (0, 75)], n=32)
        rd.matrix_world = Matrix.Translation((xs, y, 72)) @ Matrix.Rotation(math.pi / 2, 4, 'Y')
        assign(rd, M['steel_chrome']())
        bush = lathe('bushing', [(6, 0), (9, 0), (9, 16), (6, 16)], n=32)
        bush.matrix_world = Matrix.Translation((xb - 16, y, 72)) @ Matrix.Rotation(math.pi / 2, 4, 'Y')
        assign(bush, zinc)
        nut = hexprism('rod_nut', 13, 0, 6, at=(0, 0))
        nut.matrix_world = Matrix.Translation((xs + 75 + 0.01, y, 72)) @ Matrix.Rotation(math.pi / 2, 4, 'Y')
        assign(nut, zinc)
        rods += [rd, bush, nut]
    models['thrust plate'] = join('thrust_plate', [tp, cell, gland, cable, back] + rods)
    return models, adds, toW


FILES = {'nozzle': 'nozzle', 'forward closure': 'forward-closure', 'stand saddle': 'cradle', 'thrust plate': 'thrust-plate'}
ADDS = {'stand': 'stand', 'bolts': 'casing-bolts'}


def main():
    models, adds, toW = build()
    if os.environ.get('WOBBL_PREVIEW'):
        for k in ('nozzle', 'forward closure'):
            models[k].matrix_world = toW @ models[k].matrix_world
        return {**models, **adds}
    entries, extra = [], []
    for name, ob in models.items():
        kit.finish(ob, 1000.0, ao=512 if name in ('thrust plate', 'stand saddle') else 256, ao_distance=25)
        if name == 'stand saddle':
            kit.to_local(ob, SC.matrix(SC.find(name)[0], F))
        elif name == 'thrust plate':
            kit.to_local(ob, SC.matrix(SC.find(name)[0], F))
        rel = f'{EX}/{FILES[name]}.glb'
        kit.export(ob, rel)
        entries.append({'object': name, 'file': rel})
    for name, ob in adds.items():
        kit.finish(ob, 1000.0, ao=512, ao_distance=40)
        rel = f'{EX}/{ADDS[name]}.glb'
        kit.export(ob, rel)
        extra.append({'file': rel})
    kit.write_manifest(EX, {'replace': entries, 'add': extra, 'hide': ['ground']})


if __name__ == '__main__':
    main()
