"""The wiper linkage's context models (examples/wiper-linkage): the wiper motor with its die cast gearbox and
rubber mounted ears, the stamped crank on its splined shaft, the pressed coupler rod with ball socket ends, the
rocker, the cast pivot tower, the arm, the blade and the glass.

Every joint is where the run's scene.wobbl solved it: the parts are built where the linkage stands on the frame
the example page opens on, then moved into each object's own frame for export, so they follow the motion.

    blender -b --factory-startup -P site/blender/wiper.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import kit  # noqa: E402
import wobscene  # noqa: E402
from kit import bevel, box, circle, hexprism, hull2d, join, lathe, loft, plate, assign  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

EX = 'wiper-linkage'
SC = wobscene.Scene(os.path.join(kit.REPO, 'examples', EX, 'output', 'scene.wobbl'))
F = SC.home()


def mat_of(name):
    return SC.matrix(SC.find(name)[0], F)


def end(name, sign):
    o = SC.objects[SC.find(name)[0]]
    return Vector(wobscene.world(mat_of(name), [0, 0, sign * o['size'][1]]))


O = end('crank', +1)        # the motor's output shaft
PC = end('crank', -1)       # crank pin, the coupler's first ball joint
PV = end('rocker', +1)      # the pivot the rocker and the wiper arm turn about
PR = end('rocker', -1)      # rocker end, the coupler's second ball joint
for v in (O, PC, PV, PR):
    v.z = 0.0

# heights in metres: the links stand in the plane z = 0 in the model, so they are stacked about it the way a real
# linkage is, the crank and rocker plates just under it and the coupler riding over both on its ball joints
PLATE = (-0.003, 0.001)
BALL_Z = 0.0125


def link_plate(name, A, B, rA, rB, mat):
    out = hull2d(circle(A.x, A.y, rA, 64) + circle(B.x, B.y, rB, 64))
    p = plate(name, out, *PLATE)
    bevel(p, 0.0007, 3)
    # the pressed stiffening rib along the middle
    u = (B - A).normalized()
    a, b = A + u * (rA + 0.006), B - u * (rB + 0.004)
    rib = plate(name + '_rib', hull2d(circle(a.x, a.y, 0.0048, 32) + circle(b.x, b.y, 0.0036, 32)), PLATE[1] - 0.0004, PLATE[1] + 0.0007)
    bevel(rib, 0.0005, 3)
    p = join(name, [p, rib])
    assign(p, mat)
    return p


def ball_stud(name, at, steel, zinc):
    collar = hexprism(name + '_hex', 0.011, PLATE[1], PLATE[1] + 0.0035, at=(at.x, at.y))
    bevel(collar, 0.0003, 2)
    assign(collar, zinc)
    neck = lathe(name + '_neck', [(0, PLATE[1] + 0.0034), (0.0042, PLATE[1] + 0.0034), (0.0036, 0.0075), (0.0032, BALL_Z - 0.004), (0, BALL_Z - 0.004)],
                 at=(at.x, at.y, 0))
    ball = lathe(name + '_ball', [(math.sin(t * math.pi / 16) * 0.0062, BALL_Z - math.cos(t * math.pi / 16) * 0.0062) for t in range(17)],
                 at=(at.x, at.y, 0))
    for o in (neck, ball):
        assign(o, steel)
    return join(name, [collar, neck, ball])


def socket(name, at, plastic, rubber):
    cup = lathe(name + '_cup', [(0, 0.0058), (0.0082, 0.0058), (0.0102, 0.0076), (0.0106, 0.0118), (0.0102, 0.0168), (0.0084, 0.0192), (0, 0.0196)],
                at=(at.x, at.y, 0))
    bevel(cup, 0.0004, 2)
    assign(cup, plastic)
    boot = lathe(name + '_boot', [(0.0044, PLATE[1] + 0.0036), (0.0062, PLATE[1] + 0.0040), (0.0072, 0.0050), (0.0066, 0.0054), (0.0080, 0.0059), (0.0040, 0.0059)],
                 at=(at.x, at.y, 0))
    assign(boot, rubber)
    return join(name, [cup, boot])


def build():
    kit.reset()
    M = kit.mats(1.0)
    paint, black, steel, zinc = M['paint'](), M['paint_black'](), M['steel_bare'](), M['steel_zinc']()
    cast, plastic, rubber = M['aluminum_cast'](), M['plastic_black'](), M['rubber']()
    glass, frit = M['glass'](), M['frit']()
    models = {}

    # ---- crank: stamped steel, tinted live by the motor's torque, on the splined shaft under a flanged nut
    crank = link_plate('crank', O, PC, 0.0135, 0.0105, paint)
    collar = lathe('crank_collar', [(0, PLATE[1] - 0.0002), (0.0105, PLATE[1] - 0.0002), (0.0105, PLATE[1] + 0.0022), (0.0094, PLATE[1] + 0.0033), (0, PLATE[1] + 0.0033)],
                   at=(O.x, O.y, 0))
    assign(collar, paint)
    washer = lathe('crank_washer', [(0, 0.0043), (0.0088, 0.0043), (0.0088, 0.0052), (0, 0.0052)], at=(O.x, O.y, 0))
    nut = hexprism('crank_nut', 0.013, 0.0052, 0.0112, at=(O.x, O.y))
    bevel(nut, 0.0003, 2)
    tip = lathe('crank_tip', [(0, 0.0112), (0.0040, 0.0112), (0.0040, 0.0122), (0.0034, 0.0128), (0, 0.0128)], at=(O.x, O.y, 0))
    for o in (washer, nut, tip):
        assign(o, zinc)
    models['crank'] = join('crank', [crank, collar, washer, nut, tip])

    # ---- the ball studs riveted into the crank and the rocker
    models['coupler pin'] = ball_stud('coupler_pin', PC, steel, zinc)

    # ---- coupler: a pressed tube flattened into paddles riveted over two ball sockets
    u = (PR - PC).normalized()
    L = (PR - PC).length
    over = 0.0115
    total = L + 2 * over
    t0 = over / total

    def section(t):
        # distance from the nearer joint, as a share of the paddle length
        d = min(t, 1 - t)
        tip = min(1.0, math.sqrt(max(0.0, 1 - (1 - min(d / t0, 1.0)) ** 2)) * 1.0 + 0.02)
        k = max(0.0, min(1.0, (d - t0 * 1.6) / (t0 * 2.2)))
        s = k * k * (3 - 2 * k)
        a = (0.0108 * (1 - s) + 0.0062 * s) * (tip if d < t0 else 1.0)
        b = 0.0016 * (1 - s) + 0.0046 * s
        zc = 0.0212 * (1 - s) + 0.0178 * s
        return a, b, zc, 0.85 * (1 - s) + 0.1 * s
    rod = loft('coupler_rod', total, section, n=28, steps=90)
    yaxis = Vector((0, 0, 1)).cross(u)
    rod.matrix_world = kit.frame_matrix(PC - u * over, u, yaxis)
    assign(rod, paint)
    parts = [rod]
    for i, at in enumerate((PC, PR)):
        parts.append(socket(f'coupler_socket{i}', at, plastic, rubber))
        rv = lathe(f'coupler_rivet{i}', [(0, 0.0226), (0.0050, 0.0226), (0.0046, 0.0236), (0.0030, 0.0244), (0, 0.0246)], at=(at.x, at.y, 0))
        assign(rv, steel)
        parts.append(rv)
    models['coupler'] = join('coupler', parts)

    # ---- rocker: stamped, splined onto the spindle with a nut underneath, its own ball stud at the far end
    rocker = link_plate('rocker', PV, PR, 0.016, 0.0105, black)
    rnut = hexprism('rocker_nut', 0.013, PLATE[0] - 0.0062, PLATE[0], at=(PV.x, PV.y))
    bevel(rnut, 0.0003, 2)
    assign(rnut, zinc)
    rtip = lathe('rocker_tip', [(0, PLATE[0] - 0.0062), (0.0040, PLATE[0] - 0.0062), (0.0040, PLATE[0] - 0.0072), (0.0034, PLATE[0] - 0.0078), (0, PLATE[0] - 0.0078)],
                 at=(PV.x, PV.y, 0))
    assign(rtip, steel)
    rstud = ball_stud('rocker_stud', PR, steel, zinc)
    models['rocker'] = join('rocker', [rocker, rnut, rtip, rstud])

    # ---- the spindle's splined boss and thrust washer between the rocker and the tower
    boss = lathe('rocker_pin', [(0, PLATE[1] - 0.0002), (0.0092, PLATE[1] - 0.0002), (0.0092, PLATE[1] + 0.0016), (0.0080, PLATE[1] + 0.0024),
                                (0.0068, PLATE[1] + 0.0024), (0.0068, 0.0046), (0, 0.0046)], at=(PV.x, PV.y, 0))
    assign(boss, steel)
    models['rocker pin'] = boss

    # ---- output shaft between the gearbox and the crank, with its seal
    shaft = lathe('crank_shaft', [(0, -0.0112), (0.0060, -0.0112), (0.0060, -0.0035), (0.0054, -0.0030), (0, -0.0030)], at=(O.x, O.y, 0))
    assign(shaft, steel)
    seal = lathe('crank_seal', [(0.0058, -0.0112), (0.0095, -0.0112), (0.0095, -0.0098), (0.0072, -0.0090), (0.0058, -0.0090)], at=(O.x, O.y, 0))
    assign(seal, rubber)
    models['crank shaft'] = join('crank_shaft', [shaft, seal])

    # ---- the wiper motor: die cast gearbox round the worm wheel, the worm tube, the painted can and three ears
    parts = []
    gb = lathe('gearbox', [(0, -0.0445), (0.0300, -0.0445), (0.0340, -0.0410), (0.0342, -0.0165), (0.0312, -0.0132), (0.0150, -0.0128),
                           (0.0138, -0.0112), (0, -0.0112)], at=(O.x, O.y, 0))
    assign(gb, cast)
    parts.append(gb)
    seam = lathe('gearbox_seam', [(0.0330, -0.0300), (0.0354, -0.0300), (0.0356, -0.0290), (0.0354, -0.0280), (0.0330, -0.0280)], at=(O.x, O.y, 0))
    assign(seam, cast)
    parts.append(seam)
    for k in range(5):
        a = math.radians(30 + k * 72)
        sx, sy = O.x + 0.0262 * math.cos(a), O.y + 0.0262 * math.sin(a)
        sh = lathe(f'gb_screw{k}', [(0, -0.0129), (0.0030, -0.0129), (0.0030, -0.0122), (0.0024, -0.0116), (0, -0.0115)], at=(sx, sy, 0))
        assign(sh, zinc)
        parts.append(sh)
    WY, WZ = O.y - 0.0205, -0.0300   # the worm's axis, along -x and tangent to the wheel
    tube = lathe('worm_tube', [(0, 0.004), (0.0125, 0.004), (0.0140, -0.002), (0.0140, -0.046), (0.0165, -0.049), (0.0165, -0.053), (0, -0.053)],
                 n=64, at=(O.x, WY, WZ), axis='X')
    assign(tube, cast)
    parts.append(tube)
    can = lathe('motor_can', [(0, -0.052), (0.0270, -0.052), (0.0275, -0.056), (0.0275, -0.0585), (0.0262, -0.060), (0.0262, -0.118),
                              (0.0240, -0.1235), (0.0150, -0.1265), (0.0085, -0.1270), (0.0080, -0.1305), (0.0050, -0.1315), (0, -0.1315)],
                n=96, at=(O.x, WY, WZ), axis='X')
    assign(can, paint)
    parts.append(can)
    for xk in (-0.074, -0.098):
        rib = lathe(f'can_rib{xk}', [(0.0258, xk - 0.0012), (0.0266, xk - 0.0006), (0.0266, xk + 0.0006), (0.0258, xk + 0.0012)], at=(O.x, WY, WZ), axis='X')
        assign(rib, paint)
        parts.append(rib)
    for k, ang in enumerate((25, 125, 265)):
        a = math.radians(ang)
        ex, ey = O.x + 0.050 * math.cos(a), O.y + 0.050 * math.sin(a)
        ear = plate(f'ear{k}', hull2d(circle(O.x + 0.022 * math.cos(a), O.y + 0.022 * math.sin(a), 0.0115, 40) + circle(ex, ey, 0.0098, 40)),
                    -0.0425, -0.0380, holes=[(ex, ey, 0.0052)])
        bevel(ear, 0.0006, 2)
        assign(ear, cast)
        gr = lathe(f'grommet{k}', [(0.0042, -0.0478), (0.0090, -0.0478), (0.0092, -0.0440), (0.0066, -0.0430), (0.0066, -0.0375), (0.0090, -0.0365),
                                   (0.0088, -0.0330), (0.0042, -0.0330)], at=(ex, ey, 0))
        assign(gr, rubber)
        sl = lathe(f'sleeve{k}', [(0.0030, -0.0484), (0.0050, -0.0484), (0.0050, -0.0324), (0.0030, -0.0324)], at=(ex, ey, 0))
        assign(sl, zinc)
        parts += [ear, gr, sl]
    con = box('connector', (O.x - 0.012, O.y - 0.0475, -0.0380), (O.x + 0.012, O.y - 0.0310, -0.0205))
    bevel(con, 0.0012, 3)
    assign(con, plastic)
    latch = box('latch', (O.x - 0.004, O.y - 0.0492, -0.0300), (O.x + 0.004, O.y - 0.0470, -0.0215))
    bevel(latch, 0.0006, 2)
    assign(latch, plastic)
    parts += [con, latch]
    models['crank motor'] = join('crank_motor', parts)

    # ---- the cast pivot tower round the spindle, bolted through two rubber mounted ears
    parts = []
    tw = lathe('tower', [(0.0068, 0.0050), (0.0125, 0.0050), (0.0135, 0.0065), (0.0128, 0.0120), (0.0112, 0.0270), (0.0100, 0.0300),
                         (0.0070, 0.0306)], at=(PV.x, PV.y, 0))
    assign(tw, cast)
    parts.append(tw)
    away = (PV - PR).normalized()
    side = Vector((-away.y, away.x, 0))
    for k, s in enumerate((1, -1)):
        e = PV + side * s * 0.034 + away * 0.006
        ear = plate(f'tower_ear{k}', hull2d(circle(PV.x, PV.y, 0.012, 40) + circle(e.x, e.y, 0.0085, 40)), 0.0090, 0.0135, holes=[(e.x, e.y, 0.0045)])
        bevel(ear, 0.0006, 2)
        assign(ear, cast)
        gr = lathe(f'tower_gr{k}', [(0.0036, 0.0060), (0.0078, 0.0060), (0.0080, 0.0088), (0.0060, 0.0094), (0.0060, 0.0132), (0.0080, 0.0138),
                                    (0.0078, 0.0165), (0.0036, 0.0165)], at=(e.x, e.y, 0))
        assign(gr, rubber)
        bolt = hexprism(f'tower_bolt{k}', 0.010, 0.0165, 0.0215, at=(e.x, e.y))
        bevel(bolt, 0.0003, 2)
        assign(bolt, zinc)
        parts += [ear, gr, bolt]
    spin = lathe('spindle', [(0, 0.0030), (0.0060, 0.0030), (0.0060, 0.0330), (0.0052, 0.0340), (0, 0.0340)], at=(PV.x, PV.y, 0))
    assign(spin, steel)
    parts.append(spin)
    models['wiper spindle'] = join('wiper_spindle', parts)

    # ---- wiper arm, built in its own frame (x along the arm from the spindle, z up from the arm's seat)
    parts = []
    head = plate('arm_head', hull2d(circle(0, 0, 0.0148, 64) + circle(0.058, 0, 0.0082, 40)), 0.0010, 0.0135)
    bevel(head, 0.0022, 4)
    assign(head, black)
    capb = lathe('arm_cap', [(0, 0.0128), (0.0128, 0.0128), (0.0128, 0.0150), (0.0110, 0.0172), (0.0060, 0.0184), (0, 0.0186)])
    bevel(capb, 0.0004, 2, harden=False)
    assign(capb, plastic)
    hinge = lathe('arm_hinge', [(0, -0.0115), (0.0026, -0.0115), (0.0026, 0.0115), (0, 0.0115)], at=(0.052, 0, 0.0085), axis='Y')
    assign(hinge, steel)

    def arm_sec(t):
        k = t
        a = 0.0068 * (1 - k) + 0.0046 * k
        b = 0.0038 * (1 - k) + 0.0026 * k
        zc = 0.0090 * (1 - k) + 0.0088 * k + 0.0016 * math.sin(math.pi * k)
        return a, b, zc, 0.92
    bar = loft('arm_bar', 0.392, arm_sec, n=28, steps=60)
    bar.location = (0.050, 0, 0)
    assign(bar, black)
    hook = loft('arm_hook', 0.026, lambda t: (0.0042, 0.0011, 0.0102 - 0.0016 * t, 0.9), n=20, steps=8)
    hook.location = (0.432, 0, 0)
    assign(hook, steel)
    arm = join('wiper_arm', [head, capb, hinge, bar, hook])
    arm.matrix_world = Matrix(mat_of('wiper arm').tolist())
    models['wiper arm'] = arm

    # ---- beam blade: rubber squeegee, spring steel beam under a plastic spoiler, centre adapter and end caps
    half = 0.2360

    def tapered(a0, b0, zc, sq, end=0.02):
        def f(t):
            x = abs(t - 0.5) * 2 * half
            k = max(0.0, min(1.0, (half - x) / end))
            s = 0.55 + 0.45 * math.sqrt(k)
            return a0 * s, b0 * (0.75 + 0.25 * s), zc, sq
        return f
    lip = loft('blade_lip', 2 * half, tapered(0.0009, 0.0022, 0.0022, 0.3), n=16, steps=40)
    base = loft('blade_base', 2 * half, tapered(0.0026, 0.0013, 0.0052, 0.6), n=16, steps=40)
    for o in (lip, base):
        o.location = (-half, 0, 0)
        assign(o, rubber)
    spo = loft('blade_spoiler', 2 * half, tapered(0.0044, 0.0028, 0.0090, 0.55), n=24, steps=60)
    fin = loft('blade_fin', 2 * half, tapered(0.0011, 0.0024, 0.0124, 0.2), n=16, steps=60)
    for o in (spo, fin):
        o.location = (-half, 0, 0)
        assign(o, plastic)
    fin.location.y = 0.0012
    adapter = box('blade_adapter', (-0.026, -0.0052, 0.0072), (0.026, 0.0052, 0.0170))
    bevel(adapter, 0.0018, 4)
    assign(adapter, black)
    button = box('blade_button', (-0.0080, -0.0038, 0.0165), (0.0080, 0.0038, 0.0176))
    bevel(button, 0.0010, 3)
    assign(button, plastic)
    caps = []
    for s in (-1, 1):
        c = box(f'blade_cap{s}', (s * half - 0.006, -0.0040, 0.0040), (s * half + 0.004, 0.0040, 0.0112))
        if s < 0:
            c.location.x = 0.002
        bevel(c, 0.0016, 3)
        assign(c, plastic)
        caps.append(c)
    blade = join('wiper_blade', [lip, base, spo, fin, adapter, button] + caps)
    blade.matrix_world = Matrix(mat_of('wiper blade').tolist())
    models['wiper blade'] = blade

    # ---- the glass: laminated, with the black ceramic frit band round its edge on the inside
    wi = SC.find('windshield')[0]
    wo = SC.objects[wi]
    c = SC.pose(wi, F)[0]
    hx, hy, hz = wo['size']
    x0, x1, y0, y1 = c[0] - hx, c[0] + hx, c[1] - hy + 0.096, c[1] + hy   # its lower edge stops just above the pivot

    def rrect(a0, a1, b0, b1, r):
        return (circle(a1 - r, b1 - r, r, 12, 0, math.pi / 2) + circle(a0 + r, b1 - r, r, 12, math.pi / 2, math.pi) +
                circle(a0 + r, b0 + r, r, 12, math.pi, 1.5 * math.pi) + circle(a1 - r, b0 + r, r, 12, 1.5 * math.pi, 2 * math.pi))
    g = plate('glass', rrect(x0, x1, y0, y1, 0.035), c[2] - hz, c[2] + hz)
    bevel(g, 0.0008, 2, harden=False)
    assign(g, glass)
    band = 0.024
    outer = plate('frit_o', rrect(x0 + 0.002, x1 - 0.002, y0 + 0.002, y1 - 0.002, 0.033), c[2] - hz - 0.00008, c[2] - hz - 0.00002)
    inner = plate('frit_i', rrect(x0 + band, x1 - band, y0 + band, y1 - band, 0.02), c[2] - hz - 0.001, c[2] - hz + 0.001)
    kit.cut(outer, inner)
    assign(outer, frit)
    models['windshield'] = join('windshield', [g, outer])
    return models


ANCHORS = {
    'crank': {'hub': [0, 0, 0.04], 'pin': [0, 0, -0.04]},
    'coupler': {'crank_end': [0, 0, 0.15], 'rocker_end': [0, 0, -0.15]},
    'rocker': {'pivot': [0, 0, 0.055], 'coupler_end': [0, 0, -0.055]},
}
TINT = {'crank': ['paint'], 'crank motor': ['paint']}
FILES = {'crank': 'crank', 'coupler': 'coupler', 'rocker': 'rocker', 'coupler pin': 'ball-stud', 'rocker pin': 'spindle-boss',
         'crank shaft': 'output-shaft', 'crank motor': 'motor', 'wiper spindle': 'pivot-tower', 'wiper arm': 'arm',
         'wiper blade': 'blade', 'windshield': 'glass'}


def main():
    models = build()
    if os.environ.get('WOBBL_PREVIEW'):
        return models
    entries = []
    for name, ob in models.items():
        kit.finish(ob, 1.0, ao=512 if name in ('crank motor', 'wiper arm', 'wiper blade', 'wiper spindle') else 256, ao_distance=0.02)
        kit.to_local(ob, mat_of(name))
        rel = f'{EX}/{FILES[name]}.glb'
        kit.export(ob, rel)
        e = {'object': name, 'file': rel}
        if name in TINT:
            e['tint'] = TINT[name]
        if name in ANCHORS:
            e['anchors'] = ANCHORS[name]
        entries.append(e)
    kit.write_manifest(EX, {'replace': entries})


if __name__ == '__main__':
    main()
