"""Context models for the props the other example runs set their parts among: hex bolts with washer faces and rolled
threads, drilled and milled mounting plates, a NEMA stepper with its D shaft and GT2 pulley, a painted wall with
an acrylic shelf and a boxed load, a slotted cast iron weight, ground end platens, a textured PEI print bed, a
power resistor, benches and a breadboard, a Bunsen burner, a glass beaker, a falling rock and a concrete floor.

Each builder makes its prop in the scene object's own frame (a box centred on the origin with the scene's half
sizes, a cylinder or hex standing along z), so one file serves every object of that name. Runs drawn by the
strength stage (their FE mesh in mm) take the props as "add" models placed in that frame instead.

    blender -b --factory-startup -P site/blender/props.py [-- example ...]
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import bpy  # noqa: E402
import kit  # noqa: E402
import numpy as np  # noqa: E402
import wobscene  # noqa: E402
from kit import assign, bevel, box, circle, cut, hexprism, hull2d, join, lathe, plate, sweep  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

MAT = {}


def mat(name):
    """One shared material per name in a build (each model gets its own copy when it is finished)."""
    if name not in MAT:
        MAT[name] = kit.mats(kit.shapes.UNIT)[name]()
    return MAT[name]


def size_of(o):
    return o['size']


# ---------------------------------------------------------------- builders, each in the object's own frame

def hex_head(o, sc, oi):
    """A hex bolt head standing on a washer face, chamfered at 30 degrees, with grade marks on top."""
    rc, hh = size_of(o)[:2]
    af = rc * math.sqrt(3)
    h = hexprism('head', af, -hh + 0.08 * hh, hh)
    bevel(h, 0.03 * af, 2)
    wf = lathe('washer_face', [(0, -hh), (af * 0.48, -hh), (af * 0.48, -hh + 0.1 * hh), (0, -hh + 0.1 * hh)])
    parts = [h, wf]
    for k in range(3):    # the three radial marks of a grade 8.8 head, raised a hair
        a = math.pi / 2 + k * 2 * math.pi / 3
        m = box('mark', (-0.03 * af, af * 0.12, hh - 0.01), (0.03 * af, af * 0.3, hh + 0.025 * af))
        m.matrix_world = Matrix.Rotation(a, 4, 'Z')
        parts.append(m)
    b = join('bolt_head', parts)
    assign(b, mat('steel_zinc'))
    return b


def bolt_shank(o, sc, oi):
    """The shank: plain under the head, then rolled threads at the scene's pitch for its size."""
    r, hl = size_of(o)[:2]
    pitch = max(0.25, 0.45 * r)
    prof = [(0, -hl), (r * 0.82, -hl), (r * 0.9, -hl + pitch * 0.5)]
    z = -hl + pitch
    while z < hl * 0.2:
        prof += [(r * 0.84, z - pitch * 0.25), (r, z), (r * 0.84, z + pitch * 0.25)]
        z += pitch
    prof += [(r, hl * 0.25), (r, hl), (0, hl)]
    b = lathe('shank', prof, n=20, smooth=False)
    kit.smooth_by_angle(b, 60)
    assign(b, mat('steel_zinc'))
    return b


def plate_milled(o, sc, oi, holes_from=('bolt', 'M8 8.8'), material='aluminum_machined'):
    """A machined plate: the scene's box with chamfered edges and drilled holes wherever its bolts pass."""
    hx, hy, hz = size_of(o)
    M = sc.matrix(oi, sc.home())
    Mi = np.linalg.inv(M)
    holes = []
    for j, b in enumerate(sc.objects):
        if b.get('name') in holes_from and b['kind'] == 'body':
            p = wobscene.world(Mi, sc.pose(j)[0])
            if abs(p[0]) < hx and abs(p[1]) < hy:
                holes.append((p[0], p[1], b['size'][0] * 1.08))
    rr = 0.04 * min(hx, hy)
    out = (circle(hx - rr, hy - rr, rr, 8, 0, math.pi / 2) + circle(-hx + rr, hy - rr, rr, 8, math.pi / 2, math.pi) +
           circle(-hx + rr, -hy + rr, rr, 8, math.pi, 1.5 * math.pi) + circle(hx - rr, -hy + rr, rr, 8, 1.5 * math.pi, 2 * math.pi))
    p = plate('plate', out, -hz, hz, holes=holes)
    bevel(p, min(hz * 0.18, 0.004 * max(hx, hy)), 2)
    assign(p, mat(material))
    return p


def plate_steel(o, sc, oi):
    return plate_milled(o, sc, oi, material='steel_mill')


def nema_motor(o, sc, oi):
    """A NEMA stepper body: laminated stack between black end caps, chamfered corners, screws and a label."""
    hx, hy, hz = size_of(o)
    c = hx * 0.16

    def oct_(s):
        return [(s - c, -s), (s, -s + c), (s, s - c), (s - c, s), (-s + c, s), (-s, s - c), (-s, -s + c), (-s + c, -s)]
    z0 = -hz + 2 * hz * 0.14             # the front cap is the flange model's
    z1 = hz - 2 * hz * 0.22
    stack = plate('stack', oct_(hx * 0.985), z0, z1)
    bevel(stack, hx * 0.01, 1)
    assign(stack, mat('steel_stainless'))
    grooves = []
    for k in (1, 11):     # the stack's end laminations, a shade proud of it
        z = z0 + (z1 - z0) * k / 12
        g = plate('lam', oct_(hx * 0.992), z - 0.012 * hz, z + 0.012 * hz)
        assign(g, mat('steel_dark'))
        grooves.append(g)
    rear = plate('rear', oct_(hx), z1, hz)
    bevel(rear, hx * 0.03, 3)
    assign(rear, mat('aluminum_anodized'))
    parts = [stack, rear] + grooves
    for sx in (-1, 1):
        for sy in (-1, 1):
            s = lathe('screw', [(0, hz), (hx * 0.13, hz), (hx * 0.13, hz + hx * 0.07), (hx * 0.1, hz + hx * 0.1), (0, hz + hx * 0.1)],
                      at=(sx * hx * 0.705, sy * hx * 0.705, 0))
            assign(s, mat('steel_black_oxide'))
            parts.append(s)
    lab = box('label', (-hx * 0.62, hx * 0.985, z0 + (z1 - z0) * 0.12), (hx * 0.62, hx * 0.995, z1 - (z1 - z0) * 0.12))
    assign(lab, mat('label'))
    con = box('connector', (-hx * 0.34, -hx * 1.18, z1 + (hz - z1) * 0.1), (hx * 0.34, -hx * 0.99, hz - (hz - z1) * 0.1))
    bevel(con, hx * 0.02, 2)
    assign(con, mat('plastic_white'))
    parts += [lab, con]
    return join('motor', parts)


def nema_front(o, sc, oi):
    """The front cap: black anodised with its centring boss and four tapped mounting holes."""
    hx, hy, hz = size_of(o)
    c = hx * 0.16
    out = [(hx - c, -hx), (hx, -hx + c), (hx, hx - c), (hx - c, hx), (-hx + c, hx), (-hx, hx - c), (-hx, -hx + c), (-hx + c, -hx)]
    cap = plate('front', out, -hz, hz, holes=[(sx * hx * 0.705, sy * hx * 0.705, hx * 0.075) for sx in (-1, 1) for sy in (-1, 1)])
    bevel(cap, hx * 0.03, 3)
    boss = lathe('boss', [(hx * 0.12, -hz), (hx * 0.5, -hz), (hx * 0.5, -hz - hx * 0.09), (hx * 0.47, -hz - hx * 0.11), (hx * 0.12, -hz - hx * 0.11)])
    b = join('flange', [cap, boss])
    assign(b, mat('aluminum_anodized'))
    return b


def d_shaft(o, sc, oi):
    """A ground steel shaft with the D flat along its outer end."""
    r, hl = size_of(o)[:2]
    s = lathe('shaft', [(0, -hl), (r * 0.9, -hl), (r, -hl + r * 0.1), (r, hl - r * 0.1), (r * 0.9, hl), (0, hl)], n=48)
    flat = box('flat', (r * 0.8, -r * 2, -hl - 1), (r * 2, r * 2, hl * 0.2))
    cut(s, flat)
    kit.smooth_by_angle(s, 30)
    assign(s, mat('steel_bare'))
    return s


def gt2_pulley(o, sc, oi):
    """An aluminium GT2 timing pulley: twenty teeth between two flanges, and a hub with two set screws."""
    R, hl = size_of(o)[:2]
    w = 2 * hl
    fl = R
    core = R * 0.74
    hub_l = w * 0.38
    zt0, zt1 = -hl + hub_l + w * 0.06, hl - w * 0.06
    teeth = []
    n = 20
    for i in range(n * 4):
        a = 2 * math.pi * i / (n * 4)
        rr = core if (i % 4) < 2 else core * 0.9
        teeth.append((rr * math.cos(a), rr * math.sin(a)))
    t = plate('teeth', teeth, zt0, zt1)
    f1 = lathe('flange1', [(core * 0.9, zt0 - w * 0.06), (fl, zt0 - w * 0.06), (fl, zt0), (core * 0.9, zt0)])
    f2 = lathe('flange2', [(core * 0.9, zt1), (fl, zt1), (fl, zt1 + w * 0.06), (core * 0.9, zt1 + w * 0.06)])
    hub = lathe('hub', [(R * 0.32, -hl), (R * 0.7, -hl), (R * 0.72, -hl + 0.05 * w), (R * 0.72, zt0 - w * 0.06), (R * 0.32, zt0 - w * 0.06)])
    bore = lathe('bore', [(0, -hl - 1), (R * 0.32, -hl - 1), (R * 0.32, hl + 1), (0, hl + 1)])
    body = join('pulley', [t, f1, f2, hub])
    cut(body, bore)
    bevel(body, R * 0.02, 1, angle=40)
    assign(body, mat('aluminum_machined'))
    parts = [body]
    for a in (0, math.pi / 2):
        ss = lathe('set_screw', [(0, 0), (R * 0.12, 0), (R * 0.12, R * 0.05), (0, R * 0.05)])
        ss.matrix_world = Matrix.Translation((R * 0.715 * math.cos(a), R * 0.715 * math.sin(a), (-hl + zt0) / 2)) @ \
            kit.frame_matrix((0, 0, 0), (0, 0, 1), None, (math.cos(a), math.sin(a), 0))
        assign(ss, mat('steel_black_oxide'))
        parts.append(ss)
    return join('pulley', parts)


def wall(o, sc, oi):
    """A painted plaster wall: the scene's slab with a soft edge and a fine roller texture."""
    hx, hy, hz = size_of(o)
    w = box('wall', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(w, hz * 0.15, 2)
    assign(w, mat('paint_wall'))
    return w


def acrylic_shelf(o, sc, oi):
    """A clear acrylic shelf, so the bracket under it stays in view, with polished edges."""
    hx, hy, hz = size_of(o)
    s = box('shelf', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(s, hz * 0.2, 3)
    assign(s, mat('acrylic'))
    return s


def carton(o, sc, oi):
    """A taped cardboard carton for a load on a shelf."""
    hx, hy, hz = size_of(o)
    b = box('carton', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(b, min(hx, hy, hz) * 0.04, 2)
    assign(b, mat('cardboard'))
    t = box('tape', (-hx * 0.18, -hy - 0.01 * hy, hz - 0.0005 * hz), (hx * 0.18, hy + 0.01 * hy, hz + 0.006 * hz))
    assign(t, mat('tape'))
    seam = box('seam', (-0.004 * hx, -hy * 1.001, -hz * 0.99), (0.004 * hx, -hy * 0.99, hz * 0.99))
    assign(seam, mat('decal'))
    return join('carton', [b, t, seam])


def slotted_weight(o, sc, oi):
    """A cast iron slotted calibration weight hung on the hook, painted black with a machined top."""
    hx, hy, hz = size_of(o)
    b = box('weight', (-hx, -hy, -hz), (hx, hy, hz * 0.82))
    bevel(b, min(hx, hy) * 0.12, 3)
    assign(b, mat('cast_iron'))
    top = lathe('eye', [(0, hz * 0.82), (hx * 0.55, hz * 0.82), (hx * 0.5, hz * 0.88), (hx * 0.28, hz * 0.9), (hx * 0.24, hz), (0, hz)])
    assign(top, mat('steel_bare'))
    return join('weight', [b, top])


def platen(o, sc, oi):
    """A ground steel end platen with a recess the strut seats in."""
    hx, hy, hz = size_of(o)
    p = box('platen', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(p, hz * 0.12, 3)
    assign(p, mat('steel_bare'))
    return p


def print_bed(o, sc, oi):
    """A textured PEI spring steel sheet on its aluminium bed, with the sheet's tab at the front."""
    hx, hy, hz = size_of(o)
    bed = box('bed', (-hx, -hy, -hz), (hx, hy, hz - 0.3))
    bevel(bed, 0.6, 2)
    assign(bed, mat('aluminum_machined'))
    sheet = box('pei', (-hx + 2, -hy + 2, hz - 0.3), (hx - 2, hy - 2, hz))
    bevel(sheet, 0.2, 1)
    assign(sheet, mat('pei'))
    tab = plate('tab', hull2d(circle(0, -hy - 4, 8, 24) + [(-14, -hy + 2), (14, -hy + 2)]), hz - 0.3, hz)
    assign(tab, mat('pei'))
    return join('print_bed', [bed, sheet, tab])


def power_resistor(o, sc, oi):
    """The heat source as an aluminium clad power resistor screwed under the sink."""
    hx, hy, hz = size_of(o)
    b = box('clad', (-hx, -hy * 0.7, -hz), (hx, hy * 0.7, hz))
    bevel(b, hz * 0.2, 2)
    fins = []
    for s in (-1, 1):
        f = box('foot', (-hx, s * hy * 0.7 - (0 if s > 0 else hy * 0.3), -hz), (hx, s * hy * 0.7 + (hy * 0.3 if s > 0 else 0), -hz * 0.2))
        bevel(f, hz * 0.1, 1)
        fins.append(f)
    body = join('resistor', [b] + fins)
    assign(body, mat('aluminum_anodized_gold'))
    leads = []
    for s in (-1, 1):
        ld = box('lead', (s * hx, -hy * 0.12, -hz * 0.3), (s * (hx + hx * 0.35), hy * 0.12, hz * 0.1))
        assign(ld, mat('steel_zinc'))
        leads.append(ld)
    return join('resistor', [body] + leads)


def desk(o, sc, oi):
    """A wooden desk or bench top with rounded edges."""
    hx, hy, hz = size_of(o)
    d = box('desk', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(d, hz * 0.25, 3)
    assign(d, mat('wood'))
    return d


def breadboard(o, sc, oi):
    """A white solderless breadboard body with its rounded edges and a slightly glossy top."""
    hx, hy, hz = size_of(o)
    b = box('breadboard', (-hx, -hy, -hz), (hx, hy, hz))
    bevel(b, min(hz * 0.3, hx * 0.01), 3)
    assign(b, mat('plastic_white'))
    return b


def burner(o, sc, oi):
    """A Bunsen burner on the scene's barrel: cast iron base, brass barrel with an air collar and slots."""
    base = lathe('base', [(0, 0), (16, 0), (16, 1.2), (15.2, 2.5), (11, 5.4), (6.6, 6.2), (6.4, 8), (0, 8)], n=64)
    bevel(base, 0.3, 2)
    assign(base, mat('cast_iron'))
    barrel = lathe('barrel', [(4.5, 8), (6, 8), (6, 88.6), (5.6, 90), (4.5, 90)], n=48)
    assign(barrel, mat('brass'))
    collar = lathe('collar', [(6.0, 13), (7.2, 13), (7.6, 13.6), (7.6, 22.4), (7.2, 23), (6.0, 23)], n=48)
    slots = []
    for a in (0, math.pi):
        s = box('slot', (-2.2, -1.2, 15.5), (2.2, 1.2, 20.5))
        s.matrix_world = Matrix.Rotation(a + math.pi / 2, 4, 'Z') @ Matrix.Translation((0, 7.0, 0))
        slots.append(s)
    for s in slots:
        cut(collar, s)
    assign(collar, mat('brass'))
    return join('burner', [base, barrel, collar])


def hose_barb(o, sc, oi):
    """The gas inlet: a brass hose barb with three ridges."""
    r, hl = size_of(o)[:2]
    prof = [(0, -hl), (r * 0.8, -hl)]
    for k in range(3):
        z = -hl + (k + 0.3) * (2 * hl / 3.2)
        prof += [(r * 0.75, z), (r, z + 2 * hl / 6.4), (r * 0.75, z + 2 * hl / 6.4 + 0.01)]
    prof += [(r * 0.8, hl), (0, hl)]
    b = lathe('barb', prof, n=32, smooth=False)
    kit.smooth_by_angle(b, 50)
    assign(b, mat('brass'))
    return b


def beaker(o, sc, oi):
    """A borosilicate beaker on the scene's own outline, with a rolled rim, a pouring spout and white graduations."""
    P, T = sc.mesh(oi)
    r_out = float(np.hypot(P[:, 0], P[:, 1]).max())
    h = float(P[:, 2].max())
    t = r_out * 0.03
    prof = [(0, 0), (r_out - t * 2, 0), (r_out, t * 2), (r_out, h - t), (r_out + t * 0.6, h - t * 0.4), (r_out + t * 0.6, h), (r_out - t, h),
            (r_out - t, t * 3), (r_out - t * 3, t), (0, t)]
    g = lathe('beaker', prof, n=96)
    assign(g, mat('glass'))
    parts = [g]
    for k in range(1, 8):
        z = h * (0.12 + 0.1 * k)
        w = 0.035 if k % 2 else 0.06
        mk = kit.band('grad', r_out + 0.0015 * r_out, z - 0.004 * h, z + 0.004 * h, -w, w, n=4)
        assign(mk, mat('enamel_white'))
        parts.append(mk)
    return join('beaker', parts)


def rock(o, sc, oi):
    """The falling rock: its own mesh smoothed, broken up with layered noise and given a scanned rock surface."""
    P, T = sc.mesh(oi)
    me = bpy.data.meshes.new('rock')
    me.from_pydata(P.tolist(), [], T.tolist())
    me.update()
    ob = bpy.data.objects.new('rock', me)
    bpy.context.scene.collection.objects.link(ob)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(me)
    bm.free()
    kit.subdivide(ob, 2)
    size = float(np.abs(P).max())
    rng = np.random.default_rng(7)
    cen = [rng.normal(size=3) for _ in range(9)]
    for v in ob.data.vertices:
        p = np.array(v.co)
        n = np.array(v.normal)
        d = sum(0.07 * math.sin(np.dot(p / size, c) * (3 + 2 * i)) for i, c in enumerate(cen))
        v.co = Vector(p + n * d * size * 0.35)
    ob.data.update()
    kit.smooth_by_angle(ob, 70)
    assign(ob, mat('rock'))
    return ob


def floor(sc, center, half, z):
    """A polished concrete floor slab for drop tests, its top at the scene's ground."""
    f = box('floor', (center[0] - half, center[1] - half, z - half * 0.12), (center[0] + half, center[1] + half, z))
    bevel(f, half * 0.01, 3, harden=False)
    assign(f, mat('concrete_polished'))
    return f


# ---------------------------------------------------------------- which props each example gets
# key: the example key the runtime uses (the run folder without its trailing output part)
STRENGTH = {'motor-mount', 'cast-motor-mount', 'cast-mount-credibility', 'shelf-bracket', 'drone-arm', 'heat-sink', 'enclosure', 'battery-pack'}
BOLTS = {'bolt': bolt_shank, 'bolt head': hex_head}
MOTOR = {'motor': nema_motor, 'motor flange': nema_front, 'motor shaft': d_shaft, 'pulley': gt2_pulley}
# motor-mount, cast-motor-mount, shelf-bracket, drone-arm, bolted-flange, bracket-crack, steel-hook, strut-buckling
# and printed-lid are built in CAD now (site/cad), and so is the desktop arm, whose desk this script still makes
EXAMPLES = {
    'cast-mount-credibility': ('cast-mount-credibility/output', 'mm', {'plate': plate_steel, **MOTOR}),
    'heat-sink': ('heat-sink/output', 'mm', {'heat source': power_resistor}),
    'desktop-arm': ('desktop-arm/output', 'm', {'desk': desk}),
    'circuits/buck': ('circuits/buck', 'mm', {'bench': desk, 'breadboard': breadboard}),
    'circuits/kicad-board': ('circuits/kicad-board', 'mm', {'bench': desk, 'breadboard': breadboard}),
    'circuits/led': ('circuits/led', 'mm', {'bench': desk, 'breadboard': breadboard}),
    'circuits/mosfet': ('circuits/mosfet', 'mm', {'bench': desk, 'breadboard': breadboard}),
    'circuits/opamp-cable': ('circuits/opamp-cable', 'mm', {'bench': desk, 'breadboard': breadboard}),
    'propane-flame': ('propane-flame/output', 'mm', {'burner': burner, 'air inlet': hose_barb}),
    'baking-soda-vinegar': ('baking-soda-vinegar/output', 'A', {'beaker': beaker}),
    'falling-rock': ('falling-rock/output', 'm', {'rock': rock}),
    'handheld-drop': ('handheld-drop/output', 'mm', {}),
}
UNITS = {'mm': 1000.0, 'm': 1.0, 'A': 1000.0}
CAD_ENTRY = {'desktop-arm'}     # examples whose manifest entry site/cad writes, round the files made here


def build(key):
    rel, unit, table = EXAMPLES[key]
    kit.reset(UNITS[unit])
    MAT.clear()
    sc = wobscene.Scene(os.path.join(kit.REPO, 'examples', rel, 'scene.wobbl'))
    models = {}           # object name -> (model in the object's frame, index of the object it was built for)
    for name, fn in table.items():
        ids = sc.find(name)
        if not ids:
            continue
        models[name] = (fn(sc.objects[ids[0]], sc, ids[0]), ids[0])
    extra = {}
    if key == 'handheld-drop':
        g = next(o for o in sc.objects if o['kind'] == 'ground')
        extra['floor'] = floor(sc, g['center'], g['size'] / 2, g['z'])
    return sc, models, extra


def place_all(sc, name, ob):
    """Copies of a model at every object of its name, joined, in the scene's world (for strength runs)."""
    ids = sc.find(name)
    copies = []
    for i in ids:
        c = ob.copy()
        c.data = ob.data.copy()
        bpy.context.scene.collection.objects.link(c)
        c.matrix_world = Matrix(sc.matrix(i, sc.home()).tolist())
        copies.append(c)
    bpy.data.objects.remove(ob, do_unlink=True)
    return join(name.replace(' ', '_'), copies)


def slug(name):
    return name.lower().replace(' ', '-').replace('.', '')


def main():
    global SC
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else list(EXAMPLES)
    if os.environ.get('PROPS'):
        args = os.environ['PROPS'].split(',')
    preview = os.environ.get('WOBBL_PREVIEW')
    for key in args:
        sc, models, extra = build(key)
        unit = UNITS[EXAMPLES[key][1]]
        replace, add = [], []
        if key in STRENGTH or preview:
            objs = {name: place_all(sc, name, ob) for name, (ob, i) in models.items()}
        else:
            objs = {name: ob for name, (ob, i) in models.items()}
        objs.update(extra)
        if preview:
            SC = sc
            return objs
        folder = key.replace('/', '-')
        for name, ob in objs.items():
            kit.finish(ob, unit, ao=256, ao_distance=0.03 * unit)
            rel = f'{folder}/{slug(name)}.glb'
            kit.export(ob, rel)
            if key in STRENGTH or name in extra:
                # covers: the scene objects this model stands for, which a still or a scene view leaves out
                add.append({'file': rel, 'covers': [name]} if name in models else {'file': rel})
            else:
                replace.append({'object': name, 'file': rel})
        entry = {}
        if replace:
            entry['replace'] = replace
        if add:
            entry['add'] = add
        if key == 'handheld-drop':
            entry['hide'] = ['ground']
        if key not in CAD_ENTRY:
            kit.write_manifest(key, entry)


if __name__ == '__main__':
    main()
