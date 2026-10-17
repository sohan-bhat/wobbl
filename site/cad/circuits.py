"""Context models for the circuit bench runs (examples/circuits): a full size solderless breadboard with its clip rows
under every hole, a bench supply with its case, front panel, knobs and five way binding posts, and the leads and
cases of the parts that carry no live reading. Parts that show a live reading (a resistor's temperature, an LED's
glow, a display) keep the scene's own shapes and colors.

    python site/cad/circuits.py [run ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import electronics as el  # noqa: E402
from gltf import Node, export, write_manifest  # noqa: E402
from scene import Scene  # noqa: E402

EXAMPLES = os.path.join(HERE, '..', '..', 'examples', 'circuits')
RUNS = ['led', 'buck', 'mosfet', 'opamp-cable', 'kicad-board']


def local(sc, i, pts):
    """World points into object i's frame."""
    Mi = sc.inverse(i)
    P = np.c_[np.asarray(pts, dtype=np.float64), np.ones(len(pts))]
    return (P @ Mi.T)[:, :3]


def breadboard(sc, run):
    i, o, M = sc.named('breadboard')[0]
    hx, hy, hz = o['size']
    h = sc.named('holes')[0][0]
    P = local(sc, i, sc.arr(sc.objects[h]['pos']).reshape(-1, 3))
    cols = sorted(set(np.round(P[:, 0], 3)))
    rows = sorted(set(np.round(P[:, 1], 3)))
    ch = local(sc, i, [sc.named('channel')[0][2][:3, 3]])[0]
    stripes = []
    for j, so, SM in sc.named('rail stripe'):
        y = local(sc, i, [SM[:3, 3]])[0][1]
        stripes.append((y, 'red' if so['color'].lower().startswith('#d') else 'blue'))
    # the power rails are the two rows nearest each long edge
    rails = rows[:2] + rows[-2:]
    terminal = [y for y in rows if y not in rails]
    top_row, bottom_row = max(terminal), min(terminal)
    number_y = [top_row + 1.3, bottom_row - 1.3]
    upper = sorted([y for y in terminal if y > ch[1]], reverse=True)
    lower = sorted([y for y in terminal if y < ch[1]], reverse=True)
    names = [(y, c) for y, c in zip(upper, 'jihgf')] + [(y, c) for y, c in zip(lower, 'edcba')]
    letters = [(cols[0] - 2.2, names), (cols[-1] + 2.2, names)]
    parts = el.breadboard(hx, hy, hz, cols, rows, ch[1], rails, stripes, number_y, letters, funnel=len(cols) < 40)
    export(f'circuits-{run}/breadboard.glb', [Node('breadboard', parts)], tol=0.02, ang=0.3)


def supplies(sc):
    """The names of the bench supplies in a run: every object with a panel, display and posts of its own."""
    return [o['name'] for o in sc.objects if o.get('name') and sc.find(o['name'] + ' panel') and sc.find(o['name'] + ' post')]


def supply(sc):
    """One bench supply, shared by every run (they all stand at the same height on the same bench)."""
    name = supplies(sc)[0]
    i, o, M = sc.named(name)[0]
    hx, hy, hz = o['size']
    _, bo, BM = sc.named('bench')[0]
    bench_top = BM[2, 3] + bo['size'][2]
    riser = (M[2, 3] - hz) - bench_top
    back = BM[1, 3] + bo['size'][1]
    cord = (hx * 0.2, back - M[1, 3] - 1.5, bench_top - M[2, 3] + 0.9)
    export('circuits-bench/supply.glb', [Node('supply', el.supply_case(hx, hy, hz, riser, cord))], tol=0.01, ang=0.25)
    pi, po, PM = sc.named(name + ' panel')[0]
    phx, phy, phz = po['size']
    di, do, DM = sc.named(name + ' display')[0]
    d = local(sc, pi, [DM[:3, 3]])[0]
    display = (d[0], d[2], do['size'][0], do['size'][2])
    posts = sorted((p[0], p[2]) for p in local(sc, pi, [m[:3, 3] for _, _, m in sc.named(name + ' post')]))
    z = posts[0][1]
    knobs = [(-phx + 5.8, z), (phx - 5.8, z)]
    export('circuits-bench/supply-panel.glb', [Node('supply_panel', el.supply_panel(phx, phy, phz, display, posts, knobs))],
           tol=0.01, ang=0.3)
    _, qo, _ = sc.named(name + ' post')[0]
    export('circuits-bench/binding-post.glb', [Node('binding_post', el.binding_post(*qo["size"]))], tol=0.008, ang=0.3)


def supply_entries(sc):
    out = []
    for name in supplies(sc):
        out += [{'object': name, 'file': 'circuits-bench/supply.glb', 'tint': ['paint']},
                {'object': name + ' panel', 'file': 'circuits-bench/supply-panel.glb'},
                {'object': name + ' post', 'file': 'circuits-bench/binding-post.glb', 'tint': ['post_plastic']}]
    return out


BUILT = set()


def once(rel, build, **kw):
    """Export a shared file the first time a run asks for it."""
    if rel not in BUILT:
        export(rel, build(), **kw)
        BUILT.add(rel)
    return rel


def parts(sc):
    """The parts on the board that carry no live reading: their models, as (replace, hide, add) manifest lists."""
    rep, hide, add = [], [], []
    names = {o.get('name') for o in sc.objects}
    for name in sorted(n for n in names if n):
        objs = sc.named(name)
        i, o, M = objs[0]
        # a radial electrolytic's vented top
        if name.endswith(' top') and o.get('shape') == 'cylinder':
            r, hh = o['size']
            f = once(f'circuits-bench/cap-top-{r:.2f}.glb', lambda: [Node('cap_top', el.cap_top(r, hh, '+' if r < 3.2 else 'K'))],
                     tol=0.005, ang=0.25)
            rep.append({'object': name, 'file': f})
        # a toroid inductor: its core, and the winding in place of the see through coil
        if o.get('shape') == 'torus' and (name + ' coil') in names:
            R, r = o['size']
            ci, co, CM = sc.named(name + ' coil')[0]
            leads = [local(sc, ci, [m[:3, 3] + m[:3, 2] * lo['size'][1]])[0] for _, lo, m in sc.named(name + ' lead')]
            wire = (co['size'][1] - r) * 1.6
            f = once('circuits-bench/toroid-core.glb', lambda: [Node('toroid_core', el.toroid_core(R, r))], tol=0.02, ang=0.35)
            g = once('circuits-bench/toroid-winding.glb',
                     lambda: [Node('toroid_winding', el.toroid_winding(R, r, wire, 17, gap_at=-90, gap=70, leads=leads))],
                     tol=0.01, ang=0.35)
            rep += [{'object': name, 'file': f}, {'object': name + ' coil', 'file': g}]
        # a DIP's stamped leads in place of the scene's two capsules per pin
        if (name + ' pin') in names and (name + ' notch') in names:
            pins = [local(sc, i, [m[:3, 3]])[0] for _, po, m in sc.named(name + ' pin') if abs(m[2, 2]) > 0.99]
            rows = {round(float(p[1]), 2) for p in pins}
            if len(rows) == 2:
                board = local(sc, i, [[0, 0, 0]])[0][2]
                hx, hy, hz = o['size']
                rel = f'circuits-bench/dip{len(pins)}-leads.glb'
                f = once(rel, lambda: [Node('dip_leads', el.dip_leads([(p[0], p[1]) for p in pins], board, hy, -hz))],
                         tol=0.01, ang=0.3)
                hide.append(name + ' pin')
                add.append({'file': f, 'parent': name})
        # a TO-220: the plated tab with its hole and notches, and flat leads
        if (name + ' tab') in names and (name + ' tab hole') in names:
            _, to, _ = sc.named(name + ' tab')[0]
            _, ho, _ = sc.named(name + ' tab hole')[0]
            f = once('circuits-bench/to220-tab.glb', lambda: [Node('to220_tab', el.to220_tab(*to['size'], ho['size'][0]))],
                     tol=0.01, ang=0.25)
            rep.append({'object': name + ' tab', 'file': f})
            hide.append(name + ' tab hole')
            if (name + ' lead') in names:
                _, lo, _ = sc.named(name + ' lead')[0]
                g = once('circuits-bench/to220-lead.glb', lambda: [Node('to220_lead', el.to220_lead(*lo['size']))], tol=0.01)
                rep.append({'object': name + ' lead', 'file': g})
        # an axial rectifier's moulded body (its band stays the scene's)
        if (name + ' band') in names and o.get('shape') == 'cylinder' and not o.get('field') and o['size'][0] < 1.5:
            r, hh = o['size']
            f = once(f'circuits-bench/do41-{hh:.2f}.glb', lambda: [Node('do41', el.do41(r, hh))], tol=0.005, ang=0.25)
            rep.append({'object': name, 'file': f})
        # an LED's leads, splayed into the epoxy, the anode crimped
        if (name + ' dome') in names and (name + ' lead') in names:
            c = M[:3, 3]
            for k, (j, lo, m) in enumerate(sorted(sc.named(name + ' lead'), key=lambda t: -t[2][1, 3])):
                tip = c + (m[:3, 3] - c) * np.array([1, 1, 0])
                tip = c + (tip - c) / (np.linalg.norm(tip - c) or 1) * 1.27
                v = local(sc, j, [tip])[0]
                kind = 'anode' if k == 0 else 'cathode'
                f = once(f'circuits-bench/led-{kind}.glb', lambda: [Node(f'led_{kind}', el.led_lead(lo['size'][1], (v[0], v[1]), k == 0))],
                         tol=0.01)
                rep.append({'object': name + ' lead', 'index': j, 'file': f})
    return rep, hide, add


def main(runs):
    first = True
    for run in runs:
        sc = Scene(os.path.join(EXAMPLES, run, 'scene.wobbl'))
        if first:
            supply(sc)
            first = False
        breadboard(sc, run)
        rep, hide, add = parts(sc)
        write_manifest(f'circuits/{run}', {
            'replace': [
                {'object': 'bench', 'file': f'circuits-{run}/bench.glb'},
                {'object': 'breadboard', 'file': f'circuits-{run}/breadboard.glb'},
            ] + supply_entries(sc) + rep,
            'hide': ['channel', 'rail stripe'] + hide,
            **({'add': add} if add else {}),
        })


if __name__ == '__main__':
    main(sys.argv[1:] or RUNS)
