"""Chemistry runs as 3D scenes for the viewer, written through wobbl.viz.Scene from the run's own playback.json.

A molecule moves as its dynamics run did, or swings through its strongest IR mode. A crystal shows its unit cell
repeated, breathing through the volumes of its equation of state with the energy of each, or its atoms moving in a
dynamics run. A reaction shows its reactants and products with the Gibbs energy and equilibrium constant ticking
through temperature, and a reaction path shows the atoms moving over the barrier. Lengths are angstrom.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write(out: str | Path) -> Path | None:
    """Build out/scene.wobbl from out/playback.json and out/outcome.json, or None when the run has nothing to draw.
    Never raises, since a view that fails must not lose the result."""
    out = Path(out)
    try:
        pb = json.loads((out / "playback.json").read_text())
        o = json.loads((out / "outcome.json").read_text()) if (out / "outcome.json").exists() else {}
        kind = pb.get("manifest", {}).get("kind")
        fn = {"molecule": _molecule, "crystal": _crystal, "reaction": _reaction, "mixture": _mixture}.get(kind)
        if fn is None:
            return None
        s = fn(pb, o)
        if s is None:
            return None
        verdict = (o.get("verdict") or "ok").upper()
        s.verdict(verdict, o.get("headline", "")[:300], "pass" if verdict == "OK" else "warn")
        return s.save(out)
    except Exception:  # noqa: BLE001
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return None


def _scene(title: str, o: dict, source: str):
    from ..viz import Scene
    return Scene(title, domain="chemistry", question=o.get("question", ""), length="A", source=source)


def _bonds(pb) -> np.ndarray | None:
    b = pb.get("bonds") or []
    return np.array([[i, j] for i, j, *_ in b], int) if b else None


def _molecule(pb: dict, o: dict):
    el = [a["element"] for a in pb["atoms"]]
    pos = np.array(pb["positions"], float)
    anims = pb.get("animations", [])
    pick = next((a for a in anims if a["kind"] == "reaction path"), None) or \
        next((a for a in anims if a["kind"] == "dynamics"), None)
    vib = [a for a in anims if a["kind"] == "vibration"]
    name = pb["molecule"]["name"]
    if pick and pick["kind"] == "reaction path":
        s = _scene(f"{name}, along its reaction path", o, "Each frame is an image of the nudged elastic band.")
        n = len(pick["frames"])
        s.timeline(np.arange(n), unit="", label="image along the path", seconds=6, between="interpolated")
        s.atoms(el, np.array(pick["frames"]), bonds=_bonds(pb), name=name)
        s.readout("energy", pick["energies_kJ_mol"], unit="kJ/mol", digits=1)
        k = int(np.argmax(pick["energies_kJ_mol"]))
        s.event(float(k), "transition state", f"{pick['energies_kJ_mol'][k]:.0f} kJ/mol above the start", "info")
        return s
    if pick:
        s = _scene(f"{name} moving at its temperature", o, "Frames are the dynamics run's own positions.")
        n = len(pick["frames"])
        t = np.arange(n) * float(pick.get("frame_fs") or 1.0)
        s.timeline(t, unit="fs", label="time", seconds=8, between="interpolated")
        s.atoms(el, np.array(pick["frames"]), bonds=_bonds(pb), name=name)
        return s
    if vib:
        return _modes(pb, o, el, pos, vib, name)
    s = _scene(name, o, "The optimized structure.")
    s.atoms(el, pos, bonds=_bonds(pb), name=name)
    return s


def _cell(c) -> dict:
    c = np.asarray(c, float)
    return {"origin": [0.0, 0.0, 0.0], "a": c[0].tolist(), "b": c[1].tolist(), "c": c[2].tolist()}


def _crystal(pb: dict, o: dict):
    el = [a["element"] for a in pb["atoms"]]
    pos = np.array(pb["positions"], float)
    cell = np.array(pb["cell"], float)
    name = pb["crystal"]["name"]
    anims = pb.get("animations", [])
    md = next((a for a in anims if a["kind"] == "dynamics"), None)
    eos = next((a for a in anims if a["kind"] == "equation of state"), None)
    if md:
        s = _scene(f"{name}, atoms moving", o, "Frames are the dynamics run's own positions in its periodic box.")
        n = len(md["frames"])
        s.timeline(np.arange(n) * float(md.get("frame_fs") or 1.0), unit="fs", label="time", seconds=8)
        s.atoms(el, np.array(md["frames"]), cell=_cell(cell), name=name, radius=0.5)
        return s
    reps = 2
    shifts = np.array([(i, j, k) for i in range(reps) for j in range(reps) for k in range(reps)]) @ cell
    big_el = el * len(shifts)
    if eos:
        return _strained(pb, o, el, eos, name, reps)
    s = _scene(name, o, "The crystal's cell, repeated twice in each direction.")
    s.atoms(big_el, np.concatenate([pos + v for v in shifts]), cell=_cell(cell), name=name, radius=0.5)
    return s


# ------------------------------------------------------------------------------------- atoms as solid models
def _sphere() -> tuple[np.ndarray, np.ndarray]:
    """A unit icosphere, one subdivision, 42 points."""
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    mid: dict = {}

    def m(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            p = V[i] + V[j]
            V.append(p / np.linalg.norm(p))
            mid[k] = len(V) - 1
        return mid[k]
    out = []
    for a, b, c in F:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(V), np.array(out)


def _radius(sym: str) -> float:
    from .formula import element
    try:
        return float(element(sym).get("radius_pm") or 150) / 100.0
    except ValueError:
        return 1.5


def _balls(s, el, P, values, *, field: str, unit: str, colormap: str, vrange, limit, limit_label: str, name: str,
           size: float = 0.3, digits: int = 3):
    """Atoms as solid spheres at their positions in every frame (frames x atoms x 3), colored by a value per atom
    (atoms, or frames x atoms), so the view reads each atom's value where the pointer rests."""
    U, T = _sphere()
    r = np.array([_radius(e) * size for e in el])
    nv = len(U)
    base = (P[0][:, None, :] + U[None] * r[:, None, None]).reshape(-1, 3)
    tris = np.concatenate([T + i * nv for i in range(len(el))])
    m = s.mesh(base, tris, name=name, color="#c9ced4", max_vertices=len(base) + 1)
    m.deform(np.repeat(P - P[0], nv, axis=1))
    v = np.asarray(values, float)
    m.color_by(field, np.repeat(v, nv, axis=-1), unit=unit, colormap=colormap, range=vrange, limit=limit,
               limit_label=limit_label, digits=digits)
    return m


def _sticks(s, P, pairs, *, name: str, color: str = "#8a9099", radius: float = 0.09, seg: int = 8):
    """Bonds or cell edges as thin tubes between points, following the points frame by frame."""
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if not len(pairs):
        return None
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)

    def build(Q):
        X = []
        for i, j in pairs:
            p, q = Q[i], Q[j]
            d = q - p
            d = d / max(np.linalg.norm(d), 1e-9)
            u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0])
            u /= np.linalg.norm(u)
            w = np.cross(d, u)
            ring = radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * w)
            X.append(np.concatenate([p + ring, q + ring]))
        return np.concatenate(X)
    T = []
    for k in range(len(pairs)):
        o = k * 2 * seg
        for j in range(seg):
            n = (j + 1) % seg
            T += [(o + j, o + n, o + seg + n), (o + j, o + seg + n, o + seg + j)]
    X0 = build(P[0])
    m = s.mesh(X0, np.array(T), name=name, color=color, max_vertices=len(X0) + 1)
    if len(P) > 1:
        m.deform(np.stack([build(Q) for Q in P]) - X0)
    return m


def _modes(pb: dict, o: dict, el, pos, vib, name):
    """A molecule to drive by hand: the slider steps through its vibration modes, and within each one swings it
    from one turning point through its rest shape to the other. Atoms are colored by their partial charge, so
    the view reads each atom's charge under the pointer, and the readouts give the mode's frequency and IR
    strength. Nothing moves until the slider does."""
    from .sweep import mode_inputs, swings
    vib = sorted(vib, key=lambda v: v.get("frequency_cm") or 0)
    amps = swings()
    vec = []
    for v in vib:
        if v.get("vector"):
            vec.append(np.array(v["vector"], float))
        else:                                    # older runs saved only frames, a quarter period is the full swing
            fr = np.array(v["frames"], float)
            vec.append(fr[len(fr) // 4] - pos)
    P = np.stack([pos + a * d for d in vec for a in amps])
    t = mode_inputs(len(vib), amps)
    best = int(np.argmax([v.get("intensity_km_mol") or 0 for v in vib]))
    s = _scene(f"{name}, its vibrations by hand", o,
               "Drag through the modes, each swung from one turning point to the other. Swings are enlarged so "
               "they show. Frequencies, IR strengths and charges are computed.")
    s.timeline(t, unit="", label="vibration mode, then its swing", between="step", kind="input",
               design=best + 1 + 0.4 * amps[-1],
               note="Whole numbers are the modes in order of frequency. Either side of one is its swing.")
    q = pb.get("charges")
    if q is not None and len(q) == len(el):
        q = np.array(q, float)
        span = max(float(np.abs(q).max()), 0.05)
        _balls(s, el, P, q, field="partial charge", unit="e", colormap="coolwarm", vrange=(-span, span),
               limit=0.5, limit_label="an ionic charge", name=name, digits=3)
        s.note("Atoms are colored by partial charge, red positive and blue negative.")
    else:
        s.atoms(el, P, bonds=_bonds(pb), name=name)
    b = _bonds(pb)
    if q is not None and b is not None:
        _sticks(s, P, b, name="bonds")
    nf = len(amps)
    s.readout("frequency", np.repeat([v.get("frequency_cm") or 0 for v in vib], nf), unit="cm-1", digits=4)
    s.readout("IR strength", np.repeat([v.get("intensity_km_mol") or 0 for v in vib], nf), unit="km/mol",
              digits=3)
    s.readout("largest swing drawn", np.concatenate([np.abs(amps) * np.linalg.norm(d, axis=1).max() for d in vec]),
              unit="A", digits=2)
    if (pb.get("properties") or {}).get("dipole_D") is not None:
        s.readout("dipole", np.full(len(t), pb["properties"]["dipole_D"]), unit="D", digits=3)
    top = float(pos[:, 2].max()) + 2.5
    for k, v in enumerate(vib):
        s.label(v["name"] + (", the strongest IR band" if k == best else ""), [0, 0, top], t0=k + 1 - 0.45,
                t1=k + 1 + 0.45, kind="peak" if k == best else "note")
    c = pos.mean(0)
    size = float(np.ptp(pos, axis=0).max()) + 3
    s.camera(target=c, direction=(0.2, -1, 0.5), distance=size * 2.4,
             presets=[{"name": "Front", "target": c, "dir": (0, -1, 0.05), "dist": size * 2.4},
                      {"name": "Top", "target": c, "dir": (0.01, -0.05, 1), "dist": size * 2.4},
                      {"name": "Side", "target": c, "dir": (1, 0, 0.1), "dist": size * 2.4}])
    return s


def _strained(pb: dict, o: dict, el, eos: dict, name: str, reps: int = 2):
    """A crystal to squeeze and stretch by hand: the slider strains its lattice, and each strain is its own
    energy from the run, with the pressure it takes and the stiffness there from the equation of state. The atoms
    are colored by the strain energy stored in each, against the heat that melts the crystal (or the thermal
    energy at room temperature when that is not known), past which the strained lattice has no reason to hold."""
    cells = [np.array(c, float) for c in eos["cells"]]
    sh = np.array([(i, j, k) for i in range(reps) for j in range(reps) for k in range(reps)], float)
    P = np.stack([np.concatenate([np.array(p) + v for v in sh @ c]) for p, c in zip(eos["frames"], cells)])
    big = el * len(sh)
    E = np.array(eos["energies_eV_per_atom"], float)
    if "strain_pct" in eos:
        x = np.array(eos["strain_pct"], float)
    else:
        v = np.array(eos["volumes_A3"], float)
        x = 100 * ((v / v[int(np.argmin(E))]) ** (1 / 3) - 1)
    n = len(x)
    props = pb.get("properties") or {}
    lim, lim_label = props.get("fusion_eV_atom"), "the heat that melts it"
    if not lim:
        lim, lim_label = 8.617333e-5 * 298.15, "thermal energy at 298 K"
    s = _scene(f"{name}, squeezed and stretched by hand", o,
               "Each strain is its own energy from the run's calculator. Pressure and stiffness come from the "
               "Birch-Murnaghan fit through them.")
    s.timeline(x, unit="%", label="lattice strain", between="step", kind="input", design=0.0,
               note="Below zero the cell is squeezed, above it stretched, the same in every direction.")
    _balls(s, big, P, np.repeat(E[:, None], len(big), axis=1), field="strain energy", unit="eV/atom",
           colormap="viridis", vrange=(0, max(float(E.max()), lim)), limit=lim, limit_label=lim_label, name=name,
           size=0.32, digits=3)
    # the unit cell's edges, following the strain
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(corners[a] - corners[b]).sum() == 1]
    _sticks(s, np.stack([corners @ c for c in cells]), edges, name="unit cell", color="#ffd400", radius=0.04)
    if "bulk_GPa" in eos:
        s.readout("bulk modulus", eos["bulk_GPa"], unit="GPa", digits=3)
        s.readout("pressure", eos["pressure_GPa"], unit="GPa", digits=3)
    a0 = props.get("lattice_A")
    if a0:
        s.readout("lattice constant", a0 * (1 + x / 100), unit="A", digits=4)
    s.readout("strain energy", E, unit="eV/atom", digits=3, limit=lim)
    c = cells[n // 2].sum(0) * reps / 2
    d = float(np.linalg.norm(cells[-1].sum(0))) * reps
    s.camera(target=c, direction=(0.9, -1.2, 0.7), distance=d * 2.2,
             presets=[{"name": "Corner", "target": c, "dir": (1, -1, 1), "dist": d * 2.2},
                      {"name": "Face", "target": c, "dir": (0, -1, 0.02), "dist": d * 2.2},
                      {"name": "Top", "target": c, "dir": (0.01, -0.02, 1), "dist": d * 2.2}])
    return s


def _reaction(pb: dict, o: dict):
    """The reaction as a model to drive by hand, across the input chem.sweep chose for it. A reaction in water
    stands in a beaker sized to the solution the run assumed: the slider adds the solid in grams, the solution
    warms or cools by what has reacted, any of the solid the solution cannot take sits on the bottom, and the gas
    given off is gathered in one sphere at 1 atm so its true volume shows. A charge of solids burns in a crucible:
    the slider sets the mixture, and the burned charge glows at the products' temperature solved for that mix,
    with whatever is left over on top. Every frame is the whole finished result for its input, not a moment."""
    sw = pb.get("sweep")
    if not sw or not sw.get("points"):
        return _reaction_by_temperature(pb, o)
    from ..viz import props as P
    from ..viz.adapters.common import lathe
    pts, inp = sw["points"], sw["input"]
    x = np.array([p["x"] for p in pts], float)
    n = len(x)
    T0 = float((pb.get("conditions") or {}).get("T_K") or 298.15)
    lim = sw.get("limit") or {}
    s = _scene(pb.get("equation") or "reaction", o,
               "Each value of the slider is its own solve of the reaction's heat, gas and temperature. Thermodynamics "
               "gives how far and how hot, not how fast.")
    s.head["length"] = "mm"
    s.timeline(x, unit=inp["unit"], label=inp["name"], between="step", kind="input", design=inp["design"])
    TK = np.array([np.nan if p.get("T_K") is None else p["T_K"] for p in pts], float)
    if inp["what"] == "added":
        gas = np.array([p["gas_L"] for p in pts], float)
        r0 = 0.0
        sol_ml = float(inp.get("solution_g") or 1000.0)
        R = (sol_ml * 1000 / (1.8 * np.pi)) ** (1 / 3)
        fill, H = 1.8 * R, 2.7 * R
        s.mesh(*lathe([(R * 0.97, 2), (R, 0), (R, H), (R * 0.97, H), (R * 0.97, 2)], 48), name="beaker",
               color="#cfe3f0", opacity=0.2, prop=True)
        liq = lathe([(0, 2.5), (R * 0.96, 2.5), (R * 0.96, fill), (0, fill)], 48)
        L = s.mesh(*liq, name="solution", color="#9cc6e0", opacity=0.6)
        d = max(float(np.nanmax(np.abs(TK - T0))), 0.5)
        L.color_by("solution temperature", np.outer(TK, np.ones(len(liq[0]))), unit="K", colormap="coolwarm",
                   range=(T0 - d, T0 + d), limit=lim.get("T_K"), limit_label=lim.get("label", ""), digits=4)
        # what the solution could not take, heaped on the bottom at about 2.2 g/cm3
        left = np.array([p.get("left_g") or 0.0 for p in pts], float)
        if left.max() > 0:
            vmax = left.max() / 2.2 * 1000                    # mm3
            h = (3 * vmax / (np.pi * 2.5 ** 2)) ** (1 / 3)
            Xh, Th = lathe([(0, 2.5), (min(2.5 * h, R * 0.9), 2.5), (0, 2.5 + h)], 32)
            k = (left / left.max()) ** (1 / 3)
            U = np.stack([np.c_[Xh[:, 0] * (f - 1), Xh[:, 1] * (f - 1), (Xh[:, 2] - 2.5) * (f - 1)] for f in k])
            heap = s.mesh(Xh, Th, name=f"{inp['name'].removesuffix(' added')} left undissolved", color="#f4f2ec")
            heap.deform(U)
        if gas.max() > 0:
            gname = next((sp["formula"] for sp in pb.get("products", []) if sp.get("phase") == "g"), "gas")
            rg = (3 * gas * 1e6 / (4 * np.pi)) ** (1 / 3)        # mm, the gas at 1 atm in one sphere
            Xs, Ts = _sphere()
            Xs, Ts = _subdivide(Xs, Ts)
            r0 = max(float(rg.max()), 1.0)
            c0 = np.array([0, 0, H + r0])
            X0 = c0 + Xs * r0
            U = np.stack([(np.array([0, 0, H + r]) + Xs * r) - X0 for r in rg])
            ball = s.mesh(X0, Ts, name=f"{gname} gathered at 1 atm", color="#e8f3ff", opacity=0.28)
            ball.deform(U)
            s.label(f"{gname} " + "{" + f"{gname} given off" + "}", [0, 0, H + 2 * r0 + 10], kind="peak")
            s.readout(f"{gname} given off", gas, unit="L", digits=3)
        s.readout("temperature change", TK - T0, unit="K", digits=3)
        heat = np.array([p["heat_kJ"] for p in pts], float)
        s.readout("heat given off" if heat.max() > 0 else "heat taken in", np.abs(heat), unit="kJ", digits=3)
        s.readout("left undissolved", left, unit="g", digits=3)
        s.ground(0, size=max(6 * R, 4 * r0 if gas.max() > 0 else 0), grid=20)
        top = H + (2 * r0 if gas.max() > 0 else 0)
        s.camera(target=[0, 0, top * 0.45], direction=(0.3, -1, 0.35), distance=top * 2.4,
                 presets=[{"name": "Beaker", "target": [0, 0, H * 0.4], "dir": (0.3, -1, 0.45), "dist": H * 3.2},
                          {"name": "Whole", "target": [0, 0, top * 0.45], "dir": (0.3, -1, 0.35), "dist": top * 2.4},
                          {"name": "Top", "target": [0, 0, H * 0.5], "dir": (0.01, -0.2, 1), "dist": H * 3.5}])
        return s
    # a charge of solids burning in a crucible
    Rc, Hc = 30.0, 60.0
    cru = lathe([(Rc * 1.25, Hc + 4), (Rc * 1.15, Hc + 4), (Rc * 0.95 + 2, 0), (Rc * 0.75, -6), (Rc * 0.95 + 8, -6),
                 (Rc * 1.35, Hc + 4)], 36)
    s.mesh(*cru, name="crucible", color="#7a6a5e", opacity=0.45, prop=True)
    charge = np.array([p.get("charge_g") or 1.0 for p in pts], float)
    left = np.array([sum((p.get("left") or {}).values()) for p in pts], float)
    full = charge / charge.max() * Hc                       # the charge's height, its density taken as constant
    burned_h = full * (1 - left / charge)
    prof = lambda h: [(0.0, 0.0)] + [(Rc * (0.95 + 0.35 * z / Hc), z) for z in np.linspace(0, h, 12)] + [(0.0, h)]
    Xb, Tb = lathe(prof(Hc), 36)

    def squash(h):
        z = Xb[:, 2] * h / Hc
        r = np.hypot(Xb[:, 0], Xb[:, 1])
        k = np.where(r > 1e-9, (Rc * (0.95 + 0.35 * z / Hc)) / np.maximum(Rc * (0.95 + 0.35 * Xb[:, 2] / Hc), 1e-9),
                     1.0)
        k = np.where(r > 1e-9, k, 1.0)
        return np.c_[Xb[:, 0] * k, Xb[:, 1] * k, z] - Xb
    ch = s.mesh(Xb, Tb, name="burned charge", color="#8c5a3c")
    ch.deform(np.stack([squash(h) for h in burned_h]))
    tlo = T0
    thi = float(np.nanmax([np.nanmax(TK), lim.get("T_K") or 0]))
    ch.color_by("products temperature", np.outer(TK, np.ones(len(Xb))), unit="K", colormap="inferno",
                range=(tlo, thi * 1.02), limit=lim.get("T_K"), limit_label=lim.get("label", ""), digits=4)
    ch.glow = np.clip((TK - 1000) / 2500, 0, 1)
    if left.max() > 0:
        Xl, Tl = lathe([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 36)
        th = np.maximum(full - burned_h, 1e-3)
        r = Rc * (0.95 + 0.35 * full / Hc)
        U = np.stack([np.c_[Xl[:, 0] * r[i], Xl[:, 1] * r[i], burned_h[i] + Xl[:, 2] * th[i]] - Xl for i in range(n)])
        names = sorted({k for p in pts for k in (p.get("left") or {})})
        rest = s.mesh(Xl, Tl, name=" and ".join(names) + " left over", color="#a8a8a8")
        rest.deform(U)
    s.label("{products temperature}", [Rc * 1.6, 0, Hc * 0.6], kind="peak")
    s.readout("heat released", [p["heat_kJ"] for p in pts], unit="kJ", digits=4)
    s.readout("products temperature", TK, unit="K", digits=4, limit=lim.get("T_K"))
    s.readout("heat per kg of charge", [p.get("heat_kJ_per_kg") or 0 for p in pts], unit="kJ/kg", digits=4)
    s.readout("charge", charge, unit="g", digits=4)
    s.readout("left over", left, unit="g", digits=3)
    s.ground(-6, size=400, grid=20)
    s.camera(target=[0, 0, Hc * 0.45], direction=(0.4, -1, 0.5), distance=Hc * 4.2,
             presets=[{"name": "Crucible", "target": [0, 0, Hc * 0.45], "dir": (0.4, -1, 0.5), "dist": Hc * 4.2},
                      {"name": "Into it", "target": [0, 0, Hc * 0.5], "dir": (0.05, -0.3, 1), "dist": Hc * 3.5},
                      {"name": "Side", "target": [0, 0, Hc * 0.45], "dir": (0, -1, 0.05), "dist": Hc * 4.2}])
    return s


def _subdivide(V, F):
    """One more subdivision of a unit icosphere, for a large smooth sphere."""
    V = list(V)
    mid: dict = {}

    def m(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            p = V[i] + V[j]
            V.append(p / np.linalg.norm(p))
            mid[k] = len(V) - 1
        return mid[k]
    out = []
    for a, b, c in F:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(V), np.array(out)


def _reaction_by_temperature(pb: dict, o: dict):
    """A reaction with no input to sweep but its temperature: the Gibbs energy and equilibrium constant at each
    temperature, each its own evaluation, with the reaction going backward where the Gibbs energy turns positive."""
    fr = pb.get("frames") or []
    T0 = float((pb.get("conditions") or {}).get("T_K") or 298.15)
    s = _scene(pb.get("equation") or "reaction", o, "Each temperature is its own thermochemistry evaluation.")
    if len(fr) < 2:
        return s
    s.timeline([f["T_K"] for f in fr], unit="K", label="temperature", between="step", kind="input", design=T0)
    s.readout("Gibbs energy", [f["dG_kJ"] for f in fr], unit="kJ/mol", digits=4, limit=0.0)
    s.readout("log10 K", [f["log10K"] for f in fr], digits=4)
    return s


def _mixture(pb: dict, o: dict):
    fr = pb["frames"]
    s = _scene(pb["mixture"]["label"], o, "The equilibrium mixture at each temperature, from Cantera.")
    s.timeline([f["T_K"] for f in fr], unit="K", label="temperature", seconds=6, between="interpolated")
    names = []
    for f in fr:
        for k in f["x"]:
            if k not in names:
                names.append(k)
    for k in names[:6]:
        s.readout(f"{k} mole fraction", [f["x"].get(k, 0.0) for f in fr], digits=3)
    return s
