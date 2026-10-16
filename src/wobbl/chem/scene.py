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
    """The reaction happening in its vessel. A reaction with a liquid or a solution runs in a beaker, its gas
    rising as bubbles in proportion to how far it has gone and the liquid warming or cooling by the run's
    adiabatic change; an all solid charge burns in a crucible, a front running down through it from the top, the
    burned part at the run's adiabatic temperature. The timeline is how far the reaction has gone, not clock time,
    since the run solves the thermochemistry and not the rate."""
    from ..viz.adapters.common import lathe
    from ..viz import props as P
    T0 = float((pb.get("conditions") or {}).get("T_K") or 298.15)
    ext = float(pb.get("equilibrium_extent") or 1.0)
    adi = pb.get("adiabatic") or {}
    rise = float(adi.get("rise_K") if adi.get("rise_K") is not None else (adi.get("T_K") or T0) - T0)
    T_end = float(adi.get("T_K") or T0 + rise)
    dH = float((pb.get("energies") or {}).get("dH_kJ") or 0.0)
    gas = float(pb.get("gas_L") or 0.0)
    phases = {sp.get("phase") for sp in pb.get("reactants", [])}
    wet = bool(phases & {"aq", "l"})
    n = 90
    prog = np.linspace(0, 1, n) * ext
    s = _scene(pb.get("equation") or "reaction", o,
               "The timeline is the share of the reaction done, not clock time. Gas, heat and temperature at each "
               "point are the run's totals times that share.")
    s.head["length"] = "mm"
    s.timeline(prog * 100, unit="%", label="reacted", seconds=9, between="interpolated")
    lo, hi = sorted([T0, T_end])
    hi = hi if hi - lo > 1 else lo + 1
    rng = np.random.default_rng(11)
    if wet:
        R, H, fill = 40.0, 110.0, 70.0
        glass = lathe([(R * 0.96, 2), (R, 0), (R, H), (R * 0.96, H), (R * 0.96, 2)], 40)
        s.mesh(*glass, name="beaker", color="#cfe3f0", opacity=0.22, prop=True)
        liq = lathe([(0, 2.5), (R * 0.95, 2.5), (R * 0.95, fill), (0, fill)], 40)
        L = s.mesh(*liq, name="solution", color="#9cc6e0", opacity=0.55)
        d = max(abs(rise), 0.5)
        L.color_by("temperature", np.outer(T0 + rise * prog / max(ext, 1e-9), np.ones(len(liq[0]))), unit="K",
                   colormap="coolwarm", range=(T0 - d, T0 + d))
        solid = next((sp for sp in pb.get("reactants", []) if sp.get("phase") == "s"), None)
        if solid is not None:
            heap = lathe([(0, 2.5), (R * 0.55, 2.5), (0, 14)], 30)
            m = s.mesh(*heap, name=solid.get("name") or solid["formula"], color="#f2f0ea")
            m.deform(np.c_[np.zeros(len(heap[0])), np.zeros(len(heap[0])), -(heap[0][:, 2] - 2.5) * 0.95],
                     scale=prog / max(ext, 1e-9))
        if gas > 0:
            k = 260
            born = np.sort(rng.uniform(0, 1, k)) * ext            # gas comes off in step with the reaction
            life = 0.12 * ext
            r = R * 0.8 * np.sqrt(rng.uniform(0, 1, k))
            a = rng.uniform(0, 2 * np.pi, k)
            pos = np.full((n, k, 3), np.nan)
            for f, p in enumerate(prog):
                age = (p - born) / life
                live = (age >= 0) & (age <= 1)
                pos[f, live, 0] = r[live] * np.cos(a[live]) + np.sin(age[live] * 9 + a[live]) * 2
                pos[f, live, 1] = r[live] * np.sin(a[live])
                pos[f, live, 2] = 4 + age[live] * (fill - 6)
            gas_name = next((sp["formula"] for sp in pb.get("products", []) if sp.get("phase") == "g"), "gas")
            s.points(pos, color="#eef6ff", size=2.6, name=f"{gas_name} bubbles")
            s.readout(f"{gas_name} given off", gas * prog, unit="L/mol", digits=3)
        s.camera(target=[0, 0, H * 0.4], direction=(0.3, -1, 0.45), distance=H * 3.0)
    else:
        R, H = 30.0, 60.0
        cru = lathe([(R * 1.25, H + 4), (R * 1.15, H + 4), (R * 0.95 + 2, 0), (R * 0.75, -6), (R * 0.95 + 8, -6),
                     (R * 1.35, H + 4)], 36)
        s.mesh(*cru, name="crucible", color="#7a6a5e", opacity=0.45, prop=True)
        # the charge in rings, so the front can pass ring by ring
        prof = [(0.0, 0.0)] + [(R * (0.95 + 0.35 * z / H), z) for z in np.linspace(0, H, 24)] + [(0.0, H)]
        X, T = lathe(prof, 36)
        front = H * (1 - prog / max(ext, 1e-9))
        width = H * 0.06
        burned = np.clip((X[None, :, 2] - front[:, None]) / width + 0.5, 0, 1)
        temp = T0 + (T_end - T0) * burned
        ch = s.mesh(X, T, name="charge", color="#8c5a3c")
        ch.color_by("temperature", temp, unit="K", colormap="inferno", range=(lo, hi))
        disc = s.body("cylinder", [R * 1.1, 0.6], pos=np.c_[np.zeros(n), np.zeros(n), front], color="#fff1c0",
                      name="reaction front", opacity=0.85)
        disc.light(np.where((prog > 0) & (prog < ext * 0.999), 1.0, 0.0))
        disc.show(1e-6, ext * 100 * 0.999)
        k = 120
        pos = np.full((n, k, 3), np.nan)
        start = rng.uniform(0, 1, k)
        v = np.c_[rng.normal(0, 0.5, k), rng.normal(0, 0.5, k), rng.uniform(1.5, 3, k)]
        for f, p in enumerate(prog):
            if not 0 < p < ext * 0.999:
                continue
            age = (start + f * 0.13) % 1
            pos[f] = np.c_[v[:, 0] * age * H * 0.6, v[:, 1] * age * H * 0.6,
                           H + 3 + v[:, 2] * age * H * 0.5 - 9.8 * age ** 2 * H * 0.08]
        s.points(pos, color="#ffc04a", size=1.2, name="sparks")
        s.label("front {temperature at the front}", [R * 1.4, 0, 0], kind="peak", follow=disc)
        s.readout("temperature at the front", np.where(prog > 0, T_end, T0), unit="K", digits=4)
        s.camera(target=[0, 0, H * 0.45], direction=(0.4, -1, 0.5), distance=H * 4.2)
    s.ground(-6 if not wet else 0, size=400, grid=20)
    s.readout("heat given off" if dH <= 0 else "heat taken in", abs(dH) * prog, unit="kJ/mol", digits=4)
    s.readout("temperature change", rise * prog / max(ext, 1e-9), unit="K", digits=3)
    s.event(ext * 100, "equilibrium", f"{ext * 100:.3g}% reacted")
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
