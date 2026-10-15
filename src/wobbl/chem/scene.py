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
        # every mode in turn, two swings each, so the whole spectrum plays as motion
        vib = sorted(vib, key=lambda v: v.get("frequency_cm") or 0)
        best = max(vib, key=lambda v: v.get("intensity_km_mol") or 0)
        s = _scene(f"{name}, its vibrations", o, "Each mode's swing is enlarged so it shows. Frequencies and IR "
                                                 "strengths are computed.")
        P, freq, inten = [], [], []
        for v in vib:
            fr = np.array(v["frames"], float)
            for rep in range(2):
                P.append(fr)
            freq += [v.get("frequency_cm") or 0] * (2 * len(fr))
            inten += [v.get("intensity_km_mol") or 0] * (2 * len(fr))
        P = np.concatenate(P)
        t = np.arange(len(P), dtype=float)
        s.timeline(t, unit="", label="frame", seconds=3.2 * len(vib), between="interpolated")
        s.atoms(el, P, bonds=_bonds(pb), name=name)
        s.readout("frequency", freq, unit="cm-1", digits=4)
        s.readout("IR strength", inten, unit="km/mol", digits=3)
        k = 0
        for v in vib:
            s.event(float(k), v["name"], "the strongest IR band" if v is best else "")
            k += 2 * len(v["frames"])
        return s
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
        frames = []
        for p, c in zip(eos["frames"], eos["cells"]):
            c = np.array(c)
            sh = np.array([(i, j, k) for i in range(reps) for j in range(reps) for k in range(reps)]) @ c
            frames.append(np.concatenate([np.array(p) + v for v in sh]))
        s = _scene(f"{name}, squeezed and stretched", o, "Each frame is a volume of the equation of state, with "
                                                         "its computed energy.")
        n = len(frames)
        s.timeline(np.array(eos["volumes_A3"]), unit="A^3", label="cell volume", seconds=6, between="interpolated")
        s.atoms(big_el, np.array(frames), cell=_cell(eos["cells"][n // 2]), name=name, radius=0.5)
        s.readout("energy above the minimum", eos["energies_eV_per_atom"], unit="eV/atom", digits=4)
        return s
    s = _scene(name, o, "The crystal's cell, repeated twice in each direction.")
    s.atoms(big_el, np.concatenate([pos + v for v in shifts]), cell=_cell(cell), name=name, radius=0.5)
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
