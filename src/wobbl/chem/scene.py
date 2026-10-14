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
        a = max(vib, key=lambda v: v.get("intensity_km_mol") or 0)
        s = _scene(f"{name}, its {a['name']} vibration", o,
                   "The strongest IR mode, its swing enlarged so it shows. The frequency is computed.")
        n = len(a["frames"])
        s.timeline(np.linspace(0, a.get("period_fs") or 1.0, n), unit="fs", label="one period", seconds=3,
                   between="interpolated")
        s.atoms(el, np.array(a["frames"]), bonds=_bonds(pb), name=name)
        s.note(f"Other modes in the file: {', '.join(v['name'] for v in vib if v is not a)}")
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
    s = _scene(pb.get("equation") or "reaction", o, "Reactants on the left and products on the right, drawn from "
                                                     "their structures, with the computed Gibbs energy through "
                                                     "temperature.")
    fr = pb["frames"]
    s.timeline([f["T_K"] for f in fr], unit="K", label="temperature", seconds=6, between="interpolated")
    s.readout("reaction Gibbs energy", [f["dG_kJ"] for f in fr], unit="kJ", digits=1, limit=0.0)
    s.readout("log10 K", [f["log10K"] for f in fr], digits=2)
    x = -6.0
    for side, sign in (("reactants", -1), ("products", 1)):
        x = 3.0 if sign > 0 else -14.0
        for sp in pb.get(side, []):
            if not sp.get("atoms"):
                continue
            el = [a["element"] for a in sp["atoms"]]
            p = np.array(sp["positions"], float)
            width = float(np.ptp(p[:, 0])) if len(p) > 1 else 1.0
            s.atoms(el, p + np.array([x + width / 2, 0.0, 0.0]), bonds=_bonds(sp),
                    name=f"{sp.get('coef', 1):g} {sp['formula']}({sp.get('phase', '')})")
            x += width + 3.0
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
