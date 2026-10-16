"""Playback files: the run's own data, thinned to what a viewer needs to animate it truthfully.

Each run writes output/playback.json, a few tens of kB, with a manifest naming the kind of run, the units and the
frame count, the fixed geometry, and frames in order. A motor's frames step through the burn in time and carry the
chamber pressure, thrust, Kn, regression, propellant left and the casing's safety factor at that instant, with each
grain's burning cross section: the core diameter and remaining length of a round core, or the port outline of any
other core as a closed polyline in mm. A flame's frames step through equivalence ratio, each a real equilibrium
solve, and a propellant's step down the nozzle from throat to exit. Numbers are rounded to what the run resolves,
so the same input always gives the same file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grains import Bates, EndBurner, MapGrain, Tube

VERSION = 1


def _r(x, digits: int = 4):
    """Round to significant digits, so the file stays small and identical from run to run."""
    if x is None or not np.isfinite(x):
        return None
    x = float(x)
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def _write(out: Path, data: dict) -> str:
    path = Path(out) / "playback.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path.name


def motor(spec, b, case_vals: dict, out: Path, frames: int = 120) -> str:
    """Frames through the burn with each grain's cross section, pressure, thrust and the casing's margin."""
    m = spec.motor
    on = np.nonzero(b.F > 0.002 * b.peak_thrust)[0]
    t_end = float(b.t[min(on[-1] + 1, len(b.t) - 1)]) if len(on) else float(b.t[-1])
    times = np.linspace(0.0, t_end, frames)
    idx = np.clip(np.searchsorted(b.t, times), 0, len(b.t) - 1)
    burned = np.concatenate([[0.0], np.cumsum(0.5 * (b.mdot[1:] + b.mdot[:-1]) * np.diff(b.t))])
    left = np.clip(b.mass * (1 - burned / max(burned[-1], 1e-12)), 0.0, None)
    py = case_vals.get("yield_pressure_Pa")
    types, slots, z = [], [], 0.0
    for g in m.grains:
        k = next((i for i, (h, _) in enumerate(types) if h is g), None)
        if k is None:
            types.append((g, len(types)))
            k = len(types) - 1
        slots.append({"type": k, "z_mm": _r(z * 1000), "length_mm": _r(g.L * 1000)})
        z += g.L
    xs = [min(float(b.x[i]), m.web) for i in idx]
    gtypes = [_grain_type(g, xs) for g, _ in types]
    rows = []
    for j, i in enumerate(idx):
        P = float(b.P[i])
        rows.append({"t": _r(times[j], 5), "P_MPa": _r(P / 1e6), "F_N": _r(b.F[i]), "kn": _r(m.kn(xs[j])),
                     "x_mm": _r(xs[j] * 1000), "propellant_kg": _r(left[i]), "mdot_kg_s": _r(b.mdot[i]),
                     "casing_sf": _r(py / P, 3) if py and P > m.ambient * 1.5 else None})
    c = spec.casing
    data = {"manifest": {"kind": "solid-motor", "version": VERSION, "generator": "wobbl motor", "frames": len(rows),
                         "time": "frames are evenly spaced in time from ignition to the end of the tail off",
                         "units": {"t": "s", "P_MPa": "MPa", "F_N": "N", "kn": "burning area over throat area",
                                   "x_mm": "mm of regression", "propellant_kg": "kg", "mdot_kg_s": "kg/s",
                                   "casing_sf": "yield pressure over chamber pressure", "lengths": "mm"},
                         "grain_profiles": {"round": "core_mm per frame, a circle centered on the axis",
                                            "tube": "core_mm and outer_mm per frame",
                                            "end": "solid face, only length_mm changes",
                                            "outline": "port per frame as closed polylines of [x, y] in mm, "
                                                       "centered on the axis"}},
            "motor": {"name": spec.name, "designation": b.designation(), "propellant": m.propellant.name,
                      "nozzle": {"throat_mm": _r(m.nozzle.throat * 1000), "exit_mm": _r(m.nozzle.exit * 1000)},
                      "casing": None if c is None else {
                          "material": case_vals.get("material"), "outer_mm": _r(c.outer * 1000),
                          "inner_mm": _r(c.inner * 1000), "length_mm": _r((c.length or z) * 1000),
                          "yield_pressure_MPa": _r(py / 1e6), "allowable_MPa": _r(case_vals["allowable_Pa"] / 1e6),
                          "burst_MPa": _r(case_vals["burst_pressure_Pa"] / 1e6), "target_sf": c.target_sf},
                      "peak_pressure_MPa": _r(b.peak_pressure / 1e6), "impulse_Ns": _r(b.impulse),
                      "burn_time_s": _r(b.burn_time)},
            "grain_types": gtypes, "grains": slots, "frames": rows}
    return _write(out, data)


def _grain_type(g, xs: list[float]) -> dict:
    """A grain's fixed sizes and its cross section at each frame's regression."""
    base = {"shape": g.shape, "diameter_mm": _r(g.D * 1000), "length_mm": _r(g.L * 1000), "ends_burning": g.ends}
    length = [_r(g.length(min(x, g.web)) * 1000 if x < g.web else 0.0) for x in xs]
    if isinstance(g, Bates):
        return {**base, "profile": "round", "length_now_mm": length,
                "core_mm": [_r(min(g.core + 2 * x, g.D) * 1000) for x in xs]}
    if isinstance(g, Tube):
        return {**base, "profile": "tube", "length_now_mm": length,
                "core_mm": [_r(min(g.core + 2 * x, g.D) * 1000) for x in xs],
                "outer_mm": [_r(max(g.D - 2 * x, g.core + 2 * x) * 1000) for x in xs]}
    if isinstance(g, EndBurner):
        return {**base, "profile": "end", "length_now_mm": length}
    if isinstance(g, MapGrain):
        return {**base, "profile": "outline", "length_now_mm": length, "params": {k: _r(v * 1000) if isinstance(
                v, float) else v for k, v in g.params.items()}, "port": [_outline(g, x) for x in xs]}
    return {**base, "profile": "end", "length_now_mm": length}


def _outline(g: MapGrain, x: float, points: int = 48) -> list:
    """The port's edge at regression x as closed polylines in mm, thinned to about ``points`` vertices each."""
    if x >= g.web:
        return []
    # longest first, without the few cell specks the grid leaves right at the port's starting edge
    lines = sorted((ln for ln in g.port_outline(x) if len(ln) >= 8), key=len, reverse=True)
    out = []
    for ln in lines:
        step = max(1, math.ceil(len(ln) / points))
        pts = np.vstack([ln[::step], ln[-1:]]) if (len(ln) - 1) % step else ln[::step]
        out.append([[_r(px * 1000, 3), _r(py * 1000, 3)] for px, py in pts])    # 0.1 mm, the grid's cell
    return out


def flame(out: Path, phis, temps, species: list[dict], phi: float, info: dict) -> str:
    """Frames through equivalence ratio, each an equilibrium flame, with the run's own mixture marked."""
    keys = sorted({k for s in species for k in s}, key=lambda k: -max(s.get(k, 0.0) for s in species))[:10]
    rows = [{"phi": _r(p), "T_K": _r(T, 5), "x": {k: _r(s.get(k, 0.0), 3) for k in keys}}
            for p, T, s in zip(phis, temps, species)]
    here = int(np.argmin(np.abs(np.asarray(phis) - phi)))
    data = {"manifest": {"kind": "flame", "version": VERSION, "generator": "wobbl flame", "frames": len(rows),
                         "sweep": "frames step through equivalence ratio, each a separate equilibrium solve",
                         "units": {"phi": "fuel over stoichiometric fuel", "T_K": "K",
                                   "x": "mole fraction in the burned gas"}, "operating_frame": here},
            "flame": info, "frames": rows}
    return _write(out, data)


def runaway(out: Path, cell, calm, burst, t_trigger, ta_cr: float, ambient: float, frames: int = 150,
            sweep: dict | None = None) -> str:
    """Frames in time of a cell in two ovens either side of the onset, with how much of each reactant is left, and
    with sweep, the cell's peak state in each of a range of ovens (see runaway.oven_sweep)."""
    end = min(calm.t[-1], burst.t[-1])
    times = np.linspace(0.0, end, frames)

    def series(tr):
        T = np.interp(times, tr.t, tr.T)
        left = [np.interp(times, tr.t, tr.y[i]) for i in range(5)]
        # what is left of each reaction's reactant, 1 at the start and 0 when used up
        sei, neg, pos, ele = left[0] / tr.y[0, 0], left[1] / tr.y[1, 0], (1 - left[2]) / (1 - tr.y[2, 0]), left[4]
        return [{"T_C": _r(T[i] - 273.15), "sei": _r(sei[i], 3), "negative": _r(neg[i], 3),
                 "positive": _r(pos[i], 3), "electrolyte": _r(ele[i], 3)} for i in range(frames)]

    a, b = series(calm), series(burst)
    rows = [{"t_min": _r(t / 60, 4), "below": a[i], "above": b[i]} for i, t in enumerate(times)]
    data = {"manifest": {"kind": "cell-runaway", "version": VERSION, "generator": "wobbl runaway",
                         "frames": len(rows), "time": "frames are evenly spaced in time from the moment the cell "
                                                     "goes into the oven",
                         "units": {"t_min": "minutes", "T_C": "C", "sei": "fraction of each reactant left",
                                   "negative": "fraction left", "positive": "fraction left",
                                   "electrolyte": "fraction left"}},
            "cell": {"label": cell.label, "diameter_mm": _r(cell.diameter * 1000), "length_mm": _r(cell.length * 1000),
                     "critical_C": _r(ta_cr - 273.15), "surroundings_C": _r(ambient - 273.15),
                     "below_oven_C": _r(calm.Ta - 273.15), "above_oven_C": _r(burst.Ta - 273.15),
                     "trigger_min": _r(t_trigger / 60) if t_trigger else None,
                     "trigger": "the cell heats itself faster than 10 C/min"},
            "frames": rows}
    if sweep:
        data["sweep"] = {k: ([_r(x) for x in v] if isinstance(v, list) and v and not isinstance(v[0], bool) else v)
                         for k, v in sweep.items()}
    return _write(out, data)


def nozzle(out: Path, rows: list[dict], info: dict) -> str:
    """Frames from the throat to past the exit, shifting and frozen, with Isp at each area ratio."""
    data = {"manifest": {"kind": "nozzle", "version": VERSION, "generator": "wobbl propellant", "frames": len(rows),
                         "sweep": "frames step down the nozzle in pressure, each an equilibrium expansion",
                         "units": {"area_ratio": "area over throat area", "P_MPa": "MPa", "T_K": "K",
                                   "v_m_s": "m/s", "isp_s": "s at the ambient pressure",
                                   "condensed": "mass fraction of condensed products"}},
            "propellant": info, "frames": rows}
    return _write(out, data)


# ------------------------------------------------------------------------------------------- atoms in 3D
def atoms_block(symbols: list[str]) -> list[dict]:
    """Each atom's element with the CPK colour and van der Waals radius a viewer draws it with."""
    from .formula import element
    out = []
    for s in symbols:
        e = element(s)
        out.append({"element": s, "color": "#" + (e.get("color") or "FF1493"),
                    "radius_A": _r((e.get("radius_pm") or 150) / 100.0, 3)})
    return out


def _pos(p) -> list:
    return [[round(float(v), 3) for v in row] for row in np.asarray(p)]


def mode_frames(positions: np.ndarray, vector: np.ndarray, frames: int = 16, amplitude: float = 0.25) -> list:
    """One period of a normal mode, the largest atom swing set to amplitude angstrom so it can be seen."""
    v = np.asarray(vector)
    v = v / max(np.linalg.norm(v, axis=1).max(), 1e-12) * amplitude
    return [_pos(positions + math.sin(2 * math.pi * k / frames) * v) for k in range(frames)]


def molecule(out: Path, name: str, formula: str, symbols, positions, bonds, modes=None, md_frames=None,
             values: dict | None = None, path: dict | None = None) -> str:
    """Atoms, bonds and animations: the strongest IR modes vibrating, the molecular dynamics run, or a reaction
    path, each as frames of positions in angstrom."""
    pos = np.asarray(positions)
    anims = []
    if modes is not None and len(modes.freqs):
        picks = list(np.argsort(-np.asarray(modes.intensities))[:5])
        for extra in (0, len(modes.freqs) - 1):
            if extra not in picks:
                picks.append(extra)
        for k in sorted({int(i) for i in picks}, key=lambda i: modes.freqs[i]):
            label = f", {modes.labels[k]}" if modes.labels[k] else ""
            period = 1e15 / (modes.freqs[k] * 2.99792458e10) if modes.freqs[k] > 0 else None
            anims.append({"kind": "vibration", "name": f"{modes.freqs[k]:.0f} cm-1{label}",
                          "frequency_cm": _r(modes.freqs[k], 5), "intensity_km_mol": _r(modes.intensities[k], 3),
                          "period_fs": _r(period, 4) if period else None, "frames": mode_frames(pos, modes.vectors[k]),
                          "vector": _pos(mode_frames(pos, modes.vectors[k], 4)[1] - pos)})
    if md_frames:
        md = (values or {}).get("md", {})
        anims.append({"kind": "dynamics", "name": f"molecular dynamics at {_r(md.get('T_target', 0), 4)} K",
                      "frame_fs": _r(md.get("fs", 0) / max(len(md_frames), 1), 4), "frames": md_frames})
    if path:
        anims.append({"kind": "reaction path", "name": path.get("name", "reaction path"),
                      "energies_kJ_mol": [_r(e, 4) for e in path["energies"]],
                      "frames": [_pos(p) for p in path["frames"]]})
    data = {"manifest": {"kind": "molecule", "version": VERSION, "generator": "wobbl chem",
                         "frames": sum(len(a["frames"]) for a in anims),
                         "units": {"positions": "angstrom", "frequency_cm": "cm-1", "intensity_km_mol": "km/mol",
                                   "period_fs": "femtoseconds", "frame_fs": "femtoseconds per frame",
                                   "energies_kJ_mol": "kJ/mol relative to the reactant"},
                         "notes": "Vibration frames swing the largest moving atom by 0.25 A so the motion shows, and "
                                  "real amplitudes are smaller. Dynamics and reaction path frames are true positions."},
            "molecule": {"name": name, "formula": formula},
            "atoms": atoms_block(symbols), "positions": _pos(pos),
            "bonds": [[int(i), int(j), _r(o, 2)] for i, j, o in bonds], "animations": anims, "cell": None}
    if values and values.get("charges") is not None and len(values["charges"]) == len(pos):
        data["charges"] = [_r(c, 4) for c in values["charges"]]
    if values:
        data["properties"] = {k: _r(values[k], 5) for k in ("dipole_D", "gap_eV", "homo_eV", "lumo_eV",
                                                             "solvation_kJ_mol") if values.get(k) is not None}
    return _write(out, data)


def _species_3d(sp) -> dict:
    """A species drawn in 3D: an RDKit structure for molecules and ions, a single atom for metals and elements."""
    out = {"formula": sp.formula, "name": sp.name, "kind": sp.kind}
    try:
        if sp.kind in ("metal", "element") and len(sp.counts) == 1:
            sym = next(iter(sp.counts))
            out.update(atoms=atoms_block([sym]), positions=[[0.0, 0.0, 0.0]], bonds=[])
        elif sp.smiles:
            from .structure import from_smiles
            st = from_smiles(sp.smiles, conformers=1)
            c = st.positions - st.positions.mean(0)
            out.update(atoms=atoms_block(st.symbols), positions=_pos(c),
                       bonds=[[int(i), int(j), _r(o, 2)] for i, j, o in st.bonds])
    except Exception:  # noqa: BLE001  a species that cannot be drawn is listed without atoms
        pass
    return out


def reaction(out: Path, q, values: dict, quick: bool = False, sweep: dict | None = None) -> str:
    """Reactants and products in 3D with their amounts and phases, the energy levels, how the Gibbs energy and
    equilibrium constant change with temperature, and the result solved across the input a designer would change
    (chem.sweep), for a viewer to show the reaction as a model the user drives."""
    from .react import thermo
    sides = {}
    for key, terms in (("reactants", q.reactants), ("products", q.products)):
        sides[key] = []
        for t in terms:
            d = {"coef": _r(t.coef, 4), "phase": t.phase}
            d.update(_species_3d(t.species) if not quick else {"formula": t.species.formula, "name": t.species.name})
            sides[key].append(d)
    T = values.get("temperature_K", 298.15)
    aqueous = any(t.phase == "aq" for t in q.reactants + q.products)
    lo, hi = (275.0, 370.0) if aqueous else (250.0, max(1500.0, T * 1.5))
    frames = []
    for x in np.linspace(lo, hi, 24):
        th = thermo(q, float(x))
        frames.append({"T_K": _r(x, 5), "dG_kJ": _r(th["dG"] / 1000, 5), "log10K": _r(th["log10K"], 4)})
    ad = values.get("adiabatic") or {}
    data = {"manifest": {"kind": "reaction", "version": VERSION, "generator": "wobbl chem", "frames": len(frames),
                         "units": {"positions": "angstrom", "dH_kJ": "kJ per mole of reaction as written",
                                   "dG_kJ": "kJ per mole of reaction", "T_K": "K", "gas_L": "L per mole of reaction",
                                   "rise_K": "K"},
                         "notes": "Frames step through temperature, each a separate thermochemistry evaluation. "
                                  "Species structures are RDKit geometries for drawing."},
            "equation": values.get("equation"), "rule": values.get("rule"), **sides,
            "energies": {"dH_kJ": _r(values.get("dH_kJ"), 5), "dG_kJ": _r(values.get("dG_kJ"), 5),
                         "log10K": _r(values.get("log10K"), 4), "barrier_kJ": None},
            "conditions": {"T_K": _r(T, 5), "P_Pa": _r(values.get("pressure_Pa"), 5)},
            "gas_L": _r(values.get("gas_L"), 4), "adiabatic": {k: _r(v, 5) for k, v in ad.items()
                                                               if isinstance(v, (int, float))},
            "equilibrium_extent": _r((values.get("equilibrium") or {}).get("extent"), 4),
            "frames": frames}
    if sweep:
        data["sweep"] = sweep
    return _write(out, data)


def mixture(out: Path, label: str, values: dict) -> str:
    """A gas mixture's equilibrium composition as frames through temperature, with its flame point marked."""
    data = {"manifest": {"kind": "mixture", "version": VERSION, "generator": "wobbl chem",
                         "frames": len(values.get("sweep", [])),
                         "units": {"T_K": "K", "x": "mole fraction at equilibrium", "P_Pa": "Pa"},
                         "notes": "Frames step through temperature at fixed pressure, each a Cantera equilibrium."},
            "mixture": {"label": label, "start": {k: _r(v, 4) for k, v in values.get("mixture", {}).items()},
                        "T_K": _r(values.get("temperature_K"), 5), "P_Pa": _r(values.get("pressure_Pa"), 5),
                        "flame_T_K": _r(values.get("T_ad_K"), 5)},
            "frames": values.get("sweep", [])}
    return _write(out, data)


def crystal(out: Path, name: str, atoms, desc: str, frames=None, volumes=None, energies=None, values=None, md=None,
            md_fs: float = 0.0, extra: dict | None = None) -> str:
    """A crystal's unit cell and atoms, with frames of the cell breathing through the equation of state (each
    volume a computed energy) or of its atoms moving in a dynamics run."""
    anims = []
    if frames:
        e0 = min(energies)
        anims.append({"kind": "equation of state", "name": "the cell squeezed and stretched",
                      "energies_eV_per_atom": [_r((e - e0) / len(atoms), 4) for e in energies],
                      "volumes_A3": [_r(v, 5) for v in volumes],
                      "cells": [[[round(float(x), 4) for x in row] for row in f.cell] for f in frames],
                      "frames": [_pos(f.get_positions()) for f in frames],
                      **{k: [_r(x, 5) for x in v] for k, v in (extra or {}).items()}})
    if md:
        anims.append({"kind": "dynamics", "name": f"atoms at {_r((values or {}).get('T_K', 0), 4)} K",
                      "frame_fs": _r(md_fs / max(len(md), 1), 4), "frames": md})
    v = values or {}
    data = {"manifest": {"kind": "crystal", "version": VERSION, "generator": "wobbl chem",
                         "frames": sum(len(a["frames"]) for a in anims),
                         "units": {"positions": "angstrom", "cell": "angstrom, rows are the lattice vectors",
                                   "energies_eV_per_atom": "eV per atom above the minimum", "frame_fs": "fs"},
                         "notes": "Periodic: the cell repeats in every direction. Equation of state frames are the "
                                  "computed volumes, dynamics frames true positions."},
            "crystal": {"name": name, "structure": desc, "method": v.get("method")},
            "cell": [[round(float(x), 4) for x in row] for row in atoms.cell], "pbc": [True, True, True],
            "atoms": atoms_block(atoms.get_chemical_symbols()), "positions": _pos(atoms.get_positions()),
            "bonds": [], "animations": anims,
            "properties": {k: _r(v[k], 5) for k in ("lattice_A", "density_g_cm3", "bulk_modulus_GPa",
                                                     "cohesive_eV_atom", "drift_meV_atom", "fusion_eV_atom",
                                                     "melting_K") if v.get(k) is not None}}
    return _write(out, data)
