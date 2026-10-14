"""A molecule run: its 3D structure, energy, dipole, charges, orbitals, IR spectrum, thermochemistry,
conformers, solvation and motion at a temperature, each checked.

RDKit builds the conformers, GFN2-xTB ranks and refines them, and the lowest is optimized until the largest force
is under 0.003 eV/A. Its Hessian gives the normal modes, which must all be real for a minimum (an imaginary one
is followed downhill and the structure optimized again), the IR spectrum and the thermochemistry. ALPB gives the
free energy of solvation, and a short Langevin run shows the molecule moving at the temperature, with a separate
constant energy run at two time steps to check the integration. Small molecules are optimized again with B3LYP
DFT through PySCF, as a check at normal accuracy and as the answer at high accuracy, and molecules NIST has
measured are compared with experiment.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from . import reference as REF
from .structure import Structure
from .thermochem import EV_TO_KJ, rrho
from .vib import Modes, hessian, normal_modes
from .xtb import XTB, SCFError

FMAX = 0.003            # eV/A, the optimization's force threshold
IMAG = -20.0            # cm-1, below this a mode counts as imaginary


@dataclass
class Options:
    temperature: float = 298.15
    pressure: float = 101325.0
    solvent: str = ""
    accuracy: str = "normal"           # fast, normal or high
    question: str = ""
    md: bool | None = None             # None decides from the size and accuracy
    md_fs: float = 400.0
    dft: bool | None = None            # None decides from the size and accuracy
    render: bool = True


@dataclass
class Opt:
    positions: np.ndarray
    energy: float
    fmax: float
    steps: int
    converged: bool


def _ase():
    from .install import require_pkg
    return require_pkg("ase")


def optimize(engine, symbols: list[str], positions: np.ndarray, fmax: float = FMAX, steps: int = 400) -> Opt:
    _ase()
    from ase import Atoms
    from ase.optimize import BFGS, FIRE
    at = Atoms(symbols, positions=np.asarray(positions, float))
    at.calc = engine.ase()
    opt = BFGS(at, logfile=None, maxstep=0.15)
    ok = opt.run(fmax=fmax, steps=steps)
    n = opt.nsteps
    if not ok:           # BFGS can stall on flat surfaces, where FIRE keeps going
        opt2 = FIRE(at, logfile=None)
        ok = opt2.run(fmax=fmax, steps=steps)
        n += opt2.nsteps
    f = np.linalg.norm(at.get_forces(), axis=1).max()
    return Opt(at.get_positions(), float(at.get_potential_energy()), float(f), int(n), bool(ok))


def wants(q: str, *words) -> bool:
    return any(re.search(w, q or "", re.I) for w in words)


def run_molecule(st: Structure, name: str, formula: str, opts: Options, out_dir, extra: dict | None = None,
                 log=None) -> GenericOutcome:
    t0 = time.time()
    log = log or (lambda stage, msg="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fast, high = opts.accuracy == "fast", opts.accuracy == "high"
    T, P = opts.temperature, opts.pressure
    syms = st.symbols
    n = len(syms)
    checks: list[Check] = []
    values: dict = {"name": name, "formula": formula, "charge": st.charge, "multiplicity": st.multiplicity,
                    "atoms": n, "structure_source": st.source, "temperature_K": T, "pressure_Pa": P}
    eng = XTB(st.numbers, st.charge, st.uhf)

    # conformers: GFN2-xTB ranks RDKit's, the lowest goes on
    log("conformers", f"{len(st.conformers) or 1} from {st.source.split(',')[0]}")
    ranked = _conformers(st, eng, 2 if fast else 8 if high else 5, T)
    values["conformers"] = [{"energy_kJ_mol": e, "population": p} for e, p, _ in ranked]
    pos = ranked[0][2] if ranked else st.positions

    log("optimize", f"{n} atoms with {eng.label}")
    o = Opt(pos, eng.compute(pos).energy, 0.0, 0, True) if n == 1 else optimize(eng, syms, pos, FMAX)
    checks.append(_convergence_check(o, eng.method))
    modes, pos, minimum = _minimum(eng, st, o.positions, log)
    if minimum:
        checks.append(minimum)
    res = eng.compute(pos)
    checks.append(Check("SCF convergence", "pass", f"all {eng.calls} {eng.method} single points converged "
                                                   "within 300 iterations", {"calls": eng.calls}))
    xtb_pos, xtb_modes, xtb_res = pos, modes, res

    # the method ladder: DFT checks GFN2-xTB on small molecules, and at high accuracy it is the answer
    method, ladder = eng.method, None
    role = _dft_role(st, opts)
    if role:
        d = _dft_stage(st, pos, role, opts, log)
        ladder = _compare(st, xtb_pos, xtb_modes, xtb_res, d)
        if role == "primary" and "error" not in d:
            pos, modes, res, method = d["positions"], d["modes"], d["result"], d["method"]
            im = modes.imaginary(IMAG)
            checks.append(Check("SCF convergence", "pass", f"every {method} self consistent field converged",
                                {"method": method}))
            if im:
                checks.append(Check("minimum", "warn", f"{method} leaves {len(im)} imaginary frequencies at its "
                                                       "minimum", {"imaginary": [float(modes.freqs[i]) for i in im]}))
        values["ladder"] = {k: v for k, v in ladder.items() if k not in ("modes", "result")}
        checks.append(_ladder_check(ladder, role))
    pos = np.asarray(pos)

    homo, lumo = res.homo_lumo()
    dip = float(np.linalg.norm(res.dipole))
    values.update({"positions": pos.tolist(), "symbols": syms, "bonds": [list(b) for b in st.bonds],
                   "energy_eV": res.energy, "energy_Eh": res.energy / 27.211386245988, "dipole_D": dip,
                   "charges": [round(float(c), 4) for c in res.charges], "homo_eV": homo, "lumo_eV": lumo,
                   "gap_eV": (lumo - homo) if homo is not None and lumo is not None else None,
                   "method": method})
    scale = REF.FREQ_SCALE.get(method, 1.0)
    if modes is not None:
        values["frequencies_cm"] = [round(float(f), 1) for f in modes.freqs]
        values["frequency_scale"] = scale
        values["ir_km_mol"] = [round(float(i), 2) for i in modes.intensities]
        values["mode_labels"] = modes.labels

    th = rrho(modes.freqs if modes is not None else np.zeros(0), pos, syms, T, P, st.multiplicity)
    values["thermo"] = th.to_dict()
    values["G_total_eV"] = res.energy + th.G / EV_TO_KJ
    soft = int(sum(1 for f in (modes.freqs if modes is not None else []) if 0 < f < 100))
    checks.append(Check("thermochemistry", "info",
                        f"ideal gas at {fmt(T)} K and {fmt(P / 1000)} kPa with a rotational symmetry number of "
                        f"{th.sigma}" + (f", and {soft} soft modes under 100 cm-1 treated as hindered rotors"
                                          if soft else ""), {"sigma": th.sigma}))

    # solvation, always GFN2-xTB with ALPB at the GFN2-xTB minimum
    solvent = opts.solvent or "water"
    if n > 1 or st.charge:
        try:
            es = XTB(st.numbers, st.charge, st.uhf, solvent=solvent).compute(xtb_pos).energy
            values["solvation_kJ_mol"] = (es - xtb_res.energy) * EV_TO_KJ
            values["solvent"] = solvent
        except Exception as e:  # noqa: BLE001  a solvent ALPB lacks must not stop the run
            values["solvation_error"] = str(e)[:120]

    ref = REF.molecule(formula)
    if ref:
        checks.append(_reference_check(ref, syms, pos, modes, dip, method))

    md = None
    asked = wants(opts.question, r"dynamic", r"\bmd\b", r"\bmov", r"vibrat\w* at", r"jiggl", r"trajector")
    if opts.md if opts.md is not None else (1 < n and (asked or (not fast and n <= 12) or (high and n <= 40))):
        log("dynamics", f"{fmt(opts.md_fs)} fs at {fmt(T)} K")
        md = _dynamics(st, xtb_pos, eng, T, opts.md_fs)
        values["md"] = {k: v for k, v in md.items() if k != "frames"}
        checks += _md_checks(md)

    level, reasons = confidence([c for c in checks if c.status != "info"])
    acc = REF.ACCURACY.get(method, REF.ACCURACY["GFN2-xTB"])
    head, metrics = _headline(name, formula, values, modes, scale, opts)
    imgs = {}
    if opts.render and modes is not None and len(modes.freqs):
        try:
            from .charts import ir_chart
            imgs["ir"] = ir_chart(out / "images" / "ir.png", modes.freqs * scale, modes.intensities, modes.labels)
            values["_figures"] = [["ir", f"Infrared spectrum from {method}" +
                                   (f", frequencies scaled by {scale}" if scale != 1 else "") +
                                   ", broadened by 20 cm-1"]]
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    from . import playback
    values["files"] = {"playback": playback.molecule(out, name, formula, syms, pos, st.bonds, modes,
                                                     md["frames"] if md else None, values)}
    assumptions = [f"{name} as {st.source}, refined with {method}, good to about {acc.get('bond', 0.02)} A in bond "
                   f"lengths, {acc.get('angle', 2)} degrees in angles and {100 * acc.get('freq', 0.08):.0f}% in "
                   "frequencies." + ("" if method == eng.method else " GFN2-xTB found the conformers and ran the "
                                     "dynamics and solvation."),
                   "One molecule in the gas phase unless a solvent is named, and a solvent is a continuum (ALPB) "
                   "rather than explicit molecules.",
                   "Thermochemistry treats the molecule as an ideal gas, a rigid rotor and harmonic oscillators, "
                   "with modes under 100 cm-1 blended into free rotors.",
                   f"The HOMO-LUMO gap from {method} is {acc.get('gap', 'a trend only')}, so it is not an optical "
                   "gap."]
    if extra:
        values.update(extra)
    o = GenericOutcome("chemistry", name, opts.question, "ok" if level != "low" else "marginal", head, metrics,
                       level, reasons, checks, assumptions, values, imgs, solver=_solver(ladder, eng, method),
                       t_total=time.time() - t0)
    return o.write(out, [_table(values, modes, scale)])


def _convergence_check(o: Opt, method: str) -> Check:
    if o.converged and o.fmax < FMAX * 1.01:
        return Check("geometry convergence", "pass", f"{method} stopped after {o.steps} steps with the largest "
                                                     f"force at {o.fmax:.4f} eV/A, under the {FMAX} eV/A it aims for",
                     {"fmax": o.fmax, "steps": o.steps})
    return Check("geometry convergence", "warn", f"the optimization stopped after {o.steps} steps with forces still "
                                                 f"at {o.fmax:.4f} eV/A", {"fmax": o.fmax, "steps": o.steps})


def _minimum(eng, st: Structure, pos, log):
    """Normal modes at the optimized structure, following any imaginary mode downhill up to twice."""
    if len(st.symbols) == 1:
        return None, pos, None
    log("vibrations", f"Hessian from {6 * len(st.symbols)} force calls")
    followed = 0
    for _ in range(3):
        H, dmu = hessian(eng, pos)
        modes = normal_modes(H, dmu, pos, st.masses, st.symbols, st.bonds)
        im = modes.imaginary(IMAG)
        if not im or followed == 2:
            break
        followed += 1
        # step both ways along the mode until the largest atom moves 0.2 A, and keep the lower minimum
        v = modes.vectors[im[0]] / np.linalg.norm(modes.vectors[im[0]], axis=1).max() * 0.2
        tries = [optimize(eng, st.symbols, pos + s * v, FMAX / 3) for s in (1, -1)]
        pos = min(tries, key=lambda o: o.energy).positions
    im = modes.imaginary(IMAG)
    if im:
        return modes, pos, Check("minimum", "warn", f"{len(im)} imaginary frequencies remain, the lowest at "
                                                    f"{modes.freqs[im[0]]:.0f} cm-1, so this is a saddle point",
                                 {"imaginary": [float(modes.freqs[i]) for i in im]})
    soft = [f for f in modes.freqs if IMAG <= f < 0]
    return modes, pos, Check(
        "minimum", "pass", f"all {len(modes.freqs)} frequencies are real, so the structure is a true minimum"
        + (f", after following {followed} imaginary mode downhill" if followed else "")
        + (f" ({len(soft)} within {abs(IMAG):.0f} cm-1 of zero, numerical noise)" if soft else ""),
        {"followed": followed})


def _solver(ladder, eng, method) -> str:
    from .install import load_pkg
    tb = load_pkg("tblite")
    s = f"{eng.method} with tblite {getattr(tb, '__version__', '')}, RDKit conformers and ASE"
    if ladder and "error" not in ladder:
        return f"{method} in PySCF, starting from {s}" if method != eng.method else \
            f"{s}, checked with {ladder['method']} in PySCF"
    return s


def _conformers(st: Structure, eng: XTB, keep: int, T: float):
    """[(relative energy kJ/mol, Boltzmann population, positions)] of the lowest GFN2-xTB conformers."""
    confs = st.conformers or [(0.0, st.positions)]
    if len(st.symbols) == 1:
        return [(0.0, 1.0, st.positions)]
    out = []
    for _, pos in confs[:max(keep, 1)]:
        try:
            o = optimize(eng, st.symbols, pos, fmax=0.02, steps=200)
        except SCFError:
            continue
        if any(np.sqrt(((_align(o.positions, p) - p) ** 2).sum(1).mean()) < 0.05 for _, p in out):
            continue
        out.append((o.energy, o.positions))
    if not out:
        return [(0.0, 1.0, st.positions)]
    e0 = min(e for e, _ in out)
    rel = [(e - e0) * EV_TO_KJ for e, _ in out]
    w = np.exp(-np.array(rel) * 1000 / (8.314462618 * T))
    w /= w.sum()
    return [(float(rel[i]), float(w[i]), out[i][1]) for i in np.argsort(rel)]


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    return a0 @ rot.T + b.mean(0)


# --------------------------------------------------------------------------------------------- method ladder
def _dft_role(st: Structure, opts: Options) -> str:
    """'primary' when DFT is the answer, 'check' when it only checks GFN2-xTB, '' when the molecule is too big."""
    heavy, n = st.heavy(), len(st.symbols)
    if n < 2 or opts.dft is False:
        return ""
    if opts.accuracy == "high" and heavy <= 8 and n <= 20:
        return "primary"
    if opts.dft or (opts.accuracy == "normal" and heavy <= 3 and n <= 8):
        return "check"
    return ""


def _dft_stage(st: Structure, pos, role: str, opts: Options, log) -> dict:
    """B3LYP from the GFN2-xTB minimum: optimized, then its Hessian, analytic for a check or by finite differences
    of forces when the IR intensities are wanted too."""
    try:
        from .dft import DFT
        d = DFT(st.symbols, st.charge, st.multiplicity, "high" if opts.accuracy == "high" else "normal")
    except Exception as e:  # noqa: BLE001  PySCF missing or failing must not stop the GFN2-xTB answer
        return {"error": str(e)[:160]}
    log("method ladder", f"{d.label} for {len(st.symbols)} atoms, as the {'answer' if role == 'primary' else 'check'}")
    try:
        o = optimize(d, st.symbols, pos, fmax=0.005, steps=60)
        if role == "primary":
            H, dmu = hessian(d, o.positions, step=0.005)
        else:
            d.compute(o.positions)
            H, dmu = analytic_hessian(d), np.zeros((3 * len(st.symbols), 3))
        md = normal_modes(H, dmu, o.positions, st.masses, st.symbols, st.bonds)
        r = d.compute(o.positions)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:160], "method": d.label}
    return {"method": d.label, "positions": o.positions, "converged": o.converged, "modes": md, "result": r}


def analytic_hessian(d) -> np.ndarray:
    """PySCF's analytic Hessian at the last geometry, as a 3n by 3n matrix in eV/A^2."""
    from .xtb import BOHR, HARTREE
    h = d.mf.Hessian().kernel()
    n = h.shape[0]
    return h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * HARTREE / BOHR ** 2


def _bond_lengths(symbols, pos, bonds) -> dict:
    return {f"{symbols[i]}{i + 1}-{symbols[j]}{j + 1}": float(np.linalg.norm(pos[i] - pos[j])) for i, j, _ in bonds}


def _compare(st: Structure, pos, modes: Modes | None, res, d: dict) -> dict:
    if "error" in d:
        return d
    bx = _bond_lengths(st.symbols, pos, st.bonds)
    bd = _bond_lengths(st.symbols, d["positions"], st.bonds)
    db = max((abs(bx[k] - bd[k]) for k in bx), default=0.0)
    df = None
    md = d["modes"]
    if modes is not None and len(md.freqs) == len(modes.freqs):
        sel = md.freqs > 100
        df = float(np.mean(np.abs(modes.freqs[sel] - md.freqs[sel]) / md.freqs[sel])) if sel.any() else None
    dd = float(np.linalg.norm(d["result"].dipole))
    return {"method": d["method"], "positions": np.asarray(d["positions"]).tolist(), "converged": d["converged"],
            "bonds": bd, "max_bond_diff_A": db, "frequencies_cm": [round(float(f), 1) for f in md.freqs],
            "mean_freq_diff": df, "dipole_D": dd, "dipole_diff_D": abs(dd - float(np.linalg.norm(res.dipole))),
            "modes": md, "result": d["result"]}


def _ladder_check(ld: dict, role: str) -> Check:
    if "error" in ld:
        return Check("method ladder", "skip", f"the DFT rung could not run ({ld['error']})",
                     {k: v for k, v in ld.items() if k != "modes"})
    acc = REF.ACCURACY["GFN2-xTB"]
    parts = [f"{ld['method']} and GFN2-xTB differ by up to {ld['max_bond_diff_A']:.3f} A in the bonds"]
    if ld.get("mean_freq_diff") is not None:
        parts.append(f"{100 * ld['mean_freq_diff']:.1f}% in the frequencies on average")
    parts.append(f"and {ld['dipole_diff_D']:.2f} D in the dipole")
    ok = ld["max_bond_diff_A"] <= 2 * acc["bond"] and (ld.get("mean_freq_diff") or 0) <= 2 * acc["freq"]
    tail = (f", {'within' if ok else 'beyond'} what GFN2-xTB is known to miss by (about {acc['bond']} A and "
            f"{100 * acc['freq']:.0f}%)")
    if role == "primary":
        tail += f", and {ld['method']} is the answer here"
    detail = {k: v for k, v in ld.items() if k not in ("modes", "result")}
    return Check("method ladder", "pass" if ok else "warn", ", ".join(parts) + tail, detail)


def _reference_check(ref: dict, syms, pos, modes: Modes | None, dip: float, method: str) -> Check:
    """Bond lengths, angles, frequencies and dipole against NIST's measured values, for the answer's method."""
    acc = REF.ACCURACY.get(method, REF.ACCURACY["GFN2-xTB"])
    worst, lines, detail = 0.0, [], {"method": method}
    for a, b, r0 in ref.get("bonds", []):
        rs = [np.linalg.norm(pos[i] - pos[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))
              if sorted((syms[i], syms[j])) == sorted((a, b)) and np.linalg.norm(pos[i] - pos[j]) < r0 * 1.25]
        if rs:
            r = float(np.mean(rs))
            worst = max(worst, abs(r - r0) / acc["bond"])
            lines.append(f"{a}-{b} {r:.3f} A against {r0:.3f}")
            detail[f"{a}-{b}"] = [r, r0]
    for a, c, b, t0 in ref.get("angles", []):
        ang = _angle(syms, pos, a, c, b)
        if ang is not None:
            worst = max(worst, abs(ang - t0) / acc["angle"])
            lines.append(f"{a}-{c}-{b} {ang:.1f} degrees against {t0:.1f}")
            detail[f"{a}-{c}-{b}"] = [ang, t0]
    freqs = modes.freqs if modes is not None else np.zeros(0)
    if "freqs" in ref and len(freqs) == len(ref["freqs"]):
        scale = REF.FREQ_SCALE.get(method, 1.0)
        f = np.sort(freqs) * scale
        want = np.array(sorted(ref["freqs"]), float)
        worst = max(worst, float((np.abs(f - want) / want).max()) / acc["freq"])
        lines.append(f"frequencies {', '.join(str(round(x)) for x in f)} cm-1"
                     + (f" scaled by {scale}" if scale != 1 else "") + f" against {', '.join(map(str, ref['freqs']))}")
        detail["freqs"] = [f.tolist(), ref["freqs"]]
    if ref.get("dipole", 0) > 0.2:
        worst = max(worst, abs(dip - ref["dipole"]) / acc["dipole"])
        lines.append(f"dipole {dip:.2f} D against {ref['dipole']:.2f}")
        detail["dipole"] = [dip, ref["dipole"]]
    # typical errors are averages, so a miss up to twice one is still ordinary
    status = "pass" if worst <= 2.0 else "warn" if worst <= 4.0 else "fail"
    return Check("reference data", status, f"{method} against NIST CCCBDB measurements, " + "; ".join(lines)
                 + f" (the largest miss is {worst:.1f} times the method's typical error)", detail)


def _angle(syms, pos, a, c, b):
    best = None
    for ic, s in enumerate(syms):
        if s != c:
            continue
        near = [i for i in range(len(syms)) if i != ic and np.linalg.norm(pos[i] - pos[ic]) < 1.9]
        for i in near:
            for j in near:
                if i < j and sorted((syms[i], syms[j])) == sorted((a, b)):
                    u, v = pos[i] - pos[ic], pos[j] - pos[ic]
                    cos = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
                    best = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return best


# --------------------------------------------------------------------------------------------- dynamics
def _dynamics(st: Structure, pos, eng: XTB, T: float, fs_total: float) -> dict:
    """A Langevin run at the temperature for the motion, then constant energy runs at two time steps from the same
    start for the integration check."""
    _ase()
    from ase import Atoms, units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet

    rng = np.random.default_rng(11)
    at = Atoms(st.symbols, positions=pos)
    at.calc = eng.ase()
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
    Stationary(at)
    ZeroRotation(at)
    start = (at.get_positions().copy(), at.get_velocities().copy())
    at.set_constraint(FixCom())
    dt = 0.5
    steps = int(fs_total / dt)
    frames, temps = [], []
    dyn = Langevin(at, timestep=dt * units.fs, temperature_K=T, friction=0.02 / units.fs, rng=rng, fixcm=False)
    every = max(1, steps // 120)
    for k in range(steps):
        dyn.run(1)
        if k % every == 0:
            frames.append(at.get_positions().round(3).tolist())
        temps.append(at.get_temperature())
    nve = {}
    for d in (dt, dt / 2):
        b = Atoms(st.symbols, positions=start[0])
        b.set_velocities(start[1])
        b.calc = eng.ase()
        vv = VelocityVerlet(b, timestep=d * units.fs)
        es = []
        for _ in range(int(25.0 / d)):
            vv.run(1)
            es.append(b.get_potential_energy() + b.get_kinetic_energy())
        es = np.array(es)
        ke = 1.5 * len(st.symbols) * units.kB * T
        nve[str(d)] = {"spread_eV": float(es.max() - es.min()), "drift_eV": float(es[-1] - es[0]), "kinetic_eV": ke}
    return {"frames": frames, "dt_fs": dt, "fs": fs_total, "T_mean": float(np.mean(temps[len(temps) // 3:])),
            "T_target": T, "nve": nve}


def _md_checks(md: dict) -> list[Check]:
    a, b = md["nve"][str(md["dt_fs"])], md["nve"][str(md["dt_fs"] / 2)]
    rel = a["spread_eV"] / a["kinetic_eV"]
    ratio = a["spread_eV"] / max(b["spread_eV"], 1e-12)
    out = [Check("energy conservation", "pass" if rel < 0.05 else "warn",
                 f"a constant energy run of 25 fs keeps the total energy within {a['spread_eV'] * 1000:.2f} meV, "
                 f"{100 * rel:.2f}% of the kinetic energy", a),
           Check("time step convergence", "pass" if ratio > 2.0 or b["spread_eV"] < 1e-4 else "warn",
                 f"halving the {md['dt_fs']} fs step shrinks the energy wobble {ratio:.1f} times, close to the 4 "
                 "times a second order integrator should give" if ratio > 2.0 else
                 f"halving the step changed the energy wobble only {ratio:.1f} times", {"ratio": ratio})]
    t = md["T_mean"]
    out.append(Check("thermostat", "pass" if abs(t - md["T_target"]) < 0.25 * md["T_target"] else "info",
                     f"the Langevin run averaged {fmt(t)} K against the {fmt(md['T_target'])} K asked for, and a "
                     "molecule this small fluctuates around it", {"T_mean": t}))
    return out


# --------------------------------------------------------------------------------------------- answer
def _headline(name, formula, v, modes, scale, opts):
    from .formula import molar_mass, parse
    M = molar_mass(parse(formula)[0])
    q = opts.question
    lines = {}
    if modes is not None and len(modes.freqs):
        k = int(np.argmax(modes.intensities))
        lines["ir"] = (f"its strongest IR band is at {modes.freqs[k] * scale:.0f} cm-1, the "
                       f"{modes.labels[k] or 'mixed mode'}")
    lines["dipole"] = f"its dipole is {v['dipole_D']:.2f} D"
    if v.get("gap_eV") is not None:
        lines["gap"] = f"its HOMO-LUMO gap is {v['gap_eV']:.2f} eV"
    if "solvation_kJ_mol" in v:
        lines["solv"] = f"dissolving in {v['solvent']} is worth {v['solvation_kJ_mol']:.1f} kJ/mol"
    order = ["ir", "dipole", "solv"]
    if wants(q, "gap", "homo", "lumo", "orbital", "colou?r", "reactiv"):
        order = ["gap"] + order
    if wants(q, "solv", "dissolv", "solub", "in water"):
        order = ["solv"] + order
    if wants(q, "dipole", "polar"):
        order = ["dipole"] + order
    picked = list(dict.fromkeys(k for k in order if k in lines))[:3]
    th = v["thermo"]
    head = f"{name} ({formula}, {M:.2f} g/mol), by {v['method']}: " + ", ".join(lines[k] for k in picked) + "."
    head += (f" At {fmt(th['T_K'])} K its entropy is {th['S_J_molK']:.1f} J/(mol K) and its heat capacity "
             f"{th['Cp_J_molK']:.1f} J/(mol K).")
    if v.get("md"):
        head += f" A {fmt(v['md']['fs'])} fs run shows it moving at {fmt(v['md']['T_mean'])} K."
    metrics = [("dipole", f"{v['dipole_D']:.2f} D")]
    if "ir" in lines:
        k = int(np.argmax(modes.intensities))
        metrics.append(("strongest IR band", f"{modes.freqs[k] * scale:.0f} cm-1"))
    if v.get("gap_eV") is not None:
        metrics.append(("HOMO-LUMO gap", f"{v['gap_eV']:.2f} eV"))
    metrics.append((f"entropy at {fmt(th['T_K'])} K", f"{th['S_J_molK']:.1f} J/(mol K)"))
    return head, metrics


def _table(v: dict, modes: Modes | None, scale: float) -> str:
    from ..report.html import table
    rows = [["energy", f"{v['energy_Eh']:.6f} Eh ({v['method']})"], ["dipole", f"{v['dipole_D']:.3f} D"]]
    if v.get("homo_eV") is not None:
        rows.append(["HOMO and LUMO", f"{v['homo_eV']:.2f} eV and {v['lumo_eV']:.2f} eV" if v.get("lumo_eV")
                     is not None else f"{v['homo_eV']:.2f} eV"])
    th = v["thermo"]
    rows += [["zero point energy", f"{th['zpe_kJ_mol']:.2f} kJ/mol"],
             [f"H, S and G at {fmt(th['T_K'])} K", f"{th['H_corr_kJ_mol']:.2f} kJ/mol, {th['S_J_molK']:.2f} "
                                                   f"J/(mol K) and {th['G_corr_kJ_mol']:.2f} kJ/mol above the "
                                                   "electronic energy"],
             ["heat capacity Cp", f"{th['Cp_J_molK']:.2f} J/(mol K)"]]
    if "solvation_kJ_mol" in v:
        rows.append([f"solvation in {v['solvent']}", f"{v['solvation_kJ_mol']:.1f} kJ/mol"])
    if len(v.get("conformers", [])) > 1:
        rows.append(["conformers", ", ".join(f"{c['energy_kJ_mol']:.1f} kJ/mol ({100 * c['population']:.0f}%)"
                                             for c in v["conformers"][:4])])
    html = "<h2>Properties</h2>" + table(["", ""], rows)
    if modes is not None and len(modes.freqs):
        top = np.argsort(-modes.intensities)[:12]
        rows = [[f"{modes.freqs[i] * scale:.0f}", f"{modes.intensities[i]:.1f}", modes.labels[i]]
                for i in sorted(top, key=lambda i: modes.freqs[i])]
        html += "<h2>Strongest IR bands</h2>" + table(["cm-1", "km/mol", "mode"], rows, {0, 1})
    return html
