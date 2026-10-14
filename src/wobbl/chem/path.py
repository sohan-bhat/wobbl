"""A reaction path and its barrier, by the nudged elastic band with GFN2-xTB.

A path needs the same atoms at both ends, so it applies to one molecule rearranging into another, such as HCN to
HNC. Both ends are optimized, the product's atoms are matched to the reactant's (the assignment that overlays them
best), and a chain of images between them is relaxed with a climbing image until the forces across the path fall
under 0.05 eV/A. The highest image is the transition state, which must have exactly one imaginary frequency. The
barrier carries GFN2-xTB's typical error for barriers, about 30 kJ/mol.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..verify.checks import Check, confidence
from .molecule import optimize
from .reaction import equation
from .resolve import Query, ResolveError
from .structure import from_species
from .thermochem import EV_TO_KJ
from .vib import hessian, normal_modes
from .xtb import XTB


def _map(ref_syms, ref_pos, syms, pos) -> np.ndarray:
    """The order of the product's atoms that overlays them on the reactant's best."""
    from scipy.optimize import linear_sum_assignment
    best, best_rmsd = None, np.inf
    a = ref_pos - ref_pos.mean(0)
    for flip in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)):
        b = (pos - pos.mean(0)) * np.array(flip)
        order = np.arange(len(syms))
        for _ in range(6):
            order = np.empty(len(syms), int)
            for el in set(ref_syms):
                i = [k for k, s in enumerate(ref_syms) if s == el]
                j = [k for k, s in enumerate(syms) if s == el]
                cost = np.linalg.norm(a[i][:, None] - b[j][None], axis=2)
                r, c = linear_sum_assignment(cost)
                for x, y in zip(r, c):
                    order[i[x]] = j[y]
            u, _, vt = np.linalg.svd(b[order].T @ a)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1, 1, d]) @ u.T
            b = b @ rot.T
        rmsd = np.sqrt(((a - b[order]) ** 2).sum(1).mean())
        if rmsd < best_rmsd:
            best, best_rmsd = order, rmsd
    return best


def _bulge(images) -> None:
    """Where the straight path runs an atom through another, as when H moves from C to N along a linear HCN, bow the
    moving atoms out sideways so the band starts on a path that exists."""
    a = images[0].get_positions()
    b = images[-1].get_positions()
    moving = np.linalg.norm(b - a, axis=1) > 0.5
    close = min(min(np.linalg.norm(p[i] - p[j]) for i in range(len(p)) for j in range(i + 1, len(p)))
                for p in (im.get_positions() for im in images[1:-1]))
    if close > 0.8 or not moving.any():
        return
    axis = a[~moving].mean(0) - a[moving].mean(0) if (~moving).any() else b[0] - a[0]
    side = np.cross(axis, [0.0, 0.0, 1.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(axis, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    n = len(images)
    for k, im in enumerate(images[1:-1], start=1):
        p = im.get_positions()
        p[moving] += 1.2 * np.sin(np.pi * k / (n - 1)) * side
        im.set_positions(p)


def run_path(q: Query, opts, out_dir, log=None) -> GenericOutcome:
    t0 = time.time()
    log = log or (lambda stage, msg="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(q.reactants) != 1 or len(q.products) != 1 or q.reactants[0].coef != q.products[0].coef or \
            dict(q.reactants[0].species.counts) != dict(q.products[0].species.counts):
        raise ResolveError("a reaction path needs one molecule rearranging into another with the same atoms",
                           "Ask for a path like 'HCN -> HNC'. For other reactions ask for the heat or equilibrium.")
    from .install import require_pkg
    require_pkg("ase")
    from ase import Atoms
    from ase.mep import NEB
    from ase.optimize import FIRE
    ra, pa = q.reactants[0].species, q.products[0].species
    sr, sp = from_species(ra, 4), from_species(pa, 4)
    log("ends", f"{ra.label()} and {pa.label()} with GFN2-xTB")
    er = XTB(sr.numbers, sr.charge, sr.uhf)
    o1 = optimize(er, sr.symbols, sr.positions)
    ep = XTB(sp.numbers, sp.charge, sp.uhf)
    o2 = optimize(ep, sp.symbols, sp.positions)
    order = _map(sr.symbols, o1.positions, sp.symbols, o2.positions)
    p2 = o2.positions[order]
    n_img = 9
    images = [Atoms(sr.symbols, positions=o1.positions)]
    images += [Atoms(sr.symbols, positions=o1.positions) for _ in range(n_img - 2)]
    images += [Atoms(sr.symbols, positions=p2)]
    for im in images:
        im.calc = XTB(sr.numbers, sr.charge, sr.uhf).ase()
    neb = NEB(images, climb=True, allow_shared_calculator=False)
    neb.interpolate("idpp")
    _bulge(images)
    log("band", f"{n_img} images, climbing image")
    opt = FIRE(neb, logfile=None)
    ok = opt.run(fmax=0.05, steps=800)
    E = np.array([im.get_potential_energy() for im in images])
    fmax = float(max(np.linalg.norm(im.get_forces(), axis=1).max() for im in images[1:-1]))
    k = int(np.argmax(E))
    barrier = (E[k] - E[0]) * EV_TO_KJ
    reverse = (E[k] - E[-1]) * EV_TO_KJ
    dE = (E[-1] - E[0]) * EV_TO_KJ
    checks = [Check("band convergence", "pass" if ok else "warn",
                    f"the band relaxed in {opt.nsteps} steps to forces of {fmax:.3f} eV/A across its images"
                    if ok else f"the band stopped after {opt.nsteps} steps with forces of {fmax:.3f} eV/A", {})]
    ts_ok = None
    if 0 < k < len(E) - 1:
        eng = XTB(sr.numbers, sr.charge, sr.uhf)
        H, dmu = hessian(eng, images[k].get_positions())
        md = normal_modes(H, dmu, images[k].get_positions(), sr.masses, sr.symbols, sr.bonds)
        imag = [f for f in md.freqs if f < -50]
        ts_ok = len(imag) == 1
        checks.append(Check("transition state", "pass" if ts_ok else "warn",
                            f"the highest image has {len(imag)} imaginary frequency"
                            + (f" at {imag[0]:.0f} cm-1, the motion along the path, as a saddle point should"
                               if ts_ok else "ies, so it is near but not at the saddle point"),
                            {"imaginary": imag}))
    else:
        checks.append(Check("transition state", "warn", "the highest point is an end of the band, so the path has "
                                                        "no barrier above the ends", {}))
    level, reasons = confidence(checks)
    eq = equation(q.reactants, q.products, phases=False)
    head = (f"{eq}: GFN2-xTB puts the barrier at {barrier:.0f} kJ/mol from {ra.name} and {reverse:.0f} kJ/mol "
            f"back from {pa.name}, with the product {abs(dE):.0f} kJ/mol {'above' if dE > 0 else 'below'} the "
            "reactant.")
    values = {"barrier_kJ_mol": barrier, "reverse_kJ_mol": reverse, "dE_kJ_mol": dE, "images": n_img,
              "energies_kJ_mol": ((E - E[0]) * EV_TO_KJ).tolist(), "ts_ok": ts_ok}
    from . import playback
    frames = [im.get_positions() for im in images]
    values["files"] = {"playback": playback.molecule(out, eq, ra.formula, sr.symbols, o1.positions, sr.bonds,
                                                     path={"name": "reaction path", "frames": frames,
                                                           "energies": values["energies_kJ_mol"]})}
    imgs = {}
    if opts.render:
        try:
            from .charts import sweep_chart
            imgs["path"] = sweep_chart(out / "images" / "path.png", np.arange(len(E)), values["energies_kJ_mol"],
                                       "image along the path", "energy, kJ/mol", (k, values["energies_kJ_mol"][k]))
            values["_figures"] = [["path", "Energy along the reaction path"]]
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    o = GenericOutcome("chemistry", eq, opts.question, "ok" if level != "low" else "marginal", head,
                       [("barrier", f"{barrier:.0f} kJ/mol"), ("reverse barrier", f"{reverse:.0f} kJ/mol"),
                        ("reaction energy", f"{dE:.0f} kJ/mol")], level, reasons, checks,
                       ["Electronic energies from GFN2-xTB along a nudged elastic band, without zero point energy. "
                        "GFN2-xTB barriers are good to about 30 kJ/mol, so treat this as a first estimate."],
                       values, imgs, solver="GFN2-xTB with tblite and ASE's nudged elastic band",
                       t_total=time.time() - t0)
    return o.write(out)
