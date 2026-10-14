"""Mixtures of gases: their equilibrium at a temperature and pressure, and their flame at the mixture's enthalpy.

Any gases NASA's database lists can be mixed, in the amounts written, in equal moles, or as fuels with the oxygen
or air they need for an equivalence ratio. Cantera finds the equilibrium over every gas and condensed species
made of the mixture's elements. A flame is that equilibrium at the reactants' enthalpy and pressure, so its
temperature already counts dissociation, and GRI-Mech 3.0's independent data check it where they can.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .resolve import Query, ResolveError, species
from .thermodata import lookup

AIR_N2 = 3.76


def amounts(q: Query, phi: float = 1.0) -> dict:
    """Moles of each component, by formula."""
    if q.kind == "reaction":
        return {t.species.formula: t.coef for t in q.reactants}
    a = dict(q.amounts)
    fuels = a.pop("_fuel", None)
    ox = a.pop("_oxidizer", None)
    if fuels:
        need = 0.0
        out = {}
        for f in fuels:
            sp = next(s for s in q.species if s.formula == f)
            c, h, o = (sp.counts.get(e, 0) for e in ("C", "H", "O"))
            need += c + h / 4 - o / 2 + sp.counts.get("S", 0)
            out[f] = 1.0
        if need <= 0:
            raise ResolveError("the fuel needs no oxygen, so an equivalence ratio does not apply")
        out["O2"] = out.get("O2", 0.0) + need / phi
        if ox == "air":
            out["N2"] = out.get("N2", 0.0) + AIR_N2 * need / phi
        return out
    return a or {s.formula: 1.0 for s in q.species}


def run_mixture(q: Query, opts, out_dir, log=None, options: dict | None = None, flame: bool = True) -> GenericOutcome:
    t0 = time.time()
    log = log or (lambda stage, msg="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    options = options or {}
    from .install import require
    from .thermo import Equilibrium, Reactants
    ct = require()
    T, P = opts.temperature, opts.pressure
    phi = float(options.get("phi", 1.0))
    mol = amounts(q, phi)
    by_formula = {s.formula: s for s in q.species}
    els, h, mass, rows = {}, 0.0, 0.0, []
    for f, n in mol.items():
        sp = by_formula.get(f) or species(f, allow_online=False)
        d = lookup(sp, "g", T)
        if d is None:
            raise ResolveError(f"{sp.label()} has no gas phase data in NASA's tables, so it cannot join a gas mixture",
                               "Mixtures take gases such as fuels, oxygen, nitrogen, steam and CO2.")
        h += n * d.H(T)
        mass += n * sp.molar_mass
        for e, c in sp.counts.items():
            els[e] = els.get(e, 0.0) + n * c
        rows.append((f, n, d.name))
    kg = mass / 1000.0
    total = sum(mol.values())
    label = " + ".join(f"{n:g} {f}" for f, n in mol.items())
    eq = Equilibrium(Reactants({e: n / kg for e, n in els.items()}, h / kg, label))
    log("equilibrium", label)
    st_tp = eq.tp(T, P)
    st = eq.hp(P, T0=2200.0) if flame else st_tp
    el = eq.element_error(st)
    en = abs(st.h - eq.r.h0) / (st.cp * st.T) if flame else 0.0
    checks = [Check("element balance", "pass" if el < 1e-8 else "fail",
                    f"every element of the mixture is in the products to {el:.1g}")]
    if flame:
        checks.append(Check("energy balance", "pass" if en < 1e-6 else "fail",
                            f"the products at {fmt(st.T)} K hold the mixture's enthalpy to {en:.1g} of cp T"))
        gri = _gri_check(ct, mol, T, P, st.T)
        if gri:
            checks.append(gri)
    level, reasons = confidence(checks)
    mole_frac = {f: n / total for f, n in mol.items()}
    values = {"mixture": mole_frac, "moles": mol, "temperature_K": T, "pressure_Pa": P, "phi": phi if
              "_fuel" in q.amounts else None, "products": st.major(1e-4), "condensed": st.condensed}
    if flame:
        values["T_ad_K"] = st.T
        head = (f"The mixture {label} burns at {fmt(st.T)} K at {fmt(P / 1000)} kPa, starting from {fmt(T)} K, with "
                f"{_top(st)} in its flame at equilibrium.")
        metrics = [("flame temperature", f"{fmt(st.T)} K"), ("pressure", f"{fmt(P / 1000)} kPa")]
    else:
        head = f"At {fmt(T)} K and {fmt(P / 1000)} kPa the mixture {label} settles to {_top(st)} at equilibrium."
        metrics = [("temperature", f"{fmt(T)} K"), ("pressure", f"{fmt(P / 1000)} kPa")]
    sweep = []
    for x in np.linspace(max(300.0, min(T, 600.0)), max(3000.0, st.T + 200), 14):
        s = eq.tp(float(x), P)
        sweep.append({"T_K": round(float(x), 1), "x": {k: round(v, 5) for k, v in s.major(1e-3).items()}})
    values["sweep"] = sweep
    imgs = {}
    from . import playback
    values["files"] = {"playback": playback.mixture(out, label, values)}
    assumptions = [f"The gases {', '.join(f for f in mol)} from NASA's tables, ideal and fully mixed, at "
                   f"{fmt(T)} K and {fmt(P / 1000)} kPa.",
                   "Equilibrium over every gas and condensed species of the mixture's elements in NASA's database. "
                   "A real flame loses heat and runs below its equilibrium temperature."]
    o = GenericOutcome("chemistry", label, opts.question, "ok" if level != "low" else "marginal", head, metrics,
                       level, reasons, checks, assumptions, values, imgs,
                       solver=f"Cantera {ct.__version__} on NASA Glenn data", t_total=time.time() - t0)
    return o.write(out)


def _top(st) -> str:
    parts = [f"{100 * v:.3g}% {k}" for k, v in list(st.major(1e-3).items())[:4]]
    return ", ".join(parts)


def _gri_check(ct, mol: dict, T: float, P: float, t_nasa: float) -> Check | None:
    """The same flame on GRI-Mech 3.0's own thermodynamic data, when every component is in it."""
    g = ct.Solution("gri30.yaml")
    names = {s.upper(): s for s in g.species_names}
    x = {}
    for f, n in mol.items():
        nm = names.get(f.upper())
        if nm is None:
            return None
        x[nm] = n
    g.TPX = T, P, x
    g.equilibrate("HP")
    d = abs(g.T - t_nasa) / t_nasa
    return Check("thermo data", "pass" if d < 0.005 else ("warn" if d < 0.02 else "fail"),
                 f"GRI-Mech 3.0's own data give {fmt(g.T)} K for the same flame, {fmt(abs(g.T - t_nasa))} K from "
                 f"NASA's ({100 * d:.2g}% apart)", {"T_gri30_K": g.T})
