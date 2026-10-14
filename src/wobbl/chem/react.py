"""A reaction run: how much heat, whether it goes, how far, how hot it gets and how much gas it makes.

The reaction comes balanced from resolve.py. Each species' enthalpy and entropy at the temperature come from NASA's
tables for gases, solids and liquids and the NBS tables for species in water (thermodata.py), which gives the
reaction enthalpy, entropy and Gibbs energy, the equilibrium constant and the temperature where the reaction
changes direction. A gas reaction's equilibrium mixture comes from Cantera, other reactions' from the equilibrium
constant with ideal activities. Without water the adiabatic temperature is the equilibrium of the products at the
reactants' enthalpy, dissociation and melting included. In water it is the heat over the solution's heat capacity.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .reaction import display, equation
from .resolve import Query, ResolveError, Term
from .thermodata import Data, lookup

R = 8.314462618
CP_WATER = 4.184        # J/(g K)


def _ions_data(t: Term, T: float) -> Data | None:
    """A salt or strong acid in water as the sum of its ions, which is how it exists there."""
    from .reaction import _ion, ion_species, ions_of
    from .substances import ACIDS, ANIONS, STRONG_ACIDS
    sp = t.species
    parts = None
    if sp.kind in ("salt", "oxide"):
        ions = ions_of(sp)
        if ions:
            parts = [(ions[0][0], ion_species(_ion(ions[0]))), (ions[1][0], ion_species(_ion(ions[1])))]
    elif sp.formula in STRONG_ACIDS:
        an, n = ACIDS[sp.formula]
        from .resolve import species
        parts = [(n, species("H+", allow_online=False)), (1, ion_species(ANIONS[an]))]
    if not parts:
        return None
    ds = [(n, lookup(p, "aq", T)) for n, p in parts]
    if any(d is None for _, d in ds):
        return None
    h = sum(n * d.h298 for n, d in ds)
    s = sum(n * d.s298 for n, d in ds)
    cps = [d.cp298 for _, d in ds]
    cp = sum(n * c for (n, _), c in zip(ds, cps)) if all(c is not None for c in cps) else None
    names = " + ".join(f"{n:g} {d.formula}" for n, d in ds)
    return Data(sp.formula, "aq", f"{sp.name} as {names}", ds[0][1].source, h, s, cp, 250.0, 373.0)


def _data(t: Term, T: float) -> Data:
    if t.phase == "aq":
        d = _ions_data(t, T)
        if d is not None:
            return d
    d = lookup(t.species, t.phase, T)
    if d is None and not t.phase:
        d = lookup(t.species, "", T)
    if d is None:
        from .formation import estimate
        d = estimate(t.species, t.phase or "g", T)
    if d is None:
        raise ResolveError(f"no thermochemical data for {display(t.species)}"
                           + (f"({t.phase})" if t.phase else ""),
                           "Give its phase, or ask about a reaction whose species NASA's or the NBS tables list.")
    return d


def _phases(q: Query, T: float) -> None:
    """Fill in each species' phase: as written, or the one stable at the temperature."""
    for t in q.reactants + q.products:
        t.written = bool(t.phase)
        if not t.phase:
            d = lookup(t.species, "", T)
            t.phase = d.phase if d else (t.species.phase or "g")


def thermo(q: Query, T: float) -> dict:
    """Reaction enthalpy, entropy and Gibbs energy at T, per mole of reaction as written, with each species' data."""
    rows, dH, dS, dCp = [], 0.0, 0.0, 0.0
    for sign, terms in ((-1, q.reactants), (1, q.products)):
        for t in terms:
            d = _data(t, T)
            nu = sign * t.coef
            dH += nu * d.H(T)
            dS += nu * d.S(T)
            dCp += nu * d.Cp(T)
            rows.append({"species": display(t.species), "phase": d.phase, "coef": nu, "H_kJ_mol": d.H(T) / 1000,
                         "S_J_molK": d.S(T), "source": d.source, "name": d.name, "covers": d.covers(T)})
    dG = dH - T * dS
    lnK = -dG / (R * T)
    return {"T": T, "dH": dH, "dS": dS, "dG": dG, "dCp": dCp, "lnK": lnK, "log10K": lnK / math.log(10),
            "rows": rows}


def run_reaction(q: Query, opts, out_dir, log=None) -> GenericOutcome:
    t0 = time.time()
    log = log or (lambda stage, msg="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = opts.temperature
    P = opts.pressure
    _phases(q, T)
    eq_text = equation(q.reactants, q.products)
    log("thermochemistry", eq_text)
    th = thermo(q, T)
    th298 = thermo(q, 298.15) if abs(T - 298.15) > 0.5 else th
    checks = [_balance_check(q)]
    sources = {r["source"] for r in th["rows"]}
    est = [r["species"] for r in th["rows"] if "estimated" in r["source"]]
    out_of_range = [f"{r['species']}({r['phase']})" for r in th["rows"] if not r["covers"]]
    checks.append(Check("data", "warn" if est else "pass",
                        "every species comes from " + " and ".join(sorted(s.split(" (")[0] for s in sources))
                        if not est else f"{', '.join(est)} had no tabulated data, so their heats of formation are "
                                        "estimated from GFN2-xTB, good to about 25 kJ/mol",
                        {"sources": sorted(sources)}))
    if out_of_range:
        checks.append(Check("temperature range", "warn", f"{', '.join(out_of_range)} are tabulated only nearer "
                                                         f"room temperature than {fmt(T)} K, so their data are "
                                                         "carried there with a constant heat capacity", {}))
    values = {"equation": eq_text, "rule": q.reaction_source, "temperature_K": T, "pressure_Pa": P,
              "dH_kJ": th["dH"] / 1000, "dS_J_K": th["dS"], "dG_kJ": th["dG"] / 1000, "log10K": th["log10K"],
              "species": th["rows"], "dH298_kJ": th298["dH"] / 1000, "dG298_kJ": th298["dG"] / 1000}
    values["crossover_K"] = _crossover(q, th298)
    values["vapour_dH_kJ"] = _vapour(q, T)
    mass = sum(t.coef * t.species.molar_mass for t in q.reactants)
    values["reactant_mass_g"] = mass
    values["heat_kJ_per_kg"] = -th["dH"] / mass if mass else None
    # gas
    gas_in = sum(t.coef for t in q.reactants if t.phase == "g")
    gas_out = sum(t.coef for t in q.products if t.phase == "g")
    values["gas_mol"] = gas_out - gas_in
    values["gas_L"] = (gas_out - gas_in) * R * T / P * 1000 if gas_out > gas_in else 0.0
    values["gas_from_condensed"] = gas_in == 0 and gas_out > 0
    aqueous = any(t.phase == "aq" for t in q.reactants + q.products)
    # adiabatic temperature
    adiabatic = None
    if aqueous:
        adiabatic = _aqueous_rise(q, th, T)
    elif th["dH"] < 0:
        log("adiabatic", "products at the reactants' enthalpy")
        adiabatic = _adiabatic(q, T, P)
    if adiabatic:
        values["adiabatic"] = {k: v for k, v in adiabatic.items() if k != "state"}
        if adiabatic.get("check"):
            checks.append(adiabatic["check"])
    # equilibrium
    log("equilibrium", f"at {fmt(T)} K and {fmt(P / 1000)} kPa")
    eqm = _equilibrium(q, th, T, P, aqueous)
    values["equilibrium"] = {k: v for k, v in eqm.items() if k != "check"}
    if eqm.get("check"):
        checks.append(eqm["check"])
    # the method ladder for reactions of small gas molecules
    ladder = _ladder(q, th298, opts, log)
    if ladder:
        values["ladder"] = ladder
        checks.append(ladder.pop("check"))
    level, reasons = confidence([c for c in checks if c.status != "info"])
    head, metrics = _headline(q, values, th, T, aqueous)
    imgs = {}
    if opts.render:
        try:
            imgs["gibbs"] = _gibbs_chart(q, out, T, aqueous)
            values["_figures"] = [["gibbs", "Reaction Gibbs energy against temperature, below zero it goes forward"]]
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    from . import playback
    values["files"] = {"playback": playback.reaction(out, q, values, quick=opts.accuracy == "fast")}
    assumptions = [f"Standard states, 1 bar for gases and 1 mol/kg for species in water, with data from "
                   f"{' and '.join(sorted(sources))}.",
                   "Species in water are ideal, with activity equal to molality, which holds for dilute "
                   "solutions." if aqueous else "Gases are ideal.",
                   "Thermodynamics says how far and with how much heat a reaction goes, not how fast. A reaction "
                   "that can go may still need a spark, a catalyst or time."]
    if q.reaction_source and q.reaction_source != "written":
        assumptions.insert(0, f"The products come from {q.reaction_source}.")
    o = GenericOutcome("chemistry", eq_text, opts.question, "ok" if level != "low" else "marginal", head, metrics,
                       level, reasons, checks, assumptions, values, imgs,
                       solver="Wobbl reaction thermochemistry on NASA Glenn and NBS data, equilibrium with Cantera",
                       t_total=time.time() - t0)
    return o.write(out, [_table(th)])


def _crossover(q: Query, th298: dict) -> float | None:
    """Where the standard Gibbs energy changes sign, when every species' data reach that far."""
    if th298["dS"] == 0 or th298["dH"] * th298["dS"] <= 0:
        return None
    tc = th298["dH"] / th298["dS"]
    if tc <= 0:
        return None
    try:
        th = thermo(q, tc)
    except ResolveError:
        return None
    if not all(r["covers"] for r in th["rows"]):
        return None
    lo, hi = 200.0, 6000.0
    f = lambda T: thermo(q, T)["dG"]  # noqa: E731
    a, b = max(lo, tc * 0.5), min(hi, tc * 1.5)
    if f(a) * f(b) > 0:
        return tc
    for _ in range(40):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def _vapour(q: Query, T: float) -> float | None:
    """The reaction enthalpy with liquid water made as vapour instead, when the phase was not written."""
    from copy import copy
    hits = [t for t in q.products if t.species.formula == "H2O" and t.phase == "l" and not getattr(t, "written", False)]
    if not hits or T > 373.15:
        return None
    q2 = copy(q)
    q2.products = [Term(t.coef, t.species, "g" if t in hits else t.phase) for t in q.products]
    try:
        return thermo(q2, T)["dH"] / 1000
    except ResolveError:
        return None


def _balance_check(q: Query) -> Check:
    tot: dict = {}
    for sign, terms in ((-1, q.reactants), (1, q.products)):
        for t in terms:
            for e, n in t.species.counts.items():
                tot[e] = tot.get(e, 0) + sign * t.coef * n
            tot["charge"] = tot.get("charge", 0) + sign * t.coef * t.species.charge
    off = {k: v for k, v in tot.items() if abs(v) > 1e-9}
    if off:
        return Check("element and charge balance", "fail", f"the reaction does not balance in {', '.join(off)}", off)
    els = sorted(k for k in tot if k != "charge")
    return Check("element and charge balance", "pass", f"{', '.join(els)} and charge balance exactly", {})


def _adiabatic(q: Query, T: float, P: float) -> dict | None:
    """The products' temperature when the reaction's heat stays in them: their equilibrium at the reactants'
    enthalpy, so dissociation, melting and boiling all take their share."""
    from .install import available
    if not available():
        return None
    from .thermo import Equilibrium, Reactants
    els: dict = {}
    mass = 0.0
    h = 0.0
    for t in q.reactants:
        d = _data(t, T)
        h += t.coef * d.H(T)
        mass += t.coef * t.species.molar_mass
        for e, n in t.species.counts.items():
            els[e] = els.get(e, 0.0) + t.coef * n
    kg = mass / 1000.0
    st, eq, err_txt = None, None, ""
    for start in (2500.0, 3200.0, 1800.0):
        try:
            eq = Equilibrium(Reactants({e: n / kg for e, n in els.items()}, h / kg, "reactants"))
            st = eq.hp(P, T0=start)
            break
        except Exception as e:  # noqa: BLE001  Cantera's multiphase solver can fail from one start
            err_txt = "its multiphase solver stopped short" if "VCS" in str(e) or "FAILURE" in str(e) else \
                str(e).strip()[:100]
    if st is None:
        return _complete(q, T, h, err_txt)
    err = abs(st.h - eq.r.h0) / max(abs(eq.r.h0), st.cp * st.T)
    gas = {k: v for k, v in st.major(1e-3).items()}
    cond = {k: v for k, v in st.condensed.items() if v > 1e-3}
    chk = Check("adiabatic energy balance", "pass" if err < 1e-5 else "warn",
                f"the products at {fmt(st.T)} K hold the reactants' enthalpy to {err:.1g}, with their equilibrium "
                "found by Cantera on NASA's data", {"error": err})
    return {"T_K": st.T, "rise_K": st.T - T, "gas": gas, "condensed": cond, "check": chk, "kind": "equilibrium"}


def _complete(q: Query, T: float, h_react: float, why: str) -> dict | None:
    """The products' temperature when the reaction goes to completion and its heat stays in them, each product
    in its stable phase at each temperature, so melting and boiling take their latent heat. Used when the full
    equilibrium does not converge, and it leaves dissociation out."""
    from .thermodata import _nasa
    by = _nasa()
    if by is None:
        return None
    from .thermodata import Data, NASA, _counts_key
    phases = []
    for t in q.products:
        cands = []
        for ph in ("s", "l", "g"):
            for c in by.get((_counts_key(t.species.counts), ph), []):
                cands.append(Data(t.species.formula, ph, c.name, NASA, 0, 0, 0, c.thermo.min_temp, c.thermo.max_temp,
                                  c))
        if not cands:
            return None
        phases.append((t.coef, cands))

    def H(T, side=0.0):
        tot = 0.0
        for n, cands in phases:
            ok = [c for c in cands if c.covers(T + side)] or cands
            tot += n * min(ok, key=lambda c: c.G(T + side)).H(T + side)
        return tot

    lo, hi = T, 6000.0
    if H(hi) < h_react:
        return {"T_K": hi, "rise_K": hi - T, "kind": "complete", "note": "above 6000 K"}
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if H(m) < h_react:
            lo = m
        else:
            hi = m
    Tad = 0.5 * (lo + hi)
    below, above = H(Tad - 0.5), H(Tad + 0.5)
    plateau = above - below > 1000.0       # a phase change soaks up the heat right here
    note = (f"stopped at {fmt(Tad)} K, where a product changes phase and soaks up the rest of the heat" if plateau
            else "every product in its stable phase")
    chk = Check("adiabatic energy balance", "pass" if abs(H(Tad) - h_react) < 0.01 * abs(h_react) or plateau
                else "warn", f"with the reaction complete and nothing dissociating, the products reach {fmt(Tad)} K "
                             f"({note}). Cantera's full equilibrium did not converge here ({why})", {})
    return {"T_K": Tad, "rise_K": Tad - T, "kind": "complete", "check": chk, "plateau": plateau}


def _aqueous_rise(q: Query, th: dict, T: float) -> dict:
    """Temperature change of the solution per mole of reaction, with the solution's mass taken from what is
    dissolved (5% acetic acid for vinegar) or as 1 kg of water."""
    mass = 0.0
    how = []
    for t in q.reactants:
        sol = t.species.solution
        if sol[0] and sol[1] > 0:
            m = t.coef * t.species.molar_mass / sol[1]
            mass += m
            how.append(f"{fmt(m)} g of {t.species.name} at {100 * sol[1]:.0f}%")
    if not mass:
        mass = 1000.0
        how.append("1 kg of water")
    dT = -th["dH"] / (mass * CP_WATER)
    return {"rise_K": dT, "solution_g": mass, "basis": " plus ".join(how), "kind": "solution"}


def _equilibrium(q: Query, th: dict, T: float, P: float, aqueous: bool) -> dict:
    lk = th["log10K"]
    terms = q.reactants + q.products
    if all(t.phase == "g" for t in terms):
        r = _gas_equilibrium(q, T, P)
        if r:
            return r
    # extent from K with ideal activities, from the stoichiometric amounts in 1 kg of water or 1 bar of gas
    if lk > 8:
        return {"extent": 1.0, "log10K": lk, "how": "complete", "check": Check(
            "equilibrium", "pass", f"K is 10^{lk:.0f}, so the reaction runs to completion", {"log10K": lk})}
    if lk < -8:
        return {"extent": 0.0, "log10K": lk, "how": "none", "check": Check(
            "equilibrium", "pass", f"K is 10^{lk:.0f}, so almost nothing reacts", {"log10K": lk})}
    nu = [-t.coef for t in q.reactants] + [t.coef for t in q.products]

    def lnQ(x):
        s = 0.0
        for t, n in zip(terms, nu):
            if t.phase in ("s", "l"):
                continue
            amt = (t.coef * (1 - x)) if n < 0 else (t.coef * x)
            s += n * math.log(max(amt, 1e-300) * (1.0 if t.phase == "aq" else P / 1e5))
        return s
    lo, hi = 1e-15, 1 - 1e-15
    target = th["lnK"]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lnQ(mid) < target:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    resid = abs(lnQ(x) - target)
    return {"extent": x, "log10K": lk, "how": "activities", "check": Check(
        "equilibrium", "pass" if resid < 1e-6 else "warn",
        f"at equilibrium {100 * x:.1f}% of the reactants have reacted, where the reaction quotient matches K = "
        f"10^{lk:.2f} to {resid:.1g} in its logarithm", {"residual": resid})}


def _gas_equilibrium(q: Query, T: float, P: float) -> dict | None:
    """Cantera's equilibrium among just the reaction's gases, from the stoichiometric reactants."""
    from .install import available
    if not available():
        return None
    from .install import require
    from .thermodata import lookup as lk
    ct = require()
    species = []
    names = []
    for t in q.reactants + q.products:
        d = lk(t.species, "g", T)
        if d is None or d._sp is None:
            return None
        if d.name not in names:
            species.append(d._sp)
            names.append(d.name)
    gas = ct.Solution(thermo="ideal-gas", species=species)
    x0 = {}
    for t in q.reactants:
        d = lk(t.species, "g", T)
        x0[d.name] = x0.get(d.name, 0.0) + t.coef
    gas.TPX = T, P, x0
    gas.equilibrate("TP")
    X = dict(zip(gas.species_names, gas.X))
    # extent from the limiting reactant: moles are proportional to mole fractions at fixed elements
    lim = q.reactants[0]
    dlim = lk(lim.species, "g", T).name
    # moles of each species at equilibrium per mole of the start mixture, by element conservation
    els = {}
    for t in q.reactants:
        for e, n in t.species.counts.items():
            els[e] = els.get(e, 0.0) + t.coef * n
    e0 = next(iter(els))
    per = sum(X[nm] * sp.composition.get(e0, sp.composition.get(e0.upper(), 0)) for nm, sp in zip(gas.species_names,
                                                                                                gas.species()))
    scale = els[e0] / per if per else 0.0
    left = X[dlim] * scale
    extent = 1 - left / lim.coef
    nu = {nm: 0.0 for nm in names}
    for sign, terms in ((-1, q.reactants), (1, q.products)):
        for t in terms:
            nu[lk(t.species, "g", T).name] += sign * t.coef
    Kp = 1.0
    for nm, n in nu.items():
        Kp *= (X[nm] * P / 1e5) ** n if X[nm] > 0 else 1.0
    return {"extent": float(extent), "mole_fractions": {k: float(v) for k, v in X.items()}, "how": "cantera",
            "Kp": Kp, "check": Check("equilibrium", "pass", f"Cantera's equilibrium among the {len(names)} gases at "
                                                         f"{fmt(T)} K and {fmt(P / 1e5)} bar converts "
                                                         f"{100 * extent:.1f}% of the {display(lim.species)}",
                                     {"mole_fractions": X})}


def _ladder(q: Query, th: dict, opts, log) -> dict | None:
    """GFN2-xTB's reaction enthalpy for reactions of small gas molecules, against the tables."""
    if opts.accuracy == "fast":
        return None
    terms = q.reactants + q.products
    if not all(t.phase == "g" and t.species.smiles and t.species.atoms <= 20 for t in terms) or len(terms) > 6:
        return None
    from .formation import xtb_enthalpy
    log("method ladder", "GFN2-xTB reaction enthalpy")
    try:
        dh = 0.0
        for sign, ts in ((-1, q.reactants), (1, q.products)):
            for t in ts:
                dh += sign * t.coef * xtb_enthalpy(t.species)
    except Exception as e:  # noqa: BLE001
        return {"check": Check("method ladder", "skip", f"GFN2-xTB could not run on every species ({str(e)[:100]})")}
    diff = dh / 1000 - th["dH"] / 1000
    per = max(1.0, sum(t.coef for t in terms) / 2)
    ok = abs(diff) <= 30 * per
    return {"xtb_dH_kJ": dh / 1000, "diff_kJ": diff,
            "check": Check("method ladder", "pass" if ok else "info",
                           f"GFN2-xTB puts the reaction enthalpy at {dh / 1000:.0f} kJ against the tables' "
                           f"{th['dH'] / 1000:.1f} kJ, {abs(diff):.0f} kJ apart. Bond breaking reactions are where "
                           "semiempirical methods miss most, so the tables are the answer", {"diff_kJ": diff})}


def _headline(q: Query, v: dict, th: dict, T: float, aqueous: bool):
    dH, dG = v["dH_kJ"], v["dG_kJ"]
    heat = f"releases {-dH:.1f} kJ" if dH < 0 else f"takes in {dH:.1f} kJ"
    lk = v["log10K"]
    K = f"K = 10^{lk:.0f}" if abs(lk) >= 4 else f"K = {10 ** lk:.3g}"
    go = "goes forward on its own" if dG < 0 else "is uphill at standard conditions"
    head = f"{v['equation']} {heat} per mole as written and {go} at {fmt(T)} K (dG = {dG:.1f} kJ, {K})."
    if v.get("vapour_dH_kJ") is not None:
        head += f" With the water made as vapour it releases {-v['vapour_dH_kJ']:.1f} kJ."
    eqm = v.get("equilibrium") or {}
    if eqm.get("how") in ("cantera", "activities") and 0.001 < eqm.get("extent", 0) < 0.999:
        head += f" At equilibrium {100 * eqm['extent']:.3g}% of it reacts at {fmt(v['pressure_Pa'] / 1e5)} bar."
    if v.get("crossover_K"):
        tc = v["crossover_K"]
        head += f" It goes forward {'below' if v['dH_kJ'] < 0 else 'above'} about {fmt(tc)} K."
    ad = v.get("adiabatic") or {}
    if ad.get("kind") in ("equilibrium", "complete") and ad.get("T_K"):
        tail = ", where boiling or melting holds it" if ad.get("plateau") else ""
        head += f" Left to keep its own heat it reaches {fmt(ad['T_K'])} K{tail}."
    elif ad.get("kind") == "solution":
        word = "warms" if ad["rise_K"] > 0 else "cools"
        head += f" In {ad['basis']} it {word} the solution by {abs(ad['rise_K']):.1f} K."
    if v["gas_L"] > 0.01:
        verb = "gives off" if v.get("gas_from_condensed") else "makes"
        more = "" if v.get("gas_from_condensed") else " more"
        head += (f" It {verb} {v['gas_L']:.1f} L{more} of gas per mole at {fmt(T)} K and "
                 f"{fmt(v['pressure_Pa'] / 1000)} kPa.")
    metrics = [("reaction enthalpy", f"{dH:.1f} kJ"), ("Gibbs energy", f"{dG:.1f} kJ"), ("equilibrium constant", K)]
    if ad.get("T_K"):
        metrics.append(("adiabatic temperature", f"{fmt(ad['T_K'])} K"))
    elif ad.get("rise_K") is not None:
        metrics.append(("temperature change", f"{ad['rise_K']:+.1f} K"))
    return head, metrics


def _gibbs_chart(q: Query, out: Path, T: float, aqueous: bool) -> str:
    from .charts import sweep_chart
    lo, hi = (275.0, 370.0) if aqueous else (250.0, max(1500.0, T * 1.5))
    Ts = np.linspace(lo, hi, 60)
    gs = [thermo(q, t)["dG"] / 1000 for t in Ts]
    return sweep_chart(out / "images" / "gibbs.png", Ts, gs, "temperature, K", "reaction Gibbs energy, kJ",
                       (T, thermo(q, T)["dG"] / 1000), 0.0, "goes forward below this line")


def _table(th: dict) -> str:
    from ..report.html import table
    rows = [[r["species"], r["phase"], f"{r['coef']:+g}", f"{r['H_kJ_mol']:.2f}", f"{r['S_J_molK']:.2f}",
             r["source"].split(" (")[0]] for r in th["rows"]]
    return (f"<h2>Species at {fmt(th['T'])} K</h2>" +
            table(["species", "phase", "moles", "H, kJ/mol", "S, J/(mol K)", "data"], rows, {2, 3, 4}))
