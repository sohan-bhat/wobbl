"""Combustion answers: a flame's temperature and products, and a propellant's rocket performance.

A flame is a fuel burned in air or oxygen at a given equivalence ratio, pressure and starting temperature. Wobbl
finds the adiabatic equilibrium on NASA's thermodynamic data and checks it four ways: the elements and the energy
balance, the same flame on GRI-Mech 3.0's independent thermodynamic data where the fuel is in that mechanism, the
temperature with no dissociation as an upper bound, and the fuel's flammability limits in air.

A propellant is a composition of known ingredients at a chamber pressure and nozzle. Wobbl reports CEA style
characteristic velocity and specific impulse for shifting and frozen expansion, checks the elements, the energy
in the chamber, the entropy along the nozzle and the ideal gas closed form for c*, and compares with the
propellant's published combustion properties when it has them.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .thermo import G0, P_REF, Equilibrium, Reactants, cstar_ideal, rocket

AIR_N2 = 3.76     # moles of nitrogen per mole of oxygen in air, Turns' model of air
# fuel name: (NASA species, GRI-Mech 3.0 species or None, flammability limits in air in volume %, from Zabetakis,
# US Bureau of Mines Bulletin 627, 1965)
FUELS = {
    "methane": ("CH4", "CH4", (5.0, 15.0)), "natural gas": ("CH4", "CH4", (5.0, 15.0)),
    "ethane": ("C2H6", "C2H6", (3.0, 12.4)), "propane": ("C3H8", "C3H8", (2.1, 9.5)),
    "butane": ("C4H10,n-butane", None, (1.8, 8.4)), "hydrogen": ("H2", "H2", (4.0, 75.0)),
    "carbon monoxide": ("CO", "CO", (12.5, 74.0)), "ethylene": ("C2H4", "C2H4", (2.7, 36.0)),
    "acetylene": ("C2H2,acetylene", "C2H2", (2.5, 100.0)), "methanol": ("CH3OH", "CH3OH", (6.7, 36.0)),
}
ALIASES = {"ch4": "methane", "c3h8": "propane", "lpg": "propane", "h2": "hydrogen", "co": "carbon monoxide",
           "c2h6": "ethane", "c2h4": "ethylene", "c2h2": "acetylene", "ch3oh": "methanol", "c4h10": "butane",
           "gas": "natural gas"}


def _ct():
    from .install import require
    return require()


def fuel_info(name: str) -> tuple[str, str, str | None, tuple | None]:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key in FUELS:
        nasa, gri, lim = FUELS[key]
        return key, nasa, gri, lim
    return name, name, None, None          # any NASA gas species by its name, with no limits on file


def _reactants(fuel: str, oxidizer: str, phi: float, T: float, P: float):
    """The fuel and oxidizer mixture as element moles and enthalpy per kg, from NASA's data at T."""

    from .install import require
    from .thermo import _database

    ct = require()
    gas_db, _ = _database()
    by_name = {s.name: s for s in gas_db}
    if fuel not in by_name:
        raise ValueError(f"unknown fuel '{fuel}' (use {', '.join(sorted(FUELS))} or a NASA gas species name)")
    sp = by_name[fuel]
    c, h, o = (sp.composition.get(e, 0.0) for e in ("C", "H", "O"))
    o2 = c + h / 4 - o / 2
    if o2 <= 0:
        raise ValueError(f"{fuel} needs no oxygen to burn, so an equivalence ratio does not apply")
    ox = oxidizer.strip().lower()
    if ox not in ("air", "oxygen", "o2"):
        raise ValueError(f"unknown oxidizer '{oxidizer}' (use air or oxygen)")
    moles = {fuel: 1.0, "O2": o2 / phi}
    if ox == "air":
        moles["N2"] = AIR_N2 * o2 / phi
    gas = ct.Solution(thermo="ideal-gas", species=[by_name[k] for k in moles])
    gas.TPX = T, P, moles
    el = {}
    for k, name in enumerate(gas.species_names):
        n = gas.Y[k] / gas.molecular_weights[k] * 1000.0    # mol per kg
        for e in gas.element_names:
            if gas.n_atoms(name, e):
                el[e] = el.get(e, 0.0) + n * gas.n_atoms(name, e)
    x_fuel = 1.0 / sum(moles.values())
    return Reactants(el, gas.enthalpy_mass, f"{fuel} in {ox} at phi {phi:g}"), moles, x_fuel, o2


def adiabatic(fuel: str, oxidizer: str, phi: float, T: float, P: float):
    r, moles, x_fuel, o2 = _reactants(fuel, oxidizer, phi, T, P)
    eq = Equilibrium(r)
    return eq, eq.hp(P, T0=2000.0), moles, x_fuel, o2


def _no_dissociation(fuel_sp: dict, moles: dict, h0: float, P: float) -> float:
    """Flame temperature if the fuel burned completely to CO2 and H2O with nothing breaking apart, an upper bound."""
    ct = _ct()
    from scipy.optimize import brentq

    from .thermo import _database

    gas_db, _ = _database()
    by_name = {s.name: s for s in gas_db}
    c, h = fuel_sp.get("C", 0.0), fuel_sp.get("H", 0.0)
    nf, no2 = moles[next(iter(moles))], moles["O2"]
    burned = min(1.0, no2 / (c + h / 4 - fuel_sp.get("O", 0.0) / 2)) * nf
    prod = {"CO2": c * burned, "H2O": h / 2 * burned, "O2": no2 - burned * (c + h / 4 - fuel_sp.get("O", 0.0) / 2),
            "N2": moles.get("N2", 0.0)}
    prod = {k: v for k, v in prod.items() if v > 1e-12}
    if burned < nf:     # rich: the unburned fuel stays as it is, which keeps this a bound
        prod[next(iter(moles))] = nf - burned
    gas = ct.Solution(thermo="ideal-gas", species=[by_name[k] for k in prod])

    def f(T):
        gas.TPX = T, P, prod
        return gas.enthalpy_mass - h0
    return brentq(f, 300.0, 6000.0)


def run_flame(fuel: str, oxidizer: str, phi: float, T: float, P: float, out_dir, question: str = "",
              limit: float | None = None, render: bool = True) -> GenericOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key, nasa, gri, lim = fuel_info(fuel)
    eq, st, moles, x_fuel, o2 = adiabatic(nasa, oxidizer, phi, T, P)
    ct = _ct()
    checks = []
    el = eq.element_error(st)
    en = abs(st.h - eq.r.h0) / (st.cp * st.T)
    checks.append(Check("element balance", "pass" if el < 1e-8 else "fail",
                        f"every element of the reactants is in the products to {el:.1g}"))
    checks.append(Check("energy balance", "pass" if en < 1e-6 else "fail",
                        f"the products at {fmt(st.T)} K hold the reactants' enthalpy to {en:.1g} of cp T"))
    t_gri = None
    if gri:
        g = ct.Solution("gri30.yaml")
        gm = {gri if k == nasa else k: v for k, v in moles.items()}
        g.TPX = T, P, gm
        g.equilibrate("HP")
        t_gri = g.T
        d = abs(t_gri - st.T) / st.T
        checks.append(Check("thermo data", "pass" if d < 0.005 else ("warn" if d < 0.02 else "fail"),
                            f"GRI-Mech 3.0's own thermodynamic data give {fmt(t_gri)} K for the same flame, "
                            f"{fmt(abs(t_gri - st.T))} K from NASA's ({100 * d:.2g}% apart)"))
    else:
        checks.append(Check("thermo data", "skip", f"{key} is not in GRI-Mech 3.0, so only NASA's data were used"))
    t_full = _no_dissociation(eq_species(eq, nasa), moles, eq.r.h0, P)
    checks.append(Check("no dissociation bound", "pass" if st.T <= t_full + 0.5 else "fail",
                        f"burning completely to CO2 and H2O with nothing breaking apart would reach {fmt(t_full)} K, "
                        f"and dissociation takes {fmt(t_full - st.T)} K off it"))
    burns = True
    if lim and oxidizer.lower() == "air":
        lean, rich = (_phi_at(v / 100, o2) for v in lim)
        burns = lean <= phi <= rich
        checks.append(Check("flammability", "pass" if burns else "fail",
                            f"{key} burns in air from {fmt(lim[0])}% to {fmt(lim[1])}% by volume, an equivalence ratio "
                            f"of {lean:.2f} to {rich:.2f}, and this mixture is {fmt(100 * x_fuel)}% fuel"
                            + ("" if burns else ", outside them, so it will not sustain a flame"),
                            {"phi_range": [lean, rich]}))
    level, reasons = confidence([c for c in checks if c.name != "flammability"])
    co = st.gas.get("CO", 0.0)
    no = st.gas.get("NO", 0.0)
    fixes = []
    if not burns:
        verdict = "fails"
        target = min(max(phi, lean * 1.05), rich * 0.95)
        head = (f"This mixture will not burn, since {fmt(100 * x_fuel)}% {key} in air is outside its flammability "
                f"limits. At an equivalence ratio of {target:.2f} it would.")
        fixes.append({"kind": "mixture", "title": f"Bring the equivalence ratio between {lean:.2f} and {rich:.2f}",
                      "summary": f"An equivalence ratio of {target:.2f} is inside the limits.", "auto": False})
    elif limit is not None and st.T > limit:
        verdict = "hot"
        lean_phi, t_lean = _lean_for(nasa, oxidizer, T, P, limit, phi, (lean if lim else 0.3))
        head = (f"The flame reaches {fmt(st.T)} K ({fmt(st.T - 273.15)} C), {fmt(st.T - limit)} K over the "
                f"{fmt(limit)} K limit.")
        if lean_phi:
            fixes.append({"kind": "mixture", "title": f"Run leaner, at an equivalence ratio of {lean_phi:.2f}",
                          "summary": f"That is {fmt(100 * (1 / lean_phi - 1))}% excess air, and the equilibrium "
                                     f"flame then reaches {fmt(t_lean)} K, at the limit.", "auto": False,
                          "after": {"phi": lean_phi, "T_ad_K": t_lean}})
    else:
        verdict = "ok"
        trace = [f"{_ppm(x)} {n}" for x, n in ((co, "CO"), (no, "NO")) if x > 1e-7]
        head = (f"Adiabatic flame temperature {fmt(st.T)} K ({fmt(st.T - 273.15)} C) for {key} in {oxidizer} at an "
                f"equivalence ratio of {phi:g}" + (f", with {' and '.join(trace)} at equilibrium." if trace else "."))
    values = {"fuel": key, "oxidizer": oxidizer, "phi": phi, "T_ad_K": st.T, "T_no_dissociation_K": t_full,
              "T_gri30_K": t_gri, "pressure_Pa": P, "reactant_T_K": T, "products": st.major(1e-5),
              "CO_ppm": co * 1e6, "NO_ppm": no * 1e6, "fuel_fraction": x_fuel}
    imgs = {}
    if render:
        try:
            imgs["sweep"] = _flame_sweep(out, nasa, oxidizer, T, P, phi, st.T, limit, lim and oxidizer == "air",
                                         o2, lim)
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    values["_figures"] = [["sweep", "Adiabatic flame temperature against equivalence ratio"]]
    if (out / "playback.json").exists():
        values["files"] = {"playback": "playback.json"}
    assumptions = [f"{key.capitalize()} ({nasa}) and {'air as O2 + 3.76 N2' if oxidizer == 'air' else 'oxygen'} at "
                   f"{fmt(T)} K and {fmt(P / 1000)} kPa, fully mixed.",
                   "Adiabatic and at constant pressure, with the products in chemical equilibrium on NASA's "
                   "thermodynamic data (McBride, Gordon and Reno, NASA TM-4513). A real flame loses heat by radiation "
                   "and to walls, so its peak runs below this temperature.",
                   "Equilibrium NO is an upper bound on what a flame makes, since forming NO takes longer than most "
                   "flames last."]
    metrics = [("flame temperature", f"{fmt(st.T)} K"), ("equivalence ratio", f"{phi:g}"), ("CO", _ppm(co)),
               ("NO", _ppm(no))]
    o = GenericOutcome("combustion", f"{key} flame", question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, imgs, fixes, solver=f"Cantera {ct.__version__}", t_total=time.time() - t0)
    return o.write(out, [_products_table(st)])


def eq_species(eq: Equilibrium, name: str) -> dict:
    return eq.gas.species(name).composition


def _phi_at(x_fuel: float, o2: float) -> float:
    """Equivalence ratio of a fuel and air mixture that is x_fuel fuel by volume."""
    return x_fuel / (1 - x_fuel) * (1 + AIR_N2) * o2


def _ppm(x: float) -> str:
    return f"{fmt(100 * x)}%" if x >= 0.001 else f"{fmt(x * 1e6)} ppm"


def _lean_for(fuel, oxidizer, T, P, limit, phi, lo):
    """The leaner equivalence ratio whose flame reaches the limit, by bisection, or (None, None)."""
    a, b = lo, min(phi, 1.0)
    ta = adiabatic(fuel, oxidizer, a, T, P)[1].T
    if ta > limit:
        return None, None
    tb = adiabatic(fuel, oxidizer, b, T, P)[1].T
    for _ in range(30):
        m = 0.5 * (a + b)
        tm = adiabatic(fuel, oxidizer, m, T, P)[1].T
        if tm > limit:
            b, tb = m, tm
        else:
            a, ta = m, tm
        if abs(tm - limit) < 1.0:
            break
    return a, ta


def _flame_sweep(out, fuel, oxidizer, T, P, phi, t_here, limit, in_air, o2, lim):
    from .charts import sweep_chart

    lo, hi = (0.5, 1.6)
    if in_air and lim:
        lo, hi = max(0.3, _phi_at(lim[0] / 100, o2)), min(2.5, _phi_at(min(lim[1], 60) / 100, o2))
    lo, hi = min(lo, phi), max(hi, phi)
    # finer near stoichiometric, where the temperature peaks, and through the operating point
    phis = np.unique(np.concatenate([np.linspace(lo, hi, 20), np.linspace(0.8, 1.3, 6), [phi]]))
    phis = phis[(phis >= lo) & (phis <= hi)]
    states = [adiabatic(fuel, oxidizer, p, T, P)[1] for p in phis]
    temps = [s.T for s in states]
    from . import playback
    playback.flame(out, phis, temps, [s.major(1e-5) for s in states], phi,
                   {"fuel": fuel, "oxidizer": oxidizer, "phi": phi, "T_K": t_here, "pressure_Pa": P,
                    "reactant_T_K": T, "limit_K": limit})
    return sweep_chart(out / "images" / "sweep.png", phis, temps, "equivalence ratio", "flame temperature, K",
                       (phi, t_here), limit, f"limit {fmt(limit)} K" if limit else "")


def _products_table(st) -> str:
    from ..report.html import table

    rows = [[k, f"{100 * v:.4g}%"] for k, v in st.major(1e-4).items()]
    rows += [[k + " (condensed)", f"{100 * v:.4g}% by mass"] for k, v in st.condensed.items()]
    return "<h2>Equilibrium products</h2>" + table(["species", "mole fraction"], rows, {1})


# --- rocket propellant performance --------------------------------------------------------------------------------
def run_propellant(name: str, pc: float, area_ratio: float | None, exit_P: float | None, out_dir, question: str = "",
                   ambient: float = P_REF, render: bool = True) -> GenericOutcome:
    ct = _ct()

    from .propellants import get_propellant, parse_composition, reactants

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preset = None
    if ":" in name:
        comp = parse_composition(name)
    else:
        preset = get_propellant(name)
        if not preset.composition:
            raise ValueError(f"{preset.name} has no published composition, so its performance cannot be computed "
                             "from chemistry. Give a composition such as ap:70,al:16,htpb:14")
        comp = preset.composition
    if area_ratio is None and exit_P is None:
        if ambient <= 0:
            raise ValueError("a nozzle for vacuum needs an area ratio, since it never matches zero pressure")
        exit_P = ambient
    eq = Equilibrium(reactants(comp))
    ch = eq.hp(pc)
    _, sh = rocket(eq, pc, area_ratio, exit_P, ambient, "shifting", ch)
    _, fr = rocket(eq, pc, sh.stations[-1].area_ratio if exit_P is not None else area_ratio, None, ambient,
                   "frozen", ch)
    eps = sh.stations[-1].area_ratio
    checks = []
    el = eq.element_error(ch)
    en = abs(ch.h - eq.r.h0) / (ch.cp * ch.T)
    checks.append(Check("element balance", "pass" if el < 1e-8 else "fail",
                        f"every element of the propellant is in the chamber products to {el:.1g}"))
    checks.append(Check("energy balance", "pass" if en < 1e-6 else "fail",
                        f"the chamber products at {fmt(ch.T)} K hold the propellant's enthalpy to {en:.1g} of cp T"))
    drift = max(sh.entropy_error, fr.entropy_error)
    checks.append(Check("entropy", "pass" if drift < 1e-6 else "warn",
                        f"the expansion holds the chamber's entropy to {drift:.1g} at the throat and exit"))
    ideal = cstar_ideal(ch.T, ch.M, ch.gamma)
    d = abs(ideal - fr.cstar) / fr.cstar
    checks.append(Check("closed form", "pass" if d < 0.01 else "warn",
                        f"the ideal gas c* = sqrt(R T / M) / Gamma(k) with the chamber's {fmt(ch.T)} K, "
                        f"{fmt(ch.M)} kg/kmol and k {ch.gamma:.4f} gives {fmt(ideal)} m/s against the frozen nozzle's "
                        f"{fmt(fr.cstar)} m/s ({100 * d:.2g}% apart)"))
    if preset is not None:
        dd = abs(preset.cstar - fr.cstar) / fr.cstar
        checks.append(Check("published data", "pass" if dd < 0.02 else ("warn" if dd < 0.05 else "fail"),
                            f"{preset.name} is published at {fmt(preset.T)} K, {fmt(preset.M)} kg/kmol and k "
                            f"{fmt(preset.k)}, so c* {fmt(preset.cstar)} m/s, {100 * dd:.2g}% from this equilibrium"))
    level, reasons = confidence(checks)
    pe = sh.exit_P
    fixes = []
    if pe < 0.4 * ambient:
        verdict = "marginal"
        _, opt = rocket(eq, pc, None, ambient, ambient, "shifting", ch)
        fixes.append({"kind": "area_ratio", "title": f"Use an area ratio of {fmt(opt.stations[-1].area_ratio)}",
                      "summary": f"That expands to the {fmt(ambient / 1000)} kPa outside, where this nozzle's exit at "
                                 f"{fmt(pe / 1000)} kPa would let the flow separate. Isp there is "
                                 f"{fmt(opt.isp_amb)} s shifting.", "auto": False})
    else:
        verdict = "ok"
    label = preset.name if preset else ", ".join(f"{k} {100 * w:.4g}%" for k, w in comp.items())
    head = (f"{label} at {fmt(pc / 1e6)} MPa and area ratio {fmt(eps)} gives c* {fmt(sh.cstar)} m/s and Isp "
            f"{fmt(sh.isp_amb)} s at {fmt(ambient / 1000)} kPa and {fmt(sh.isp_vac)} s in vacuum shifting, "
            f"{fmt(fr.isp_amb)} s and {fmt(fr.isp_vac)} s frozen, from a {fmt(ch.T)} K chamber.")
    if verdict == "marginal":
        head += f" The exit at {fmt(pe / 1000)} kPa is far below ambient, so the flow separates in the nozzle."
    values = {"composition": comp, "chamber_pressure_Pa": pc, "area_ratio": eps, "ambient_Pa": ambient,
              "T_chamber_K": ch.T, "M": ch.M, "gamma": ch.gamma, "condensed_fraction": ch.condensed_fraction,
              "cstar_shifting_m_s": sh.cstar, "cstar_frozen_m_s": fr.cstar, "isp_shifting_s": sh.isp_amb,
              "isp_vac_shifting_s": sh.isp_vac, "isp_frozen_s": fr.isp_amb, "isp_vac_frozen_s": fr.isp_vac,
              "exit_pressure_Pa": pe, "products": ch.major(1e-4), "condensed": ch.condensed,
              "stations": [s.__dict__ for s in sh.stations]}
    imgs = {}
    if render:
        try:
            imgs["isp"] = _isp_curve(out, eq, ch, sh, fr, pc, ambient, eps)
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    values["_figures"] = [["isp", "Specific impulse against nozzle area ratio"]]
    if (out / "playback.json").exists():
        values["files"] = {"playback": "playback.json"}
    assumptions = [f"{label}, its ingredients' enthalpies from NASA's CEA data and the NIST WebBook.",
                   "CEA's infinite area combustor, with the chamber in equilibrium and the nozzle isentropic. Shifting "
                   "keeps the gas in equilibrium as it expands and frozen keeps the chamber composition, the two "
                   "bounds of a real nozzle.",
                   "Condensed products move with the gas at its speed and temperature, and their volume is neglected.",
                   "These are ideal numbers. Real motors deliver about 85 to 95% of them after two phase flow, "
                   "divergence, friction and heat loss."]
    metrics = [("c*", f"{fmt(sh.cstar)} m/s"), ("Isp", f"{fmt(sh.isp_amb)} s"), ("Isp vacuum", f"{fmt(sh.isp_vac)} s"),
               ("chamber", f"{fmt(ch.T)} K")]
    o = GenericOutcome("combustion", label, question, verdict, head, metrics, level, reasons, checks, assumptions,
                       values, imgs, fixes, solver=f"Cantera {ct.__version__}", t_total=time.time() - t0)
    return o.write(out, [_stations_table(sh, fr), _products_table(ch)])


def _stations_table(sh, fr) -> str:
    from ..report.html import table

    rows = []
    for perf in (sh, fr):
        for s in perf.stations:
            rows.append([f"{perf.mode} {s.name}", f"{fmt(s.P / 1e6)} MPa", f"{fmt(s.T)} K", f"{fmt(s.v)} m/s",
                         "" if s.name == "chamber" else fmt(s.area_ratio), f"{fmt(100 * s.condensed)}%"])
    return "<h2>Nozzle stations</h2>" + table(["station", "pressure", "temperature", "velocity", "area ratio",
                                               "condensed"], rows, {1, 2, 3, 4, 5})


def _isp_curve(out, eq, ch, sh, fr, pc, ambient, eps):
    from .charts import sweep_chart

    from . import playback

    G_t = pc / sh.cstar
    G_tf = pc / fr.cstar
    pts, rows = [], []
    pe_lo = min(sh.exit_P, ambient if ambient > 0 else sh.exit_P) / 4
    for P in np.geomspace(sh.stations[1].P * 0.9, pe_lo, 16):
        guess = ch.T * (P / pc) ** ((ch.gamma - 1) / ch.gamma)
        s1 = eq.sp(P, ch.s, guess)
        v1 = math.sqrt(max(2 * (ch.h - s1.h), 0.0))
        s2 = eq.frozen_sp(ch, P, ch.s, guess)
        v2 = math.sqrt(max(2 * (ch.h - s2.h), 0.0))
        g1, g2 = s1.rho * v1, s2.rho * v2
        pts.append((G_t / g1, (v1 + (P - ambient) / g1) / G0, (v2 + (P - ambient) / g2) / G0, G_tf / g2))
        r = playback._r
        rows.append({"P_MPa": r(P / 1e6), "area_ratio": r(G_t / g1), "T_K": r(s1.T), "v_m_s": r(v1),
                     "condensed": r(s1.condensed_fraction, 3), "isp_s": r(pts[-1][1]),
                     "frozen": {"area_ratio": r(G_tf / g2), "T_K": r(s2.T), "v_m_s": r(v2), "isp_s": r(pts[-1][2])}})
    playback.nozzle(out, rows, {"chamber_pressure_Pa": pc, "chamber_T_K": ch.T, "ambient_Pa": ambient,
                                "area_ratio": eps, "cstar_m_s": sh.cstar, "isp_s": sh.isp_amb})
    pts.sort()
    xs = [p[0] for p in pts]
    froz = list(np.interp(xs, sorted(p[3] for p in pts), [p[2] for p in sorted(pts, key=lambda p: p[3])]))
    return sweep_chart(out / "images" / "isp.png", xs, [p[1] for p in pts], "nozzle area ratio",
                       f"Isp at {fmt(ambient / 1000)} kPa, s", (eps, sh.isp_amb), label="shifting",
                       second=(froz, "frozen"))
