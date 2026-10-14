"""Solid rocket motors: a design in, a thrust curve and a safety verdict out.

A motor is its grains, propellant, nozzle and casing (spec.py). Wobbl burns it back with the internal ballistics in
ballistics.py, then checks the simulation itself,

  - time step: the burn is run again at half the step until impulse, peak pressure and burn time settle,
  - mass balance: the propellant burned equals what left through the throat plus what the chamber still holds,
  - closed form: at peak pressure, where the pressure holds still, P = (a rho c* Kn)^(1/(1-n)),
  - geometry: the burning area swept over the web equals the propellant's volume,
  - chemistry: the propellant's c* against Cantera's equilibrium for its composition, and the delivered Isp
    against the ideal one at this nozzle,
  - burn rate data: whether the pressures stay inside the range the burn rate law was measured over,
  - erosive burning: the port to throat area ratio at ignition and the peak mass flux down the cores,

and judges the design on the casing (Lame's thick wall hoop stress at peak pressure, with the thin wall hand
calculation and Barlow's burst pressure) and the nozzle (separation from an overexpanded exit). Confidence comes
from the first group, the verdict from the casing, the ports and the nozzle. Each fix is simulated, so it comes
with its own peak pressure and impulse. The thrust curve is written as a CSV and a RASP .eng file that the motion
domain and OpenRocket read.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .ballistics import Burn, Motor, converge, simulate, steady_pressure
from .curves import write_curves
from .grains import Bates, MapGrain
from .spec import Casing, MotorSpec

MASS_FLUX_LIMIT = 1406.0         # kg/(m2 s), 2 lb/(in2 s), openMotor's default erosive burning alert
PORT_THROAT_MIN = 2.0            # openMotor's default alert for the port to throat area ratio
DESIGN_CHECKS = {"casing", "nozzle"}


def _mpa(p: float) -> str:
    return f"{fmt(p / 1e6)} MPa"


def run_motor(spec: MotorSpec, out_dir: str | Path, cb=None, render: bool = True) -> GenericOutcome:
    t0 = time.time()
    cb = cb or (lambda stage, detail="": None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = spec.motor
    cb("burn", f"{len(m.grains)} grain{'s' if len(m.grains) > 1 else ''} of {m.propellant.key}, "
               f"{fmt(m.nozzle.throat * 1000)} mm throat")
    b, runs = converge(m)
    i_pk = int(np.argmax(b.P))
    checks = [_step_check(runs), _mass_check(b), _closed_form_check(m, b, i_pk), _geometry_check(m)]
    cb("chemistry", "equilibrium cross check with Cantera")
    chem, chem_vals = _chemistry_check(m, b)
    checks.append(chem)
    checks.append(_range_check(m, b))
    ero, ero_vals = _erosive_check(m, b)
    checks.append(ero)
    case, case_vals = _casing_check(spec.casing, b.peak_pressure)
    checks.append(case)
    noz, noz_vals = _nozzle_check(m, b)
    checks.append(noz)
    level, reasons = confidence([c for c in checks if c.name not in DESIGN_CHECKS])
    status = {c.name: c.status for c in checks}
    if status["casing"] == "fail" or ero_vals["port_throat"] < 1.0:
        verdict = "fails"
    elif "warn" in (status["casing"], status["nozzle"], status["erosive burning"]):
        verdict = "marginal"
    else:
        verdict = "ok"
    cb("fixes", "simulating the changes that would help" if verdict != "ok" else "none needed")
    fixes = _fixes(spec, b, case_vals, ero_vals, noz_vals) if verdict != "ok" else []
    head = _headline(spec, b, case_vals, verdict)
    kns = [m.kn(x) for x in np.linspace(0, m.web * 0.999, 200)]
    values = {"designation": b.designation(), "class": b.designation()[:1], "impulse_Ns": b.impulse,
              "burn_time_s": b.burn_time, "average_thrust_N": b.average_thrust, "peak_thrust_N": b.peak_thrust,
              "peak_pressure_Pa": b.peak_pressure, "isp_s": b.isp, "propellant_kg": b.mass,
              "kn_initial": kns[0], "kn_max": max(kns), "kn_final": kns[-1], "cstar_m_s": m.cstar,
              "time_step_runs": runs, **ero_vals, **case_vals, **noz_vals, **chem_vals}
    files = write_curves(spec, b, out)
    values["files"] = files
    imgs = {}
    if render:
        from .charts import motor_chart
        try:
            allow = case_vals.get("allowable_Pa")
            imgs["burn"] = motor_chart(out / "images" / "burn.png", b.t, b.F, b.P, allow,
                                       f"casing allowable at safety factor {fmt(spec.casing.target_sf)}"
                                       if allow else "")
        except Exception as e:  # noqa: BLE001
            imgs["error"] = str(e)
    values["_figures"] = [["burn", "Thrust and chamber pressure over the burn"]]
    metrics = [("peak pressure", _mpa(b.peak_pressure)),
               ("casing safety factor", f"{case_vals['safety_factor']:.2f}" if "safety_factor" in case_vals
                else "not checked"),
               ("total impulse", f"{fmt(b.impulse)} N s ({b.designation()})"), ("burn time", f"{fmt(b.burn_time)} s")]
    o = GenericOutcome("motor", spec.source or spec.name, spec.question, verdict, head, metrics, level, reasons,
                       checks, assumptions(spec, chem_vals), values, imgs, fixes, solver=_solver(chem_vals),
                       t_total=time.time() - t0)
    return o.write(out)


def _solver(chem_vals: dict) -> str:
    s = "Wobbl internal ballistics (Runge-Kutta, scipy)"
    return s + (f" and Cantera {chem_vals['cantera']}" if chem_vals.get("cantera") else "")


# --- checks on the simulation -------------------------------------------------------------------------------
def _step_check(runs: list[dict]) -> Check:
    a, c = runs[-2], runs[-1]
    ch = c.get("change", 0.0)
    st = "pass" if ch < 0.003 else ("warn" if ch < 0.01 else "fail")
    return Check("time step convergence", st,
                 f"total impulse {fmt(a['impulse_Ns'])} N s with a {fmt(a['dt_s'] * 1000)} ms step and "
                 f"{fmt(c['impulse_Ns'])} N s with {fmt(c['dt_s'] * 1000)} ms, peak pressure "
                 f"{_mpa(a['peak_Pa'])} and {_mpa(c['peak_Pa'])}, largest change {100 * ch:.2g}%",
                 {"runs": runs})


def _mass_check(b: Burn) -> Check:
    err = abs(b.expelled + b.gas_end - b.gas_start - b.mass) / b.mass
    st = "pass" if err < 0.005 else ("warn" if err < 0.02 else "fail")
    return Check("mass balance", st, f"{fmt(b.mass)} kg of propellant burned and {fmt(b.expelled)} kg left "
                                     f"through the throat, with {fmt(b.gas_end * 1000)} g still in the chamber "
                                     f"({100 * err:.2g}% apart)", {"error": err})


def _closed_form_check(m: Motor, b: Burn, i: int) -> Check:
    x = min(float(b.x[i]), m.web * 0.999)
    pcf = steady_pressure(m, x)
    classic = steady_pressure(m, x, classic=True)
    err = abs(b.P[i] - pcf) / pcf
    st = "pass" if err < 0.005 else ("warn" if err < 0.02 else "fail")
    return Check("closed form", st,
                 f"at peak pressure, where it holds still, Kn {fmt(m.kn(x))} gives (a rho c* Kn)^(1/(1-n)) = "
                 f"{_mpa(pcf)} against the simulated {_mpa(b.P[i])} ({100 * err:.2g}% apart). The textbook form "
                 f"that leaves out the gas filling the room the propellant leaves gives {_mpa(classic)}",
                 {"kn": m.kn(x), "closed_form_Pa": pcf, "classic_Pa": classic, "error": err})


def _geometry_check(m: Motor) -> Check:
    worst, what = 0.0, ""
    for g in {id(g): g for g in m.grains}.values():
        n = 2000
        xs = np.linspace(0, g.web, n + 1)
        a = np.array([g.burn_area(x) for x in xs[:-1]] + [0.0])
        swept = float(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(xs)))
        err = abs(swept - g.volume(0.0)) / g.volume(0.0)
        if isinstance(g, MapGrain):
            err = max(err, g.coarea_error())
        if err >= worst:
            worst, what = err, g.describe()
    st = "pass" if worst < 0.01 else ("warn" if worst < 0.03 else "fail")
    return Check("geometry", st, f"the burning area swept over the web matches the propellant's volume to "
                                 f"{100 * worst:.2g}% (worst grain, {what})", {"error": worst})


def _chemistry_check(m: Motor, b: Burn) -> tuple[Check, dict]:
    p = m.propellant
    if not p.composition:
        return Check("chemistry", "skip", f"{p.name} has no published composition, so its c* of {fmt(p.cstar)} "
                                          "m/s could not be checked against equilibrium"), {}
    burning = b.P > 0.5 * b.peak_pressure
    pavg = float(np.sum(b.P[burning] * np.gradient(b.t)[burning]) / np.sum(np.gradient(b.t)[burning]))
    try:
        from . import install
        from .propellants import reactants
        from .thermo import Equilibrium, rocket
        ct = install.require()
        eq = Equilibrium(reactants(p.composition))
        ch = eq.hp(pavg)
        _, fr = rocket(eq, pavg, area_ratio=m.nozzle.area_ratio, ambient=m.ambient, mode="frozen", chamber=ch)
        _, sh = rocket(eq, pavg, area_ratio=m.nozzle.area_ratio, ambient=m.ambient, chamber=ch)
    except Exception as e:  # noqa: BLE001
        return Check("chemistry", "skip", f"the equilibrium cross check could not run ({str(e)[:160]})"), {}
    diff = abs(p.cstar - fr.cstar) / fr.cstar
    el, en = eq.element_error(ch), abs(ch.h - eq.r.h0) / max(abs(eq.r.h0), 1.0)
    st = "pass" if diff < 0.02 and el < 1e-6 and en < 1e-6 else ("warn" if diff < 0.05 else "fail")
    delivered = b.isp
    vals = {"cantera": ct.__version__, "equilibrium_T_K": ch.T, "equilibrium_M": ch.M, "equilibrium_gamma": ch.gamma,
            "equilibrium_cstar_m_s": fr.cstar, "ideal_isp_frozen_s": fr.isp_amb, "ideal_isp_shifting_s": sh.isp_amb,
            "average_pressure_Pa": pavg, "delivered_fraction": delivered / fr.isp_amb if fr.isp_amb > 0 else 0.0,
            "condensed_fraction": ch.condensed_fraction}
    s = (f"Cantera's equilibrium for {p.key} at the average {_mpa(pavg)} gives {fmt(ch.T)} K, {fmt(ch.M)} kg/kmol "
         f"and c* {fmt(fr.cstar)} m/s against the propellant data's {fmt(p.cstar)} m/s ({100 * diff:.2g}% apart), "
         f"with its elements and energy balanced to {el:.1g} and {en:.1g}. The ideal Isp at this nozzle is "
         f"{fmt(fr.isp_amb)} s frozen and {fmt(sh.isp_amb)} s shifting, and the motor delivers {fmt(delivered)} s, "
         f"{100 * vals['delivered_fraction']:.0f}% of frozen")
    return Check("chemistry", st, s, vals), vals


def _range_check(m: Motor, b: Burn) -> Check:
    lo, hi = m.propellant.burn_rate.range
    on = b.P > 0.3 * b.peak_pressure
    pmin = float(b.P[on].min())
    issues = []
    if b.peak_pressure > hi:
        issues.append(f"the peak of {_mpa(b.peak_pressure)} is above the {_mpa(hi)} top of the burn rate data, "
                      "so the rate there is extrapolated")
    if lo > 0 and pmin < lo:
        issues.append(f"the burn dips to {_mpa(pmin)}, under the {_mpa(lo)} bottom of the data")
    base = (f"the burn runs from {_mpa(pmin)} to {_mpa(b.peak_pressure)}, inside the {_mpa(lo)} to {_mpa(hi)} the "
            "burn rate was measured over")
    return Check("burn rate data", "warn" if issues else "pass", " and ".join(issues) if issues else base,
                 {"min_Pa": pmin, "data_range_Pa": [lo, hi]})


def _erosive_check(m: Motor, b: Burn) -> tuple[Check, dict]:
    At = m.nozzle.At
    ported = [g for g in m.grains if g.port(0.0) > 0]
    if not ported:
        return Check("erosive burning", "pass", "an end burner has no core for the flow to scour"), \
            {"port_throat": math.inf, "peak_mass_flux": 0.0}
    ratio = min(g.port(0.0) for g in ported) / At
    rho = m.propellant.density
    peak = 0.0
    for k in np.linspace(0, len(b.t) - 1, min(len(b.t), 300)).astype(int):
        x, r = float(b.x[k]), m.propellant.burn_rate.rate(float(b.P[k]))
        if x >= m.web:
            break
        made = 0.0
        for g in m.grains:              # forward to aft: each core carries what burned upstream of its aft end
            made += rho * g.burn_area(x) * r
            if g.port(x) > 0:
                peak = max(peak, made / g.port(x))
    vals = {"port_throat": ratio, "peak_mass_flux": peak}
    s = (f"the narrowest port is {ratio:.2f} times the throat area at ignition and the peak mass flux down the cores "
         f"is {fmt(peak)} kg/(m2 s), against the usual limits of {PORT_THROAT_MIN:g} and {fmt(MASS_FLUX_LIMIT)}")
    if ratio < 1.0:
        return Check("erosive burning", "fail", s + ". The port is smaller than the throat, so the flow chokes in the "
                                                   "core and the motor cannot work as drawn", vals), vals
    if ratio < PORT_THROAT_MIN or peak > MASS_FLUX_LIMIT:
        return Check("erosive burning", "warn", s + ". Fast core flow changes the burn rate, which this model leaves "
                                                   "out, so the real pressure early in the burn can differ from this "
                                                   "prediction", vals), vals
    return Check("erosive burning", "pass", s, vals), vals


# --- checks on the design ------------------------------------------------------------------------------------
def casing_stress(c: Casing, P: float) -> dict:
    from ..materials import resolve_material

    mat = resolve_material(c.material)
    ro, ri, t = c.outer / 2, c.inner / 2, c.wall
    lame = P * (ro ** 2 + ri ** 2) / (ro ** 2 - ri ** 2)         # hoop stress at the bore, Pa
    thin = P * ri / t
    sy, su = mat.yield_strength * 1e6, mat.tensile_strength * 1e6
    p_yield = sy * (ro ** 2 - ri ** 2) / (ro ** 2 + ri ** 2)
    p_burst = 2 * su * t / c.outer                               # Barlow on the outside diameter
    return {"material": mat.name, "hoop_Pa": lame, "hoop_thin_Pa": thin, "yield_Pa": sy, "tensile_Pa": su,
            "yield_pressure_Pa": p_yield, "burst_pressure_Pa": p_burst, "safety_factor": p_yield / P,
            "burst_margin": p_burst / P, "allowable_Pa": p_yield / c.target_sf, "target_sf": c.target_sf}


def _casing_check(c: Casing | None, P: float) -> tuple[Check, dict]:
    if c is None:
        return Check("casing", "info", "no casing was given, so the peak pressure was not checked against one"), {}
    v = casing_stress(c, P)
    sf = v["safety_factor"]
    s = (f"peak {_mpa(P)} puts {_mpa(v['hoop_Pa'])} of hoop stress on the bore of the {fmt(c.outer * 1000)} mm "
         f"{v['material']} casing with a {fmt(c.wall * 1000)} mm wall ({_mpa(v['hoop_thin_Pa'])} by the thin wall "
         f"hand calculation P r / t), a safety factor of {sf:.2f} on its {_mpa(v['yield_Pa'])} yield against a target "
         f"of {fmt(c.target_sf)}. It yields at {_mpa(v['yield_pressure_Pa'])} and bursts near "
         f"{_mpa(v['burst_pressure_Pa'])} by Barlow's formula on the tensile strength")
    st = "pass" if sf >= c.target_sf else ("warn" if sf >= 1.0 else "fail")
    return Check("casing", st, s, v), v


def _nozzle_check(m: Motor, b: Burn) -> tuple[Check, dict]:
    from .ballistics import SEPARATION
    from .thermo import _ideal_exit_ratio

    k, eps, Pa = m.propellant.k, m.nozzle.area_ratio, m.ambient
    pr = _ideal_exit_ratio(eps, k) if eps > 1 else (2 / (k + 1)) ** (k / (k - 1))
    burning = b.P > 0.5 * b.peak_pressure
    pavg = float(np.mean(b.P[burning]))
    pe = pr * pavg
    sep = burning & (pr * b.P < SEPARATION * Pa)
    frac = float(np.sum(np.gradient(b.t)[sep]) / max(np.sum(np.gradient(b.t)[burning]), 1e-12))
    opt = _optimum_ratio(pavg, Pa, k)
    vals = {"exit_pressure_Pa": pe, "area_ratio": eps, "optimum_area_ratio": opt, "separated_fraction": frac}
    s = (f"area ratio {fmt(eps)} expands the gas to {fmt(pe / 1000)} kPa at the average chamber pressure, against "
         f"{fmt(Pa / 1000)} kPa outside (an area ratio of {fmt(opt)} would match it)")
    if frac > 0.05:
        return Check("nozzle", "warn", s + f". The exit pressure stays under 40% of ambient for {100 * frac:.0f}% of "
                                           "the burn, so the flow separates inside the bell and the exit is larger "
                                           "than this motor can use", vals), vals
    return Check("nozzle", "pass", s, vals), vals


def _optimum_ratio(pc: float, pa: float, k: float) -> float:
    from .ballistics import _area_ratio
    pr = pa / pc
    crit = (2 / (k + 1)) ** (k / (k - 1))
    return _area_ratio(pr, k) if pr < crit else 1.0


# --- fixes, each simulated ------------------------------------------------------------------------------------
def _quick(m: Motor, dt: float) -> Burn:
    return simulate(m, dt)


def _fixes(spec: MotorSpec, b: Burn, case_vals: dict, ero_vals: dict, noz_vals: dict) -> list[dict]:
    m, c = spec.motor, spec.casing
    dt = b.dt
    fixes = []
    if c is not None and case_vals and case_vals["safety_factor"] < c.target_sf:
        allow = case_vals["allowable_Pa"]
        _, n = m.propellant.burn_rate.band(b.peak_pressure)
        dthroat = m.nozzle.throat
        for _ in range(4):           # P scales with throat area to the power -1/(1-n), so a few passes land on it
            trial = replace(m, nozzle=replace(m.nozzle, throat=dthroat,
                                              exit=max(m.nozzle.exit, dthroat)))
            tb = _quick(trial, dt)
            if abs(tb.peak_pressure - allow) / allow < 0.01:
                break
            dthroat *= (tb.peak_pressure / allow) ** ((1 - max(min(n, 0.9), -0.5)) / 2)
        sf = casing_stress(c, tb.peak_pressure)["safety_factor"]
        port = min((g.port(0.0) for g in m.grains if g.port(0.0) > 0), default=0.0) / trial.nozzle.At
        fixes.append({"kind": "throat", "title": f"Open the throat to {fmt(dthroat * 1000)} mm",
                      "summary": f"Simulated, the peak falls to {_mpa(tb.peak_pressure)} for a casing safety factor "
                                 f"of {sf:.2f}, with {fmt(tb.impulse)} N s over {fmt(tb.burn_time)} s "
                                 f"({tb.designation()}). The port to throat ratio becomes {port:.2f}.",
                      "auto": False, "after": {"throat_mm": dthroat * 1000, "peak_pressure_Pa": tb.peak_pressure,
                                               "impulse_Ns": tb.impulse, "safety_factor": sf}})
        S = case_vals["yield_Pa"] / c.target_sf
        P = b.peak_pressure
        if S > P:
            ro = c.inner / 2 * math.sqrt((S + P) / (S - P))
            fixes.append({"kind": "wall", "title": f"Thicken the casing wall to {fmt((ro - c.inner / 2) * 1000)} mm",
                          "summary": f"A {fmt(2 * ro * 1000)} mm tube over the same {fmt(c.inner * 1000)} mm bore "
                                     f"holds {_mpa(P)} with a safety factor of {fmt(c.target_sf)} by Lame's solution.",
                          "auto": False})
        else:
            fixes.append({"kind": "material", "title": "Use a stronger casing material",
                          "summary": f"No wall of {case_vals['material']} reaches a safety factor of "
                                     f"{fmt(c.target_sf)} at {_mpa(P)}, so a stronger tube is needed.", "auto": False})
        same = [g for g in m.grains if g is m.grains[-1]]
        if len(same) > 1:
            trial = replace(m, grains=m.grains[:-1])
            tb = _quick(trial, dt)
            sf = casing_stress(c, tb.peak_pressure)["safety_factor"]
            fixes.append({"kind": "segments", "title": f"Fly {len(m.grains) - 1} segments instead of {len(m.grains)}",
                          "summary": f"Simulated, the peak falls to {_mpa(tb.peak_pressure)} for a safety factor of "
                                     f"{sf:.2f}, with {fmt(tb.impulse)} N s ({tb.designation()}).", "auto": False})
    if ero_vals.get("port_throat", math.inf) < PORT_THROAT_MIN and any(isinstance(g, Bates) for g in m.grains):
        need = math.sqrt(4 * PORT_THROAT_MIN * m.nozzle.At / math.pi)
        grains = [replace(g, core=max(g.core, need)) if isinstance(g, Bates) else g for g in m.grains]
        if all(not isinstance(g, Bates) or g.core < g.D * 0.9 for g in grains):
            tb = _quick(replace(m, grains=grains), dt)
            fixes.append({"kind": "core", "title": f"Open the cores to {fmt(need * 1000)} mm",
                          "summary": f"That gives a port to throat ratio of {PORT_THROAT_MIN:g}. Simulated, the "
                                     f"peak is {_mpa(tb.peak_pressure)} with {fmt(tb.impulse)} N s "
                                     f"({tb.designation()}), since a wider core carries less propellant.",
                          "auto": False})
    if noz_vals.get("separated_fraction", 0.0) > 0.05:
        d = m.nozzle.throat * math.sqrt(noz_vals["optimum_area_ratio"])
        tb = _quick(replace(m, nozzle=replace(m.nozzle, exit=d)), dt)
        fixes.append({"kind": "exit", "title": f"Shrink the nozzle exit to {fmt(d * 1000)} mm",
                      "summary": f"That matches the exit to the outside air at the average chamber pressure. "
                                 f"Simulated, the motor gives {fmt(tb.impulse)} N s ({tb.designation()}).",
                      "auto": False})
    return fixes


def _headline(spec: MotorSpec, b: Burn, cv: dict, verdict: str) -> str:
    d = f"{b.designation()} motor, {fmt(b.impulse)} N s over {fmt(b.burn_time)} s at {fmt(b.isp)} s Isp"
    if not cv:
        return f"Peak chamber pressure {_mpa(b.peak_pressure)}. {d}. No casing was given to check it against."
    c = spec.casing
    if verdict == "fails" and cv["safety_factor"] < 1:
        return (f"The casing yields, since peak chamber pressure {_mpa(b.peak_pressure)} is over the "
                f"{_mpa(cv['yield_pressure_Pa'])} that {fmt(c.wall * 1000)} mm of {cv['material']} holds. {d}.")
    return (f"Peak chamber pressure {_mpa(b.peak_pressure)} against a casing that yields at "
            f"{_mpa(cv['yield_pressure_Pa'])}, a safety factor of {cv['safety_factor']:.2f} (target "
            f"{fmt(c.target_sf)}). {d}.")


def assumptions(spec: MotorSpec, chem_vals: dict) -> list[str]:
    m = spec.motor
    p, n = m.propellant, m.nozzle
    out = [f"{p.name}, {fmt(p.density)} kg/m3, burning at r = a P^n by its published law ({p.source}).",
           "The burning itself comes from that burn rate law, not resolved flame chemistry. Erosive burning, "
           "ignition delay, throat erosion and slag are not modeled.",
           f"c* {fmt(m.cstar)} m/s from T {fmt(p.T)} K, M {fmt(p.M)} kg/kmol and k {fmt(p.k)}"
           + (f", times a combustion efficiency of {fmt(m.cstar_efficiency)}." if m.cstar_efficiency < 1 else "."),
           f"Thrust from the ideal thrust coefficient for k {fmt(p.k)} and area ratio {fmt(n.area_ratio)}, times "
           f"{fmt(100 * n.divergence)}% for a {fmt(n.half_angle)} degree cone and a nozzle efficiency of "
           f"{fmt(n.efficiency)}.",
           "Every surface lights at once at ambient pressure, and the gas in the chamber is combustion gas from the "
           "start."]
    if spec.casing:
        out.append("The casing is checked as a long tube under internal pressure at room temperature strength. "
                   "Closures, retaining rings, bolts and the tube's heating are not checked, so line the casing and "
                   "size those parts on their own.")
    out.append("In Wobbl's validation the MIT O3800 static fire peaked 8% above its predicted pressure, so keep the "
               "casing's safety factor margin for what this model leaves out.")
    return out
