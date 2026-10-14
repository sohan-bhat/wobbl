"""Propeller and drone rotor studies: how fast a prop has to spin to hover or make a thrust, the power it takes, and
whether the motor and prop leave enough margin.

"Can four 10x4.7 props lift a 1.2 kg quadcopter?" becomes: each rotor carries 2.94 N, blade element momentum theory
finds the speed and shaft power for that, the prop's rated top speed gives the most thrust it can make, and the
thrust to weight ratio decides the verdict. Props in the bundled UIUC set are also read from their measured curves,
which take the lead in the answer, with the theory as the check.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from . import bem
from .fluids import G, P_ATM, R_AIR, get_fluid
from .verify import compare_check, gci, gci_check

# APC's published top speeds: rpm = constant / diameter in inches (APC Propellers, RPM limits, rev 5, 2022)
APC_LIMIT = {"apcsf": (65000.0, "Slow Flyer"), "apce": (150000.0, "Thin Electric")}
TW_GOOD, TW_LOW = 2.0, 1.4     # thrust to weight a multirotor needs to climb and steer, and the least that flies


@dataclass
class PropStudy:
    prop: str = "apcsf_10x4.7"
    rpm: float = 0.0               # a given speed, or 0
    thrust: float = 0.0            # N wanted from each rotor, or 0
    mass: float = 0.0              # kg of the whole craft, to hover
    rotors: int = 1
    speed: float = 0.0             # m/s, forward or climb
    air_temp: float = 20.0         # C
    altitude: float = 0.0          # m
    max_rpm: float = 0.0           # 0 to use the maker's limit when it is known
    question: str = ""


def air_density(temp: float, altitude: float) -> tuple[float, float, float]:
    """Density, viscosity and speed of sound at a temperature and altitude (pressure from the standard atmosphere)."""
    p = P_ATM * (1 - 2.25577e-5 * altitude) ** 5.25588
    a = get_fluid("air", temp)
    return p / (R_AIR * (temp + 273.15)), a.mu, math.sqrt(1.4 * R_AIR * (temp + 273.15))


def _measured_rpm(prop: bem.Propeller, thrust: float, rho: float) -> tuple[float, float] | None:
    """Speed and power for a thrust from the measured static curves, or None when the thrust is outside them."""
    s = prop.data.get("static")
    if not s:
        return None
    rpm = np.array(s["rpm"])
    n = rpm / 60
    t = np.array(s["CT"]) * rho * n * n * prop.diameter ** 4
    if not t[0] <= thrust <= t[-1]:
        return None
    r = float(np.interp(thrust, t, rpm))
    cp = float(np.interp(r, rpm, s["CP"]))
    return r, cp * rho * (r / 60) ** 3 * prop.diameter ** 5


def run_prop(study: PropStudy, out_dir: str | Path) -> GenericOutcome:
    from .render import chart

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prop = bem.load_prop(study.prop)
    rho, mu, sound = air_density(study.air_temp, study.altitude)
    kw = dict(rho=rho, mu=mu, sound=sound)
    want = study.thrust or (study.mass * G / max(study.rotors, 1) if study.mass else 0.0)
    if want:
        r = bem.rpm_for_thrust(prop, want, study.speed, **kw)
    elif study.rpm:
        r = bem.solve(prop, study.rpm, study.speed, **kw)
    else:
        raise ValueError("give a speed (rpm), a thrust per rotor, or the craft's mass and rotor count")
    family = prop.key.split("_")[0] if prop.key else ""
    limit, limit_src = study.max_rpm, "the limit given"
    if not limit and family in APC_LIMIT:
        c, kind = APC_LIMIT[family]
        limit, limit_src = c / (prop.diameter / bem.INCH), f"APC's limit for {kind} props, {fmt(c)}/diameter in inches"
    checks: list[Check] = []
    extra = []

    # station convergence: the midpoint sum over the blade is second order
    levels = [bem.solve(prop, r.rpm, study.speed, stations=n, **kw) for n in (80, 40, 20)]
    g = gci([x.thrust for x in levels], [1 / 80, 1 / 40, 1 / 20])
    checks.append(gci_check(g, "thrust", "N", tol=0.01, what="station counts"))
    meas = None
    if study.speed == 0 and prop.data.get("static"):
        m = bem.measured_static(prop, r.rpm)
        if m:
            checks.append(compare_check("measured thrust", r.CT, m[0], "UIUC measured static CT", "", 0.10))
            checks.append(compare_check("measured power", r.CP, m[1], "UIUC measured static CP", "", 0.15))
            meas = _measured_rpm(prop, want, rho) if want else (r.rpm, m[1] * rho * (r.rpm / 60) ** 3 *
                                                                 prop.diameter ** 5)
    if study.speed == 0 and r.thrust > 0:
        fm = r.figure_of_merit
        st = "pass" if 0.4 <= fm <= 0.8 else "warn"
        checks.append(Check("momentum theory", st, f"ideal hover power from momentum theory is "
                                                     f"{fmt(fm * r.power)} W, so the figure of merit is {fm:.2f}"
                                                     + ("" if st == "pass" else ", outside the 0.4 to 0.8 real rotors "
                                                                                "reach"), {"figure_of_merit": fm}))
    tip = r.tip_mach
    checks.append(Check("tip speed", "pass" if tip < 0.6 else ("warn" if tip < 0.8 else "fail"),
                        f"tip Mach {tip:.2f}" + ("" if tip < 0.6 else ", where compressibility raises drag and "
                                                                       "noise beyond what the model holds")))
    re75 = r.re75
    checks.append(Check("Reynolds number", "pass" if re75 >= 50000 else "warn",
                        f"{fmt(re75)} at 75% radius" + ("" if re75 >= 50000 else ", low enough for laminar "
                                                                                   "separation, which can cost lift")))
    if r.stall_fraction > 0.5:
        checks.append(Check("stall", "info", f"{100 * r.stall_fraction:.0f}% of the blade is past stall, normal near "
                                             "the hub of a static prop, where the model is least accurate"))
    if meas is None:
        extra.append(("medium", "blade element momentum theory with a generic section, about 15% either way on "
                                "thrust and 20% on power for small props"))
    rpm_ans, p_ans = (meas if meas else (r.rpm, r.power))
    src = "the measured curves of this prop" if meas else "blade element momentum theory"
    t_max = None
    if limit:
        t_max = bem.solve(prop, limit, study.speed, **kw).thrust
        if meas and prop.data.get("static"):
            mm = bem.measured_static(prop, limit)
            if mm:
                t_max = mm[0] * rho * (limit / 60) ** 2 * prop.diameter ** 4
    weight = want if want else r.thrust
    tw = t_max / weight if (t_max and weight > 0) else None
    if limit and rpm_ans > limit:
        verdict = "fails"
        head = (f"{prop.name} would need {fmt(rpm_ans)} rpm to make {fmt(weight)} N, past {fmt(limit)} rpm "
                f"({limit_src}).")
    elif tw is not None and study.mass:
        verdict = "ok" if tw >= TW_GOOD else ("marginal" if tw >= TW_LOW else "fails")
        head = (f"Hover takes {fmt(rpm_ans)} rpm and {fmt(p_ans)} W of shaft power per rotor, read from {src}, and "
                f"at {fmt(limit)} rpm the rotors give {fmt(tw)} times the weight" +
                ("." if verdict == "ok" else f", short of the {TW_GOOD:g} a multirotor needs to climb and steer."))
    else:
        verdict = "ok"
        head = (f"{prop.name} at {fmt(rpm_ans)} rpm makes {fmt(want or r.thrust)} N for {fmt(p_ans)} W of shaft "
                f"power, read from {src}.")
    level, reasons = confidence(checks, extra)
    fixes = []
    if verdict != "ok" and want:
        for spec in _bigger(prop):
            try:
                alt = bem.rpm_for_thrust(bem.load_prop(spec), want, study.speed, **kw)
            except ValueError:
                continue
            fixes.append({"kind": "propeller", "title": f"Switch to the {bem.load_prop(spec).name}",
                          "summary": f"It hovers at {fmt(alt.rpm)} rpm for {fmt(alt.power)} W by the same method, "
                                     "since a larger disk moves more air more slowly, which costs less power "
                                     "for the same thrust.", "auto": False})
        if study.mass:
            need = weight * TW_GOOD
            fixes.append({"kind": "motor", "title": "Use a motor and battery that reach the thrust for 2x the weight",
                          "summary": f"Each rotor needs about {fmt(need)} N at full throttle, which this prop makes "
                                     f"at {fmt(bem.rpm_for_thrust(prop, need, **kw).rpm)} rpm if its rating "
                                     "allows.", "auto": False})
    imgs = {}
    try:
        top = max(limit or r.rpm * 1.5, r.rpm * 1.2)
        rpms = np.linspace(0.3 * top, top, 14)
        res = [bem.solve(prop, x, study.speed, **kw) for x in rpms]
        series = {"thrust, blade element theory": (rpms, [x.thrust for x in res])}
        pts = {}
        s = prop.data.get("static")
        if s and study.speed == 0:
            n = np.array(s["rpm"]) / 60
            pts["thrust, UIUC measured"] = (s["rpm"], np.array(s["CT"]) * rho * n * n * prop.diameter ** 4)
        imgs["thrust"] = chart(out / "images" / "thrust.png", series, "rpm", "thrust per rotor, N", pts,
                               f"{prop.name}, {fmt(rho)} kg/m3 air", hline=(weight, "needed") if want else None)
        imgs["blade"] = chart(out / "images" / "blade.png",
                              {"angle of attack, deg": (r.stations["r"] / prop.radius, r.stations["alpha_deg"]),
                               "lift coefficient x 10": (r.stations["r"] / prop.radius, 10 * r.stations["cl"])},
                              "radius / tip radius", "", None, f"blade loading at {fmt(r.rpm)} rpm")
    except Exception as e:  # noqa: BLE001
        imgs["error"] = str(e)
    metrics = [("rpm", fmt(rpm_ans)), ("shaft power", f"{fmt(p_ans)} W"), ("thrust", f"{fmt(want or r.thrust)} N")]
    if tw is not None:
        metrics.append(("thrust to weight", f"{tw:.2f}"))
    assumptions = [
        f"{prop.name}, {prop.blades} blades, {fmt(prop.diameter * 1000)} mm across"
        + (", chord and twist measured by UIUC" if prop.key else ", APC Slow Flyer chord shape with a constant pitch"),
        f"Air at {fmt(study.air_temp)} C and {fmt(study.altitude)} m above sea level, {fmt(rho)} kg/m3.",
        "Each rotor works in clean air, with no interference between rotors or with the frame, which usually "
        "costs a few percent of thrust." if study.rotors > 1 else "An isolated rotor in clean air.",
        "Sections use QPROP's generic thin cambered profile, and real sections differ, most near stall.",
        "Power is at the shaft, and motor and speed controller losses come on top, typically 20 to 35% more from "
        "the battery.",
    ]
    if prop.key:
        assumptions.append(f"Measured data come from the {bem.uiuc_source()}")
    values = {"rpm": rpm_ans, "power_W": p_ans, "thrust_N": want or r.thrust, "bem_rpm": r.rpm,
              "bem_power_W": r.power, "CT": r.CT, "CP": r.CP, "tip_mach": tip, "re75": re75,
              "thrust_to_weight": tw, "max_rpm": limit, "measured": bool(meas),
              "_figures": [["thrust", "Thrust against speed, with the measured points"],
                           ["blade", "Angle of attack and lift along the blade"]]}
    o = GenericOutcome("flow", prop.name, study.question, verdict, head, metrics, level, reasons, checks,
                       assumptions, values, imgs, fixes, solver="blade element momentum theory",
                       t_total=time.time() - t0)
    return o.write(out)


def _bigger(prop: bem.Propeller) -> list[str]:
    """Bundled props one size up from this one."""
    d = prop.diameter / bem.INCH
    keys = [k for k in bem.prop_names() if k.startswith(prop.key.split("_")[0] if prop.key else "apcsf")]
    return [k for k in keys if float(k.split("_")[1].split("x")[0]) > d][:2]
