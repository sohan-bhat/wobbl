"""The checks behind a motion answer.

Verification checks decide the confidence: the flight flown again with a tolerance ten times tighter, closed
forms wherever the model reduces to one (a fall without air, a fall through air of one density, a descent under
a parachute, Tsiolkovsky's delta v less what gravity, drag and steering took), an energy balance and units.
Rocketry checks decide the verdict: Barrowman's stability margin, speed off the rail, the peak loads, apogee
against its target, landing speed and drift. Fixes give the nose weight, fin span, rail length or main parachute
that would pass.
"""
from __future__ import annotations

import math
from dataclasses import replace

from ..units import fmt
from ..verify.checks import Check
from .atmosphere import air, gravity
from .flight import E_PROP, L_DRAG, L_GRAV, L_STEER, M, VX, VZ, W_DRAG, W_GRAV, W_THRUST, X, Z, Flight, fly
from .spec import RAIL_EXIT_FLOOR, STABILITY_FIX, MotionStudy, barrowman, burnout_cg, margin

SOFT_LANDING = 5.0         # m/s a main parachute fix aims for
APOGEE_BAND = 0.05         # how close to its target an apogee passes


def _n(v: float) -> str:
    """A length or speed for text, with the integrator's dust at zero shown as 0."""
    return fmt(0.0 if abs(v) < 5e-4 else v)


def _energy(j: float) -> str:
    a = abs(j)
    return f"{fmt(j / 1e6)} MJ" if a >= 1e6 else f"{fmt(j / 1e3)} kJ" if a >= 1e3 else f"{fmt(j)} J"


def _speed(y) -> float:
    return math.hypot(y[VX], y[VZ])


# --- the flight's numbers ---------------------------------------------------------------
def points(fl: Flight, per_segment: int = 60) -> list[dict]:
    """Every sample of the flight with what the table and the peaks need."""
    out = []
    for t, y, ph in fl.samples(per_segment):
        f = fl.model.forces(t, y, ph)
        out.append({"t": t, "y": y, "phase": ph.name, "held": ph.held, "accel": math.hypot(f["ax"], f["az"]),
                    "felt": 0.0 if ph.held else math.hypot(f["ax"], f["az"] + f["g"]), "mach": f["mach"],
                    "q": f["q"], "thrust": f["T"]})
    return out


def key_numbers(fl: Flight, pts: list[dict]) -> dict[str, tuple[float, str]]:
    """The answers a run is judged by, which the step convergence check compares between tolerances."""
    st, end = fl.study, fl.end
    out = {"flight time": (end.t, "s")}
    apo = fl.event("apogee")
    if apo is not None:
        out["apogee"] = (apo.y[Z], "m")
    if st.rocket:
        out["burnout speed"] = (_speed(fl.event("burnout").y), "m/s")
        out["max Mach"] = (max(p["mach"] for p in pts), "")
    out["landing speed" if st.chutes else "impact speed"] = (-end.y[VZ] if st.chutes else _speed(end.y), "m/s")
    if abs(end.y[X]) > 1.0:
        out["drift" if st.rocket else "distance"] = (end.y[X], "m")
    return out


def peak(pts: list[dict], key: str, phases=None) -> dict:
    cand = [p for p in pts if phases is None or p["phase"] in phases]
    return max(cand, key=lambda p: p[key]) if cand else pts[0]


# --- verification ------------------------------------------------------------------------
def step_convergence(a: dict, b: dict) -> Check:
    """The run flown again with a tolerance ten times tighter has to give the same answers."""
    changes = {k: abs(a[k][0] - b[k][0]) / max(abs(b[k][0]), 1e-9) for k in a if k in b}
    worst = max(changes, key=changes.get)
    ch = changes[worst]
    items = [f"{k} {_n(a[k][0])}{' ' + a[k][1] if a[k][1] else ''}" for k in a]
    listed = ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]
    s = (f"flown again with a tolerance ten times tighter, the {listed} moved by at most {100 * ch:.2g}% (the "
         f"{worst})")
    st = "pass" if ch <= 1e-3 else ("warn" if ch <= 1e-2 else "fail")
    if st != "pass":
        s += ", so the steps are too coarse for this flight to trust every digit"
    return Check("step convergence", st, s, {"normal": {k: v[0] for k, v in a.items()},
                                             "tight": {k: v[0] for k, v in b.items()}, "largest_change": ch})


def _mid_air(st: MotionStudy, z: float) -> tuple[float, float]:
    """Density and gravity at z above the ground, as the run sees them."""
    rho = air(st.site_altitude + z)[0] if st.air == "standard" else (0.0 if st.air == "vacuum" else float(st.air))
    g = gravity(st.site_altitude) if st.uniform_gravity else gravity(st.site_altitude + z)
    return rho, g


def vacuum_check(st: MotionStudy, fl: Flight) -> tuple[Check, dict]:
    """The same body flown with the air taken away and gravity held at its ground value, which is exactly the
    closed form for a fall or a throw without air. It checks the integrator and the ground event on this very
    body, and shows how much the air changes."""
    vac = fly(replace(st, air="vacuum", wind=0.0, uniform_gravity=True))
    end = vac.end
    a = math.radians(st.angle)
    vs, vh = st.speed * math.sin(a), st.speed * math.cos(a)
    g = gravity(st.site_altitude)
    t_c = (vs + math.sqrt(vs * vs + 2 * g * st.height)) / g
    if t_c <= 0:
        raise ValueError("the body starts on the ground at rest, so the run has no fall to check. Give the height "
                         "it falls from or a launch speed.")
    v_c = math.hypot(vh, vs - g * t_c)
    err = max(abs(end.t - t_c) / t_c, abs(_speed(end.y) - v_c) / v_c)
    drop = st.speed == 0
    form = ("sqrt(2h/g) gives" if drop else "the closed form for a throw without air gives")
    s = (f"with the air taken away and gravity held at its ground value the run lands after {end.t:.4g} s at "
         f"{fmt(_speed(end.y))} m/s, and {form} {t_c:.4g} s and {fmt(v_c)} m/s ({100 * err:.2g}% apart)")
    if fl.study.air != "vacuum":
        dt, dv = fl.end.t - end.t, _speed(end.y) - _speed(fl.end.y)
        s += (f". The air adds {fmt(dt)} s to the {'fall' if drop else 'flight'} and takes {fmt(dv)} m/s off "
              "the impact")
    st_ = "pass" if err < 1e-5 else ("warn" if err < 1e-3 else "fail")
    rows = vac.samples(200)
    curve = {"t": [t for t, _, _ in rows], "alt": [y[Z] for _, y, _ in rows], "speed": [_speed(y) for _, y, _ in rows]}
    return Check("vacuum drop" if drop else "vacuum flight", st_, s,
                 {"time_s": end.t, "closed_form_time_s": t_c, "speed_mps": _speed(end.y),
                  "closed_form_speed_mps": v_c, "range_m": end.y[X]}), curve


def drag_fall_check(st: MotionStudy, fl: Flight) -> Check | None:
    """A drop from rest in still air against the quadratic drag solution, v = vt tanh(g t / vt) and
    h = vt^2 / g ln cosh(g t / vt), with the air's density and gravity taken halfway down."""
    if st.speed != 0 or st.wind != 0 or st.air == "vacuum" or st.height <= 0 or st.chutes:
        return None
    h = st.height
    rho, g = _mid_air(st, h / 2)
    cd = st.drag_coefficient()(0.0)
    vt = math.sqrt(2 * st.mass * g / (rho * cd * st.area))
    t_c = vt / g * math.acosh(math.exp(g * h / (vt * vt)))
    v_c = vt * math.tanh(g * t_c / vt)
    t_s, v_s = fl.end.t, _speed(fl.end.y)
    err = max(abs(t_s - t_c) / t_c, abs(v_s - v_c) / v_c)
    thin = _mid_air(st, h)[0] / _mid_air(st, 0.0)[0]
    s = (f"with the air at its density halfway down ({rho:.4g} kg/m3) the terminal speed is {fmt(vt)} m/s, and "
         f"v = vt tanh(g t / vt) with h = vt^2/g ln cosh(g t / vt) gives {t_c:.4g} s and {fmt(v_c)} m/s against "
         f"{t_s:.4g} s and {fmt(v_s)} m/s from the run ({100 * err:.2g}% apart)")
    if thin < 0.8:
        return Check("hand calculation", "info", s + f". The air at the top is {100 * thin:.2g}% as dense as at "
                                                     "the ground, too far from one density for the formula to hold",
                     {"terminal_mps": vt})
    if err > 1e-4:
        s += ", the difference coming from the air thickening on the way down"
    st_ = "pass" if err < 0.01 else ("warn" if err < 0.03 else "fail")
    return Check("hand calculation", st_, s, {"terminal_mps": vt, "closed_form_time_s": t_c,
                                              "closed_form_speed_mps": v_c, "error": err})


def delta_v_check(fl: Flight) -> tuple[Check, dict]:
    """Speed at burnout against Tsiolkovsky's ideal delta v less what gravity, drag and steering took."""
    st, mo = fl.study, fl.study.motor
    y = fl.event("burnout").y
    ideal = mo.exhaust_speed * math.log(st.liftoff_mass / st.burnout_mass)
    lg, ld, ls = y[L_GRAV], y[L_DRAG], y[L_STEER]
    pred, flown = ideal - lg - ld - ls, _speed(y)
    err = abs(pred - flown) / ideal
    s = (f"Tsiolkovsky gives {fmt(ideal)} m/s from a {fmt(mo.exhaust_speed)} m/s exhaust speed and a mass ratio "
         f"of {st.liftoff_mass / st.burnout_mass:.4g}. Gravity takes {fmt(lg)} m/s, drag {fmt(ld)} m/s and "
         f"steering {fmt(ls)} m/s, which leaves {fmt(pred)} m/s, and the flight reaches {fmt(flown)} m/s at "
         f"burnout ({100 * err:.2g}% apart)")
    budget = {"ideal": ideal, "gravity": lg, "drag": ld, "steering": ls, "predicted": pred, "flown": flown}
    return Check("delta v budget", "pass" if err < 5e-3 else ("warn" if err < 0.02 else "fail"), s,
                 {**budget, "error": err}), budget


def descent_check(fl: Flight) -> Check | None:
    """Sink rate under each parachute against the terminal speed sqrt(2 m g / (rho Cd A)) where it is read."""
    st = fl.study
    if not st.chutes:
        return None
    m = fl.end.y[M]
    parts, errs, settled = [], [], True
    drogue, main = st.chute("drogue"), st.chute("main")
    dep = fl.event("main deploy")
    if drogue and main and dep is not None and fl.event("drogue deploy") is not None:
        rho, g = _mid_air(st, dep.y[Z])
        vt = math.sqrt(2 * m * g / (rho * drogue.drag_area))
        sink = -dep.y[VZ]
        errs.append(abs(sink - vt) / vt)
        parts.append(f"under the drogue sqrt(2 m g / (rho Cd A)) gives {fmt(vt)} m/s at {_n(dep.y[Z])} m and the "
                     f"flight sinks at {fmt(sink)} m/s there")
    end = fl.end
    opened = [c for c in st.chutes if fl.event(f"{c.name} deploy") is not None]
    if not opened:
        return None
    rho, g = _mid_air(st, 0.0)
    vt = math.sqrt(2 * m * g / (rho * sum(c.drag_area for c in opened)))
    sink = -end.y[VZ]
    errs.append(abs(sink - vt) / vt)
    last = max(fl.event(f"{c.name} deploy").t for c in opened)
    settled = end.t - last > 5 * vt / g
    under = " and ".join(c.name for c in opened)
    formula = "it" if parts else "sqrt(2 m g / (rho Cd A))"
    parts.append(f"under the {under} {formula} gives {fmt(vt)} m/s at the ground against {fmt(sink)} m/s from "
                 "the run")
    err = max(errs)
    s = ". ".join(p[0].upper() + p[1:] for p in parts) + f" ({100 * err:.2g}% apart at most)"
    if not settled:
        return Check("parachute descent", "info", s + ". The last parachute opens too close to the ground for "
                                                      "the descent to settle at its terminal speed",
                     {"error": err})
    return Check("parachute descent", "pass" if err < 0.01 else ("warn" if err < 0.05 else "fail"), s,
                 {"error": err})


def energy_check(fl: Flight) -> Check:
    """The work done by thrust, drag and gravity, less the kinetic energy the burned propellant carried off, has
    to equal the change in kinetic energy from release to the ground."""
    y0, y1 = fl.events[0].y, fl.end.y
    ke0, ke1 = 0.5 * y0[M] * _speed(y0) ** 2, 0.5 * y1[M] * _speed(y1) ** 2
    wt, wd, wg, ep = y1[W_THRUST], y1[W_DRAG], y1[W_GRAV], y1[E_PROP]
    work = wt + wd + wg - ep
    scale = max(abs(wt), abs(wd), abs(wg), abs(ep), ke0, ke1, 1e-12)
    err = abs(ke1 - ke0 - work) / scale
    terms = []
    if fl.study.rocket:
        terms.append(f"thrust did {_energy(wt)} of work")
    terms.append(f"drag took {_energy(-wd)}")
    terms.append(f"gravity {'gave' if wg >= 0 else 'took'} {_energy(abs(wg))}")
    if fl.study.rocket:
        terms.append(f"the burned propellant carried off {_energy(ep)}")
    start = f" on top of the {_energy(ke0)} it started with" if ke0 > 1e-9 else ""
    what = "landing" if fl.study.chutes else "impact"
    s = (", ".join(terms[:-1]) + f" and {terms[-1]}, which leaves {_energy(ke0 + work)} of kinetic energy at "
         f"{what}{start}, and the {what} speed carries {_energy(ke1)} (apart by {100 * err:.2g}% of the largest "
         "term)")
    return Check("energy balance", "pass" if err < 1e-3 else ("warn" if err < 1e-2 else "fail"), s,
                 {"thrust_J": wt, "drag_J": wd, "gravity_J": wg, "propellant_J": ep, "ke_start_J": ke0,
                  "ke_end_J": ke1, "error": err})


def sanity_check(st: MotionStudy, fl: Flight, pts: list[dict]) -> Check:
    status, notes = "pass", []
    mach = max(p["mach"] for p in pts)
    notes.append(f"speeds stay under Mach {mach:.3g}")
    if st.rocket:
        burned = st.liftoff_mass - fl.end.y[M]
        ok = abs(burned - st.motor.propellant) <= 1e-6 * st.motor.propellant
        notes.append(f"all {fmt(st.motor.propellant)} kg of propellant burned" if ok else
                     f"{fmt(burned)} kg of the {fmt(st.motor.propellant)} kg propellant burned")
        status = "pass" if ok else "fail"
        tw = st.motor.impulse / st.motor.burn_time / (st.liftoff_mass * gravity(st.site_altitude))
        notes.append(f"an average thrust {tw:.3g} times the liftoff weight")
        if st.airframe is not None:
            rho = st.liftoff_mass / (st.area * st.airframe.length)
            notes.append(f"{fmt(st.liftoff_mass)} kg in a {_n(st.diameter * 1000)} mm by {_n(st.airframe.length)} m "
                         f"tube, {fmt(rho)} kg/m3 on average")
            if not 100 <= rho <= 3000:
                status = "warn"
                notes.append("which is far from any rocket, so check the mass and size units")
        if mach > 3:
            status = "warn" if status == "pass" else status
            notes.append("past Mach 3 the drag curve and Barrowman's method are out of their range")
    else:
        rho = st.mass / (math.pi / 6 * st.diameter ** 3)
        notes.append(f"{fmt(st.mass)} kg in a {fmt(st.diameter * 100)} cm body, {fmt(rho)} kg/m3 for a sphere")
        if not 1 <= rho <= 25000:
            status = "warn"
            notes.append("which no solid body has, so check the mass and size units")
        if mach > 0.8 and not isinstance(st.cd, list):
            status = "warn"
            notes.append("past Mach 0.8 one drag coefficient misses the transonic rise, so give a table against "
                         "Mach in a spec")
    apo = fl.event("apogee")
    if apo is not None and st.site_altitude + apo.y[Z] > 86000:
        status = "warn" if status == "pass" else status
        notes.append("above 86 km the standard atmosphere is extrapolated")
    return Check("units and sanity", status, ", ".join(notes))


# --- rocketry checks, which set the verdict --------------------------------------------------
def rocket_checks(st: MotionStudy, fl: Flight, pts: list[dict]) -> tuple[list[Check], dict]:
    out, v = [], {}
    if st.airframe is not None and st.airframe.cg is not None:
        cal = margin(st)
        cg_bo = burnout_cg(st)
        bm = barrowman(st.airframe, st.diameter)
        v["margin_cal"] = cal
        s = (f"{cal:.3g} calibers at liftoff, the center of pressure {fmt(bm['cp'])} m from the nose tip and the "
             f"balance point {fmt(st.airframe.cg)} m")
        if cg_bo is not None:
            v["margin_burnout_cal"] = margin(st, cg_bo)
            s += f", rising to {v['margin_burnout_cal']:.3g} at burnout as the propellant burns away"
        if cal < st.stability_min:
            s += f". Under {st.stability_min:g} caliber it can turn sideways off the rail"
        elif cal > 3:
            s += ". Over 3 calibers it is overstable and turns hard into the wind"
        s += ". Barrowman's method holds below about Mach 0.8"
        out.append(Check("stability margin", "pass" if cal >= st.stability_min else "fail", s,
                         {"cp_m": bm["cp"], "cg_m": st.airframe.cg, "margin_cal": cal}))
    rail = fl.event("rail exit")
    v["rail_exit_speed_mps"] = vr = _speed(rail.y)
    st_ = "pass" if vr >= st.rail_exit_min else ("warn" if vr >= RAIL_EXIT_FLOOR else "fail")
    out.append(Check("rail exit speed", st_, f"{fmt(vr)} m/s after {fmt(st.rail_length)} m of rail, "
                     + ("above" if vr >= st.rail_exit_min else "under")
                     + f" the {fmt(st.rail_exit_min)} m/s the fins need to steer"))
    q = peak(pts, "q")
    v["max_q_pa"] = q["q"]
    out.append(Check("max dynamic pressure", "info",
                     f"{fmt(q['q'] / 1000)} kPa at Mach {q['mach']:.3g} and {_n(q['y'][Z])} m, {q['t']:.3g} s after "
                     "ignition, the load the fins and their joints have to carry without fluttering"))
    a = peak(pts, "accel", ("rail", "boost"))
    v["max_accel_mps2"] = a["accel"]
    s = (f"{a['accel'] / 9.80665:.3g} g ({fmt(a['accel'])} m/s2) at {a['t']:.3g} s under thrust")
    for c in st.chutes:
        cp = [p for p in pts if p["phase"] == c.name]
        if cp:
            pk = max(cp, key=lambda p: p["felt"])
            s += (f". The {c.name} opens with {pk['felt'] / 9.80665:.2g} g of drag, pulling "
                  f"{fmt(pk['felt'] * pk['y'][M])} N on its harness")
    out.append(Check("max acceleration", "info", s))
    apo = fl.event("apogee")
    v["apogee_m"], v["apogee_t_s"] = apo.y[Z], apo.t
    s = f"{fmt(apo.y[Z])} m ({fmt(apo.y[Z] / 0.3048)} ft) above the ground at {apo.t:.3g} s"
    if st.target_apogee:
        off = (apo.y[Z] - st.target_apogee) / st.target_apogee
        s += f", {100 * abs(off):.2g}% {'over' if off > 0 else 'under'} the {fmt(st.target_apogee)} m target"
        out.append(Check("apogee", "pass" if abs(off) <= APOGEE_BAND else "warn", s))
    else:
        out.append(Check("apogee", "info", s))
    end = fl.end
    sink, ground = -end.y[VZ], _speed(end.y)
    v["landing_speed_mps"] = sink
    if st.chutes:
        under = " and ".join(c.name for c in st.chutes if fl.event(f"{c.name} deploy") is not None) or "nothing"
        s = (f"descends at {fmt(sink)} m/s under the {under}, " + ("under" if sink <= st.landing_limit else "over")
             + f" the {fmt(st.landing_limit)} m/s limit")
        if ground > sink * 1.02:
            s += f", and with the wind it moves {fmt(ground)} m/s over the ground"
        st_ = "pass" if sink <= st.landing_limit else ("warn" if sink <= 1.5 * st.landing_limit else "fail")
    else:
        s, st_ = f"hits the ground at {fmt(ground)} m/s with no parachute", "fail"
    out.append(Check("landing speed", st_, s))
    v["drift_m"] = end.y[X]
    side = "upwind" if (end.y[X] > 0) == (st.wind > 0) else "downwind"
    s = f"lands {_n(abs(end.y[X]))} m {side if st.wind else 'downrange'} of the pad"
    if st.wind:
        s += (f" in a {fmt(abs(st.wind))} m/s wind, after reaching {_n(abs(apo.y[X]))} m "
              f"{'upwind' if (apo.y[X] > 0) == (st.wind > 0) else 'downwind'} at apogee")
    dd = fl.event("drogue deploy")
    if dd is not None:
        s += f". The descent from apogee takes {fmt(end.t - dd.t)} s"
    out.append(Check("drift", "info", s))
    return out, v


def rocket_fixes(st: MotionStudy, fl: Flight, pts: list[dict], design: dict[str, Check], apogee: float) -> list[dict]:
    fixes = []
    af = st.airframe
    if "stability margin" in design and design["stability margin"].status == "fail":
        bm = barrowman(af, st.diameter)
        cg_goal = bm["cp"] - STABILITY_FIX * st.diameter
        x_nose = af.nose_length / 2
        add = st.liftoff_mass * (af.cg - cg_goal) / (cg_goal - x_nose)
        heavier = replace(st, mass=st.mass + add, airframe=replace(af, cg=cg_goal))
        new_apo = fly(heavier).event("apogee").y[Z]
        fixes.append({"kind": "nose weight", "title": f"Add {fmt(add * 1000)} g of nose weight",
                      "summary": f"{fmt(add * 1000)} g halfway down the nose cone moves the balance point to "
                                 f"{fmt(cg_goal)} m from the tip, for {STABILITY_FIX:g} calibers. It lowers the "
                                 f"apogee from {fmt(apogee)} m to {fmt(new_apo)} m.", "auto": False,
                      "params": {"mass_kg": add, "at_m": x_nose, "cg_m": cg_goal, "apogee_m": new_apo}})
        span = _fin_span_for(st, STABILITY_FIX)
        if span:
            area = replace(af.fins, span=span).area / af.fins.area
            fixes.append({"kind": "fin area", "title": f"Grow the fin span to {fmt(span * 100)} cm",
                          "summary": f"Fins {fmt(span * 100)} cm tall instead of {fmt(af.fins.span * 100)} cm, with "
                                     f"the same chords, have {fmt(100 * (area - 1))}% more area and move the center "
                                     f"of pressure back to {STABILITY_FIX:g} calibers behind the balance point. The "
                                     "extra fin weight sits aft and the extra drag costs a little apogee.",
                          "auto": False, "params": {"span_m": span}})
    vr = _speed(fl.event("rail exit").y)
    if vr < st.rail_exit_min:
        need = _distance_to_speed(pts, st.rail_exit_min)
        if need:
            fixes.append({"kind": "rail", "title": f"Use a rail at least {fmt(need)} m long",
                          "summary": f"The rocket reaches {fmt(st.rail_exit_min)} m/s after about {fmt(need)} m of "
                                     "travel, so a rail that long, or a motor with more thrust at ignition, lets "
                                     "the fins steer from the moment it leaves.", "auto": False,
                          "params": {"rail_length_m": need}})
    main = st.chute("main")
    sink = -fl.end.y[VZ]
    if main and sink > st.landing_limit:
        rho, g = _mid_air(st, 0.0)
        others = sum(c.drag_area for c in st.chutes if c is not main)
        cda = 2 * fl.end.y[M] * g / (rho * SOFT_LANDING ** 2) - others
        d = math.sqrt(4 * cda / (math.pi * main.cd))
        fixes.append({"kind": "main parachute", "title": f"Use a {fmt(d)} m main parachute",
                      "summary": f"A {fmt(d)} m main with the same drag coefficient ({main.cd:g}) lands the "
                                 f"{fmt(fl.end.y[M])} kg rocket at {fmt(SOFT_LANDING)} m/s, against {fmt(sink)} m/s "
                                 f"under the {fmt(main.diameter)} m one.", "auto": False, "params": {"diameter_m": d}})
    return fixes


def _fin_span_for(st: MotionStudy, goal: float) -> float | None:
    """The fin span that gives this margin with the same chords and sweep, by bisection."""
    f = st.airframe.fins
    lo, hi = f.span, 4 * f.span
    if margin(st, fins=replace(f, span=hi)) < goal:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin(st, fins=replace(f, span=mid)) < goal else (lo, mid)
    return hi


def _distance_to_speed(pts: list[dict], speed: float) -> float | None:
    """Distance along the path at which the rocket first reaches this speed."""
    s, last = 0.0, None
    for p in pts:
        y = p["y"]
        if last is not None:
            s += math.hypot(y[X] - last[X], y[Z] - last[Z])
        if _speed(y) >= speed:
            return s
        last = y
    return None
