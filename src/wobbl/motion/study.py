"""Motion: when and how fast a body lands, and whether a rocket is safe to fly.

"How fast does a rock hit the ground from a 300 m cliff?" becomes a granite sphere falling through the 1976
standard atmosphere under gravity that falls off with altitude. "Is this rocket stable, and will it land
softly?" becomes a flight from the rail through burnout, apogee and both parachutes to the ground. Wobbl
integrates it with scipy, flies it again with a tolerance ten times tighter, compares it with closed form
answers wherever the model reduces to one, closes an energy balance and a delta v budget, and reports the
numbers a rocketry person checks before flying.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .checks import (_n, _speed, delta_v_check, descent_check, drag_fall_check, energy_check, key_numbers, points,
                     rocket_checks, rocket_fixes, sanity_check, step_convergence, vacuum_check)
from .flight import M, RTOL, X, Z, Flight, fly
from .spec import MotionStudy

COLUMNS = ["t_s", "x_m", "alt_m", "speed_mps", "accel_mps2", "mass_kg", "mach", "q_pa", "phase"]
MAX_ROWS = 500


@dataclass
class MotionOutcome(GenericOutcome):
    events: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0
    model: dict = field(default_factory=dict)

    def write(self, out_dir, sections_extra: list[str] | None = None) -> "MotionOutcome":
        from ..report.html import table

        rows = [[e["name"], f"{e['t_s']:.3f} s", f"{_n(e['alt_m'])} m", f"{_n(e['x_m'])} m",
                 f"{_n(e['speed_mps'])} m/s"] for e in self.events]
        extra = ["<h2>Events</h2>" + table(["event", "time", "altitude", "downrange", "speed"], rows, {1, 2, 3, 4})]
        dv = self.values.get("delta_v")
        if dv:
            extra.append("<h2>Delta v to burnout</h2>" + table(["", "m/s"], [
                ["Tsiolkovsky, ve ln(m0 / m1)", fmt(dv["ideal"])], ["lost to gravity", fmt(-dv["gravity"])],
                ["lost to drag", fmt(-dv["drag"])], ["lost to steering", fmt(-dv["steering"])],
                ["left at burnout", fmt(dv["predicted"])], ["flown at burnout", fmt(dv["flown"])]], {1}))
        return super().write(out_dir, extra + (sections_extra or []))


# --- outputs ---------------------------------------------------------------------------------
def table_rows(fl: Flight, pts: list[dict]) -> list[dict]:
    """At most MAX_ROWS rows of the trajectory: a row at every event, the rest spread along the curve of time,
    distance, altitude and speed so the fast parts get more of them, and a last row at rest on the ground."""
    times = np.array([p["t"] for p in pts])
    events = {0, len(pts) - 1}
    for e in fl.events:
        hit = np.nonzero(np.abs(times - e.t) <= 1e-9)[0]
        if len(hit):
            events.add(int(hit[0]))
    keep = set(events)
    budget = MAX_ROWS - 1 - len(keep)
    if len(pts) > budget + len(keep):
        cols = np.array([[p["t"], p["y"][X], p["y"][Z], _speed(p["y"])] for p in pts])
        span = np.ptp(cols, axis=0)
        span[span == 0] = 1.0
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cols / span, axis=0), axis=1))])
        keep |= set(np.searchsorted(s, np.linspace(0, s[-1], max(budget, 0))).clip(0, len(pts) - 1).tolist())
    else:
        keep = set(range(len(pts)))
    rows = []
    for i in sorted(keep):
        r = _row(pts[i])
        if rows and r["t_s"] == rows[-1]["t_s"] and r["phase"] == rows[-1]["phase"]:
            if i in events:      # an event row wins over the plain row it lands on
                rows[-1] = r
            continue
        rows.append(r)
    rows = rows[:MAX_ROWS - 1]
    rows.append({**rows[-1], "speed_mps": 0.0, "accel_mps2": 0.0, "mach": 0.0, "q_pa": 0.0, "phase": "landed",
                 "alt_m": 0.0})
    return rows


def _row(p: dict) -> dict:
    y = p["y"]
    r = lambda v, k: round(float(v), k) + 0.0  # noqa: E731  no negative zeros
    return {"t_s": r(p["t"], 4), "x_m": r(y[X], 3), "alt_m": r(y[Z], 3), "speed_mps": r(_speed(y), 3),
            "accel_mps2": r(p["accel"], 3), "mass_kg": r(y[M], 4), "mach": r(p["mach"], 4), "q_pa": r(p["q"], 1),
            "phase": p["phase"]}


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def event_list(fl: Flight) -> list[dict]:
    return [{"name": e.name, "t_s": round(e.t, 4) + 0.0, "alt_m": round(float(e.y[Z]), 2) + 0.0,
             "x_m": round(float(e.y[X]), 2) + 0.0, "speed_mps": round(_speed(e.y), 3)} for e in fl.events]


def effects(st: MotionStudy) -> list[str]:
    out = ["uniform gravity" if st.uniform_gravity else "gravity falling off with altitude"]
    if st.air != "vacuum":
        out.append("1976 standard atmosphere" if st.air == "standard" else "constant air density")
        out.append("air drag")
        if isinstance(st.cd, list):
            out.append("drag coefficient against mach")
        elif st.transonic:
            out.append("transonic drag rise")
        if st.wind:
            out.append("wind")
    if st.rocket:
        out += ["launch rail", "thrust curve", "propellant burn"]
        if st.air != "vacuum":
            out.append("weathercocking")
    out += [f"{c.name} parachute" for c in st.chutes]
    return out


def assumptions(st: MotionStudy) -> list[str]:
    out = ["A point mass in a vertical plane, so it does not spin, tumble or move sideways, and its drag acts "
           "against the air rushing past it."]
    cd = st.drag_coefficient()
    if st.air != "vacuum":
        if isinstance(st.cd, list):
            out.append(f"Drag coefficient from the table against Mach, {fmt(cd(0.0))} at low speed, on a frontal "
                       f"area {fmt(st.diameter * 100)} cm across.")
        elif st.transonic:
            out.append(f"Drag coefficient {fmt(cd(0.0))} below Mach 0.6, raised through the transonic range to "
                       f"{fmt(cd(1.1))} near Mach 1.1 by a generic curve for a slender finned rocket, the same "
                       "whether the motor burns or not.")
        else:
            out.append(f"Drag coefficient {fmt(cd(0.0))} on a frontal area {fmt(st.diameter * 100)} cm across, the "
                       "same at every speed.")
    if st.air == "standard":
        out.append(f"The 1976 US standard atmosphere from {_n(st.site_altitude)} m above sea level, a standard day.")
    elif st.air != "vacuum":
        out.append(f"Air of constant density {fmt(float(st.air))} kg/m3.")
    out.append("Gravity " + ("held at its ground value" if st.uniform_gravity else
                             "falls off with the square of the distance to the Earth's center")
               + ", over a flat Earth that does not rotate.")
    if st.wind:
        out.append(f"A steady wind of {fmt(abs(st.wind))} m/s at every height, blowing back toward the start.")
    elif st.air != "vacuum":
        out.append("Still air.")
    if st.rocket:
        mo = st.motor
        out.append(f"Thrust follows the {mo.name} curve in straight lines between its points, {fmt(mo.impulse)} N s "
                   f"over {fmt(mo.burn_time)} s, and propellant burns in proportion to thrust, so all "
                   f"{fmt(mo.propellant)} kg is gone at burnout.")
        out.append(f"The rocket travels the full {fmt(st.rail_length)} m of rail before it leaves it, tilted "
                   f"{fmt(90 - st.angle)} deg from vertical.")
        if st.airframe is not None and st.airframe.cg is not None:
            out.append("Off the rail it turns into the wind it feels at its weathercock frequency, from Barrowman's "
                       "normal force and margin and the pitch inertia of a uniform rod of its length and mass, "
                       "with no lift at angle of attack and no overshoot.")
        else:
            out.append("Off the rail it points into the wind it feels at once, which overstates how far it "
                       "weathercocks. Give [nose], [fins] and the balance point for its turning rate.")
    if st.chutes:
        out.append("Each parachute opens over the time it takes to travel eight of its diameters, its drag area "
                   "growing with the square of time, and the drogue stays on under the main.")
        out.append("Under the parachutes only their drag counts, so the descent is slightly faster than with the "
                   "airframe's drag added, which errs on the safe side for landing speed.")
    return out


# --- the run ---------------------------------------------------------------------------------
@dataclass
class Analysis:
    """A flight with its verification and rocketry checks, before anything is written."""
    flight: Flight
    points: list[dict]
    checks: list[Check]             # verification, which sets the confidence
    design: list[Check]             # rocketry, which sets the verdict
    confidence: str
    reasons: list[str]
    verdict: str
    headline: str
    metrics: list[tuple[str, str]]
    values: dict
    fixes: list[dict]
    vacuum: dict | None = None      # the same body's fall without air, for the chart


def analyze(st: MotionStudy, progress=None) -> Analysis:
    """Fly the study, fly it again with a tolerance ten times tighter and run every check."""
    say = progress or (lambda a, b: None)
    say("flight", "integrating from " + ("ignition" if st.rocket else "release") + " to the ground")
    fl = fly(st)
    pts = points(fl)
    key = key_numbers(fl, pts)
    say("verify", "flying again with a tolerance ten times tighter")
    tight = fly(st, RTOL / 10)
    checks = [step_convergence(key, key_numbers(tight, points(tight)))]
    vacuum, values = None, {}
    if st.rocket:
        dv, values["delta_v"] = delta_v_check(fl)
        checks.append(dv)
    else:
        hand = drag_fall_check(st, fl)
        if hand:
            checks.append(hand)
        if st.air != "vacuum":
            vac, vacuum = vacuum_check(st, fl)
            checks.append(vac)
    desc = descent_check(fl)
    if desc:
        checks.append(desc)
    checks += [energy_check(fl), sanity_check(st, fl, pts)]
    level, reasons = confidence(checks)
    design, fixes = [], []
    end, apo = fl.end, fl.event("apogee")
    if st.rocket:
        design, rv = rocket_checks(st, fl, pts)
        values.update(rv)
        by = {c.name: c for c in design}
        fixes = rocket_fixes(st, fl, pts, by, apo.y[Z])
        verdict, head, metrics = _rocket_answer(st, fl, by, rv, pts)
    else:
        verdict, head, metrics = _body_answer(st, fl, checks)
    values.update({"diameter_m": st.diameter, "flight_time_s": end.t, "impact_speed_mps": _speed(end.y),
                   "distance_m": end.y[X],
                   "max_speed_mps": max(_speed(p["y"]) for p in pts), "max_mach": max(p["mach"] for p in pts)})
    if apo is not None:
        values.setdefault("apogee_m", apo.y[Z])
    return Analysis(fl, pts, checks, design, level, reasons, verdict, head, metrics, values, fixes, vacuum)


def run_motion(st: MotionStudy, out_dir: str | Path, progress=None) -> MotionOutcome:
    """Analyze the flight and write outcome.json, summary.txt, report.html, trajectory.csv and the charts."""
    import scipy

    from .charts import flight_chart, path_chart

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    a = analyze(st, say)
    fl = a.flight
    say("outputs", "trajectory table, charts and report")
    # the peaks need a few dozen points per segment, but a flight of few segments, like a drop, gets more for its
    # table so it still fills the rows it is allowed
    dense = MAX_ROWS // len(fl.segments)
    rows = table_rows(fl, points(fl, dense) if dense > 60 else a.points)
    write_csv(out / "trajectory.csv", rows)
    events = event_list(fl)
    img = {}
    try:
        img["flight"] = flight_chart(out / "images" / "flight.png", rows, events, a.vacuum)
        if st.rocket or abs(fl.end.y[X]) > 1.0:
            img["path"] = path_chart(out / "images" / "path.png", rows, events, st.wind)
    except Exception as e:  # noqa: BLE001
        img["error"] = str(e)
    values = {**a.values, "trajectory": "trajectory.csv",
              "_figures": [["flight", "Altitude and speed over the flight, by phase"],
                           ["path", "The path through the air"]]}
    states = 4 + (1 if st.rocket else 0) + (1 if fl.model.cn is not None else 0)
    o = MotionOutcome("motion", st.source or st.name, st.question, a.verdict, a.headline, a.metrics, a.confidence,
                      a.reasons, a.checks + a.design, assumptions(st), values, img, a.fixes, list(st.notes),
                      solver=f"scipy {scipy.__version__} solve_ivp (DOP853)")
    o.events = events
    o.model = {"states": states, "effects": effects(st)}
    o.runtime_s = o.t_total = round(time.time() - t0, 3)
    if st.rocket:
        v = o.values
        o.criteria = [c for c in (
            {"name": "stability margin", "value": "margin_cal", "limit": st.stability_min, "sense": "min"},
            {"name": "rail exit speed", "value": "rail_exit_speed_mps", "limit": st.rail_exit_min, "sense": "min"},
            {"name": "landing speed", "value": "landing_speed_mps", "limit": st.landing_limit, "sense": "max"})
            if isinstance(v.get(c["value"]), (int, float))]
    o.write(out)
    _scene(st, out)
    return o


def _scene(st: MotionStudy, out: Path):
    """The run's interactive 3D view (wobbl.viz), drawn with the body's real size and shape from the study."""
    try:
        from ..viz.adapters.motion import build
        from ..viz.embed import after_report
        if build(out, st):
            after_report(out)
    except Exception:  # noqa: BLE001  a view that cannot be built never fails the run
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise


def _body_answer(st: MotionStudy, fl: Flight, checks: list[Check]) -> tuple[str, str, list]:
    end = fl.end
    v, t = _speed(end.y), end.t
    vac = next((c for c in checks if c.name.startswith("vacuum")), None)
    hand = next((c for c in checks if c.name == "hand calculation"), None)
    if st.speed == 0:
        head = f"Hits the ground {t:.3g} s after release at {fmt(v)} m/s ({fmt(v * 3.6)} km/h)."
        metrics = [("fall time", f"{t:.3g} s"), ("impact speed", f"{fmt(v)} m/s")]
        if vac is not None:
            tv, vv = vac.detail["time_s"], vac.detail["speed_mps"]
            head += (f" Without air it would land after {tv:.3g} s at {fmt(vv)} m/s, so drag takes "
                     f"{100 * (1 - v / vv):.2g}% off the impact speed.")
            metrics.append(("without air", f"{tv:.3g} s, {fmt(vv)} m/s"))
        if st.wind and abs(end.y[X]) > 1.0:
            head += f" The wind carries it {_n(abs(end.y[X]))} m."
    else:
        apo = fl.event("apogee")
        top = apo.y[Z] if apo is not None else st.height
        head = f"Lands {_n(end.y[X])} m away {t:.3g} s after launch at {fmt(v)} m/s, after rising to {_n(top)} m."
        metrics = [("range", f"{_n(end.y[X])} m"), ("flight time", f"{t:.3g} s"), ("peak height", f"{_n(top)} m"),
                   ("impact speed", f"{fmt(v)} m/s")]
        if vac is not None:
            head += f" Without air it would land {_n(vac.detail['range_m'])} m away."
    if hand is not None and "terminal_mps" in hand.detail:
        metrics.append(("terminal speed", f"{fmt(hand.detail['terminal_mps'])} m/s"))
    return "ok", head, metrics


def _rocket_answer(st: MotionStudy, fl: Flight, by: dict[str, Check], v: dict, pts) -> tuple[str, str, list]:
    bad = [n for n, c in by.items() if c.status == "fail"]
    weak = [n for n, c in by.items() if c.status == "warn"]
    verdict = "fails" if bad else ("marginal" if weak else "ok")
    mach = max(p["mach"] for p in pts)
    apo, end = v["apogee_m"], fl.end
    side = "upwind" if (end.y[X] > 0) == (st.wind > 0) else "downwind"
    where = f"{_n(abs(end.y[X]))} m {side if st.wind else 'downrange'} of the pad"
    flight = (f"reaches {fmt(apo)} m ({fmt(apo / 0.3048)} ft) at Mach {mach:.3g} and lands {where} at "
              f"{fmt(v['landing_speed_mps'])} m/s")
    if verdict == "ok":
        extra = f"a {v['margin_cal']:.3g} caliber margin and " if "margin_cal" in v else ""
        head = f"{flight[0].upper()}{flight[1:]}, with {extra}{fmt(v['rail_exit_speed_mps'])} m/s off the rail."
    else:
        head = " ".join(_issue(st, n, v) for n in bad + weak) + f" It {flight}."
    metrics = [("apogee", f"{fmt(apo)} m"), ("max Mach", f"{mach:.3g}")]
    if "margin_cal" in v:
        metrics.append(("stability", f"{v['margin_cal']:.3g} cal"))
    metrics += [("off the rail", f"{fmt(v['rail_exit_speed_mps'])} m/s"),
                ("landing", f"{fmt(v['landing_speed_mps'])} m/s"), ("drift", f"{_n(abs(end.y[X]))} m")]
    return verdict, head, metrics


def _issue(st: MotionStudy, name: str, v: dict) -> str:
    """One sentence for a rocketry check that did not pass."""
    if name == "stability margin":
        return (f"Unstable off the rail, with a {v['margin_cal']:.3g} caliber margin under the "
                f"{st.stability_min:g} caliber minimum.")
    if name == "rail exit speed":
        return (f"Leaves the rail at {fmt(v['rail_exit_speed_mps'])} m/s, under the {fmt(st.rail_exit_min)} m/s "
                "its fins need to steer.")
    if name == "landing speed":
        if not st.chutes:
            return "Comes down with no parachute."
        return f"Lands at {fmt(v['landing_speed_mps'])} m/s, over the {fmt(st.landing_limit)} m/s limit."
    off = (v["apogee_m"] - st.target_apogee) / st.target_apogee
    return f"Misses its {fmt(st.target_apogee)} m target by {100 * abs(off):.2g}%."
