"""Mechanisms: whether the servos lift it, how far a linkage swings and where it binds, what a gearbox gives, how
hard parts hit and whether something tips.

A template, a spec file or an MJCF or URDF model becomes a Mechanism and MuJoCo runs it. Every answer is checked
the same way: against a closed form wherever the case reduces to one, again at half the time step, with its energy
account closed over every force, and with its contacts and constraints inspected. Bodies are rigid and contacts
are stiff springs, and every report says so.
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence
from .sim import Run
from .spec import Mechanism, SpecError

MAX_ROWS = 500
LIMITS = [
    "Bodies are rigid, so links do not bend. A long or thin printed link flexes and rings more than this shows.",
    "Contacts are stiff springs that let parts sink into each other by the depth the contact check reports.",
    "Joints and gears have no backlash, and friction acts only where the inputs give it.",
]


@dataclass
class MechanismOutcome(GenericOutcome):
    events: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0
    model: dict = field(default_factory=dict)

    def write(self, out_dir, sections_extra: list[str] | None = None) -> "MechanismOutcome":
        from ..report.html import table

        rows = [[e["name"], f"{e['t_s']:.3f} s", e.get("detail", "")] for e in self.events]
        extra = ["<h2>Events</h2>" + table(["event", "time", "detail"], rows, {1})] if rows else []
        return super().write(out_dir, extra + (sections_extra or []))


@dataclass
class Answer:
    """What one analysis hands back before the outcome is written."""
    verdict: str
    headline: str
    metrics: list[tuple[str, str]]
    checks: list[Check]
    assumptions: list[str]
    values: dict = field(default_factory=dict)
    fixes: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    figures: list[list[str]] = field(default_factory=list)
    images: dict = field(default_factory=dict)
    run: Run | None = None                 # the run the trajectory table comes from
    extra: list[tuple[str, str]] = field(default_factory=list)      # (level, reason) for the confidence
    model: dict = field(default_factory=dict)
    kin: tuple | None = None               # (mechanism, run) of a slow ideal drive, where the drive checks reuse it


def event(name: str, t: float, detail: str = "") -> dict:
    return {"name": name, "t_s": round(float(t), 4) + 0.0, "detail": detail}


# --- checks every answer shares ---------------------------------------------------------------------------
def step_check(a: float, b: float, label: str, unit: str = "", tol: tuple = (0.01, 0.05)) -> Check:
    d = abs(a - b) / max(abs(b), 1e-12)
    st = "pass" if d < tol[0] else ("warn" if d < tol[1] else "fail")
    return Check("time step convergence", st, f"{label} {fmt(a)}{unit} with the normal step and {fmt(b)}{unit} "
                                              f"with half the step ({100 * d:.2g}% apart)",
                 {"normal": a, "half": b, "change": d})


def typical_size(run: Run) -> float:
    """A typical part size from the geoms, for judging penetration, drift and energy."""
    sizes = [max(g["size"][:3]) for g in run.model_info["geoms"] if g["type"] != "plane"]
    return float(np.median(sizes)) * 2 if sizes else 0.1


def energy_check(run: Run, passive: bool = False) -> Check:
    """With nothing driving or damping it the energy has to stay put; otherwise the work of every force has to
    add up to the change in kinetic and potential energy. A hundredth of the energy to lift the whole mechanism
    by its own size sets the smallest scale, so a run where almost nothing moves is not judged on its settling."""
    floor = 1e-2 * run.model_info.get("mass_kg", 0.0) * 9.80665 * typical_size(run)
    end, worst, info = run.energy_residual(floor)
    st = "pass" if worst < 0.01 else ("warn" if worst < 0.03 else "fail")
    if passive:
        return Check("energy conservation", st, f"with nothing driving or damping it, kinetic plus potential energy "
                                                f"stayed within {100 * worst:.2g}% of the largest energy in the run",
                     {**info, "worst": worst})
    parts = ", ".join(f"{k} {fmt(v)} J" for k, v in info.items() if k not in ("scale_J", "delta_E_J") and abs(v) > 1e-9)
    return Check("energy account", st, f"the work of the forces ({parts or 'none'}) matches the change in kinetic and "
                                       f"potential energy, {fmt(info['delta_E_J'])} J, to {100 * worst:.2g}% of the "
                                       "largest energy flow", {**info, "worst": worst, "end": end})


def contact_check(run: Run, size: float, support_only: bool = False, settled: bool = False) -> Check | None:
    """MuJoCo's contacts are springs, so a deep sink means the contact is too soft for the numbers to hold.
    ``support_only`` judges only what rests on the floor, for a run where a part that fell is already the answer,
    and ``settled`` only the contacts at the end, for forces read once it has come to rest. A sink within the
    width a model's own contact settings give it is the soft pad or foot the model means."""
    if settled:
        last = run.model_info.get("final_contacts", [])
        if not last:
            return None
        deep = max(last, key=lambda c: c["depth"])
        depth, width = deep["depth"], deep.get("width", 0.0)
    else:
        contacts = [c for c in run.contacts.values() if c["support"] or not support_only]
        if not contacts:
            return None
        deep = max(contacts, key=lambda c: c["depth_m"])
        depth, width = deep["depth_m"], deep.get("width_m", 0.0)
    if width > 0.002 and depth <= 1.2 * width:
        return Check("contact settings", "pass", f"the deepest contact sinks {fmt(depth * 1000)} mm, within the "
                                                 f"{fmt(width * 1000)} mm the model's own contact settings let it "
                                                 "sink, as a soft pad or foot does",
                     {"depth_m": depth, "width_m": width})
    rel = depth / max(size, 1e-9)
    st = "pass" if rel < 0.01 or depth < 2e-4 else ("warn" if rel < 0.05 else "fail")
    return Check("contact settings", st, f"the deepest contact sinks {fmt(depth * 1000)} mm, {100 * rel:.2g}% of the "
                                         f"{fmt(size * 1000)} mm part, so the soft contact model is not hiding a "
                                         "real overlap" if st == "pass" else
                 f"the deepest contact sinks {fmt(depth * 1000)} mm, {100 * rel:.2g}% of the part, which is deep "
                 "enough to change the forces and positions", {"depth_m": depth, "relative": rel})


def constraint_check(run: Run, scale: float, unit: str = "mm", label: str = "loop") -> Check | None:
    if not run.model_info.get("neq"):
        return None
    err = float(np.max(run.eq_err)) if len(run.eq_err) else 0.0
    shown, ref = (err * 1000, scale * 1000) if unit == "mm" else (math.degrees(err), math.degrees(scale))
    rel = err / max(scale, 1e-12)
    st = "pass" if rel < 1e-3 else ("warn" if rel < 1e-2 else "fail")
    return Check("constraint drift", st, f"the {label} held together to {fmt(shown)} {unit} at worst "
                                         f"({100 * rel:.2g}% of {fmt(ref)} {unit})", {"max": err, "relative": rel})


def warnings_check(run: Run) -> Check | None:
    w = run.model_info.get("warnings") or {}
    if not w:
        return None
    return Check("solver warnings", "fail", "MuJoCo warned about " + ", ".join(f"{k} ({v} times)" for k, v in w.items())
                 + ". The run went unstable or ran out of room, so its numbers are not reliable", w)


def model_info(m: Mechanism, run: Run) -> dict:
    c = m.contact
    contact = (f"linear spring and dashpot, {fmt(math.sqrt(c['stiffness']) / (2 * math.pi))} Hz natural frequency "
               f"per unit mass, damping ratio {fmt(c.get('damping', 0) / (2 * math.sqrt(c['stiffness'])))}"
               if "stiffness" in c else f"stiff contact settling in {fmt(c.get('timeconst', 0.005) * 1000)} ms")
    return {"bodies": run.model_info["nbody"] - 1, "joints": run.model_info["njnt"],
            "degrees_of_freedom": run.model_info["nv"], "constraints": run.model_info["neq"],
            "timestep_s": run.dt, "steps": run.steps, "integrator": "semi implicit Euler", "contact": contact}


# --- outputs ----------------------------------------------------------------------------------------------
def write_trajectory(path: Path, run: Run, events: list[dict], slide: set | None = None) -> None:
    """trajectory.csv: time, then each joint's angle, speed and drive torque, then where each point mass is. At most
    MAX_ROWS rows, evenly spread in time, with a row at every event."""
    slide = slide or set()
    cols = ["t_s"]
    for j in run.joints:
        key = j.replace(" ", "_")
        cols += [f"{key}_mm", f"{key}_mm_s", f"{key}_N"] if j in slide else [f"{key}_deg", f"{key}_rpm", f"{key}_Nm"]
    pts = list(run.points)
    for p in pts:
        key = p.replace(" ", "_")
        cols += [f"{key}_x_m", f"{key}_y_m", f"{key}_z_m"]
    free = list(run.free.values())
    for f in free:
        key = f["body"].replace(" ", "_")
        cols += [f"{key}_x_m", f"{key}_y_m", f"{key}_z_m", f"{key}_speed_mps", f"{key}_tilt_deg"]
    touching = any(not c["support"] for c in run.contacts.values())
    if touching:
        cols.append("contact_N")
    n = len(run.t)
    keep = set(np.linspace(0, n - 1, min(n, MAX_ROWS - len(events))).round().astype(int).tolist())
    for e in events:
        keep.add(int(np.argmin(np.abs(run.t - e["t_s"]))))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for i in sorted(keep):
            row = [round(float(run.t[i]), 5) + 0.0]
            for c, j in enumerate(run.joints):
                if j in slide:
                    row += [run.q[i, c] * 1000, run.v[i, c] * 1000, run.tau[i, c]]
                else:
                    row += [math.degrees(run.q[i, c]), run.v[i, c] * 60 / (2 * math.pi), run.tau[i, c]]
            for p in pts:
                row += list(run.points[p][i])
            for f in free:
                row += [*f["xy"][i], f["z"][i], float(np.linalg.norm(f["vel"][i])), f["tilt"][i]]
            if touching:
                row.append(run.peak_force[i])
            w.writerow([row[0]] + [round(float(x), 5) + 0.0 for x in row[1:]])


PLAYBACK_BYTES = 300_000


def write_playback(path: Path, run: Run, mech: Mechanism, o: "MechanismOutcome") -> int:
    """playback.json, enough to animate the run truthfully: every geom as a primitive with its pose at up to 150
    instants, and each joint's angle, torque, needed torque and motor limit at the same instants. Returns its
    size in bytes, kept under PLAYBACK_BYTES by dropping every other instant if it has to."""
    import json

    P = run.poses
    if not P:
        return 0
    t = P["t"]
    keep = np.arange(len(t))
    geoms = run.model_info["geoms"]
    drives = run.model_info.get("drives") or {d.joint: d.at_joint for d in mech.drives if d.motor is not None}

    def at(series):
        return np.interp(t[keep], run.t, series)

    def r(x, k):
        return [round(float(v), k) + 0.0 for v in x]

    while True:
        bodies = []
        for g, info in enumerate(geoms):
            pos, quat = P["pos"][keep, g], P["quat"][keep, g]
            if any(info.get("offset", (0, 0, 0))):
                pos = pos + _rotate(quat, np.asarray(info["offset"]))
            fixed = bool(np.ptp(pos, axis=0).max() < 1e-7 and np.ptp(quat, axis=0).max() < 1e-7)
            rgba = info["rgba"]
            bodies.append({"name": info["name"], "shape": info["type"], "size": r(info["size"], 5),
                           "color": "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgba[:3]),
                           "static": fixed, **({"bounds": True} if info.get("bounds") else {}),
                           "pos": r(pos[0], 5) if fixed else [r(p, 5) for p in pos],
                           "quat": r(quat[0], 4) if fixed else [r(q, 4) for q in quat]})
        series = {}
        for c, j in enumerate(run.joints):
            s = {"angle": {"unit": "rad", "values": r(at(run.q[:, c]), 4)},
                 "speed": {"unit": "rad/s", "values": r(at(run.v[:, c]), 4)},
                 "torque": {"unit": "N m", "values": r(at(run.tau[:, c]), 4)}}
            if j in drives:
                need = at(run.tau_need[:, c])
                w = at(run.v[:, c])
                lim = [drives[j].limit(float(wi))[1 if ni >= 0 else 0] for wi, ni in zip(w, need)]
                s["needed"] = {"unit": "N m", "values": r(need, 4)}
                s["limit"] = {"unit": "N m", "values": r(lim, 4),
                              "note": "the motor's torque at that speed, in the direction the joint needs it"}
            series[j] = s
        extra = {}
        if any(not c["support"] for c in run.contacts.values()):
            extra["contact force"] = {"unit": "N", "values": r(at(run.peak_force), 3)}
        for j, mot in drives.items():
            c = run.joints.index(j)
            w_j = at(run.v[:, c])
            slide = j in {x.name for x in mech.joints if x.kind == "slide"}
            extra[f"{j} speed"] = {"unit": "mm/s" if slide else "rpm",
                                   "values": r(w_j * (1000 if slide else 60 / (2 * math.pi)), 3)}
            amps = mot.current(at(run.tau[:, c]))
            if amps is not None:
                extra[f"{j} current"] = {"unit": "A", "values": r(amps, 4)}
                if mot.voltage > 0:
                    extra[f"{j} electrical power"] = {"unit": "W", "values": r(mot.voltage * amps, 3)}
        for f in run.free.values():
            extra[f"{f['body']} tilt"] = {"unit": "deg", "values": r(at(f["tilt"]), 3)}
        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        doc = {
            "manifest": {
                "format": "wobbl-playback", "kind": "mechanism", "version": 1, "generator": "wobbl mechanism",
                "domain": "mechanism", "title": mech.name,
                "question": o.question, "verdict": o.verdict, "headline": o.headline, "solver": o.solver,
                "frames": int(len(keep)), "duration_s": round(dur, 6),
                "suggested_seconds": round(min(max(dur, 2.0), 12.0), 3),
                "units": {"length": "m", "time": "s", "angle": "rad", "torque": "N m", "force": "N"},
                "axes": "z is up; quaternions are w, x, y, z and turn each shape from its own frame into the world",
                "shapes": "sizes follow MuJoCo: sphere [radius], capsule and cylinder [radius, half length] along "
                          "their own z axis, box [half x, half y, half z], plane [half x, half y] at z = pos; a "
                          "body marked bounds is a mesh drawn as the box that bounds it",
                "source": "recorded by the MuJoCo run itself at evenly spaced steps; series are interpolated "
                          "from the run's samples at the same instants",
            },
            "t": r(t[keep], 6),
            "bodies": bodies,
            "joints": series,
            "series": extra,
            "events": o.events,
        }
        text = json.dumps(doc, separators=(",", ":"))
        if len(text) <= PLAYBACK_BYTES or len(keep) < 20:
            break
        keep = keep[::2]
    Path(path).write_text(text, encoding="utf-8")
    return len(text)


GENERAL = {"payload", "payload_at", "servo", "motor", "base", "passive", "analyses", "quick"}


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vector ``v`` turned by each unit quaternion (w, x, y, z) in ``q``."""
    w, u = q[:, :1], q[:, 1:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def load_target(target: str, params: dict) -> Mechanism:
    """A spec file, or an MJCF or URDF model as it is (its own gravity, contacts and time step). Either takes the
    general values: a payload and where it sits, a servo for every driven joint, a free base, passive and the
    analyses to run."""
    from . import quantities as Q
    from .spec import read_spec

    p = Path(target)
    if not p.exists():
        raise SpecError(f"no mechanism template or file called '{target}' (the templates are pendulum, arm, fourbar, "
                        "gears, impact and tip)")
    suffix = p.suffix.lower()
    if suffix in (".toml", ".json"):
        extra = set(params) - GENERAL
        if extra:
            raise SpecError(f"a spec file carries its own values, so edit the file rather than setting "
                            f"{', '.join(sorted(extra))}")
        return read_spec(str(p))
    if suffix in (".xml", ".mjcf", ".urdf"):
        extra = set(params) - GENERAL - {"duration", "timestep"}
        if extra:
            raise SpecError(f"a model file takes duration, timestep, {', '.join(sorted(GENERAL))}, not "
                            f"{', '.join(sorted(extra))}. To drive its joints along a move, write a spec with model = "
                            f'"{p.name}" and a move')
        return Mechanism(name=p.stem, model_file=str(p.resolve()),
                         duration=Q.time(params["duration"]) if params.get("duration") else 0.0,
                         timestep=Q.time(params["timestep"], "ms") if params.get("timestep") else 0.0)
    raise SpecError(f"'{p.name}' is not a mechanism spec (.toml, .json) or model (.xml, .urdf)")


def run_mechanism(target: str, params: dict | None, out_dir: str | Path, question: str = "",
                  progress=None, drive: bool = False) -> MechanismOutcome:
    """Build, run, check and write a mechanism study: outcome.json, summary.txt, report.html, trajectory.csv,
    model.xml and the pictures."""
    from . import general
    from .analyses import ANALYSES
    from .install import require
    from .mjcf import to_mjcf
    from .templates import is_template, template

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    mj = require()
    params = dict(params or {})
    if is_template(target):
        b = template(target, params)
        mech, part = b.mech, b.kind
        say("solve", f"MuJoCo {mj.__version__}, {b.kind}")
        a = ANALYSES[b.kind](b, out / "images", say, **({"drive": True} if drive else {}))
    else:
        mech = load_target(target, params)
        part = str(Path(target).resolve())
        say("solve", f"MuJoCo {mj.__version__}, {Path(target).name}")
        names = [x.strip() for x in str(params.get("analyses", "")).split(",") if x.strip()] or None
        a = general.answer(mech, question or mech.question, {k: v for k, v in params.items() if k != "analyses"},
                           out / "images", say, names)
    say("outputs", "trajectory table, pictures and report")
    values = {**a.values, "_figures": a.figures}
    if a.run is not None:
        write_trajectory(out / "trajectory.csv", a.run, a.events, {j.name for j in mech.joints if j.kind == "slide"})
        values["trajectory"] = "trajectory.csv"
    if not mech.model_file:
        (out / "model.xml").write_text(to_mjcf(mech), encoding="utf-8")
        values["model_file"] = "model.xml"
    level, reasons = confidence(a.checks, a.extra)
    o = MechanismOutcome("mechanism", part, question or mech.question, a.verdict, a.headline, a.metrics, level,
                         reasons, a.checks, a.assumptions + LIMITS, values, a.images, a.fixes,
                         solver=f"MuJoCo {mj.__version__}")
    o.events = a.events
    o.model = a.model
    if a.run is not None and a.run.poses:
        o.values["playback"] = "playback.json"
        write_playback(out / "playback.json", a.run, mech, o)
    o.runtime_s = o.t_total = round(time.time() - t0, 3)
    return o.write(out)
