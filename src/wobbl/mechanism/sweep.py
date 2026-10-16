"""A mechanism solved again over the one input a designer would change, so its view is a model to work with rather
than a recording to watch.

The input comes from the mechanism and its question. An arm sweeps its payload (or its move time when the question
asks how fast), a linkage or slider driven by a motor sweeps the load it works against, as a wiper's blade drag on
the glass. Every value is a full MuJoCo run of the mechanism. A servo that follows a move is judged on what the move
asks of it, from a run with no motor limits, against what its torque speed line gives at that speed. A motor
running open loop is judged on the torque it draws against its stall torque. Each value keeps the mechanism at its
hardest instant, every driven joint's torque and limit there, the path its point masses take in a run with the
real motors, and a few numbers. sweep.json holds them in the shape of playback.json with the input in place of
time, so the same viewer code draws either.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POSES = 400              # recorded poses per run, so the hardest instant has a pose close to it
PATH_POINTS = 100        # points kept along each path
START_SHARE = 1.0        # a motor run open loop has started once it first reaches its usual (median) speed
GLASS = {"wiper", "wipers", "windshield", "windscreen", "glass"}


@dataclass
class Input:
    key: str                     # the template value it sets
    label: str
    unit: str
    values: list[float]
    design: float
    text: str = "{v:g} {unit}"   # how a value is written for the template
    marks: list[tuple[float, str]] = field(default_factory=list)
    note: str = ""
    least: float = 0.0           # the smallest value solved, where nothing at all would drop a part from the model

    def param(self, v: float) -> str:
        return self.text.format(v=round(max(float(v), self.least), 6), unit=self.unit)


def _words(q: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (q or "").lower()))


def _nice(hi: float, n: int) -> list[float]:
    """n values from 0 to about hi, on a round step."""
    raw = hi / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(max(raw, 1e-12)))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * 0.999)
    k = max(2, int(math.ceil(hi / step - 1e-9)) + 1)
    return [round(i * step, 9) for i in range(k)]


def choose(kind: str, params: dict, question: str = "") -> Input | None:
    """The input to sweep for a ready made mechanism, or None when it has none worth a slider."""
    from .templates import template

    b = template(kind, dict(params))
    info, w = b.info, _words(question)
    if kind == "arm":
        if w & {"fast", "faster", "quick", "quickly", "speed", "seconds"} and not w & {"payload", "lift", "carry"}:
            t = info["segment_s"]
            vals = [round(t * f, 4) for f in np.linspace(0.4, 2.0, 17)]
            return Input("move_time", "time for each step of the move", "s", vals, t)
        g = info["payload"] * 1000
        # an empty gripper is solved with a tenth of a gram, so the payload stays in the model at every value
        return Input("payload", "payload", "g", _nice(max(1.6 * g, 100.0), 17), g, least=0.1)
    if kind in ("fourbar", "slider") and info.get("motor") is not None:
        unit = "N m" if kind == "fourbar" else "N"
        design = float(info["load"] or 0.0)
        mot = info["motor"]
        hi = 4 * design if design > 0 else 0.8 * mot.stall_torque
        vals = _nice(hi, 13)
        text = "{v:g} N*m" if kind == "fourbar" else "{v:g} N"
        if kind == "fourbar" and w & GLASS:
            marks, note = [(design, "dry glass")] if "dry" in w else [(design, "as asked")], ""
            if "dry" in w and design > 0:
                marks.insert(0, (design / 3, "wet glass"))
                note = ("Wet glass is marked at a third of the dry drag, as a rubber blade's friction on wet glass "
                        "is roughly a third of its friction on dry glass.")
            return Input("load", "blade drag on the glass", unit, vals, design, text, marks, note)
        label = "load on the rocker" if kind == "fourbar" else "load on the slider"
        return Input("load", label, unit, vals, design, text, [(design, "as asked")] if design else [])
    return None


def _geom_pose(P: dict, geoms: list[dict], i: int) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = P["pos"][i].copy(), P["quat"][i]
    for g, info in enumerate(geoms):
        off = np.asarray(info.get("offset", (0, 0, 0)), float)
        if off.any():
            w, u = quat[g, :1], quat[g, 1:]
            t = 2 * np.cross(u, off)
            pos[g] = pos[g] + off + w * t + np.cross(u, t)
    return pos, quat


def _thin(a: np.ndarray, n: int = PATH_POINTS) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))]


def solve(kind: str, params: dict, inp: Input, v: float) -> dict:
    """One value of the input: the mechanism at its hardest instant and what each driven joint needs there."""
    from .sim import simulate
    from .templates import template

    b = template(kind, {**params, inp.key: inp.param(v)})
    m = b.mech
    drives = {d.joint: d for d in m.drives if d.at_joint is not None}
    servo = [j for j, d in drives.items() if d.mode == "servo"]
    real = simulate(m, frames=0, poses=POSES)
    asked = simulate(m, frames=0, poses=POSES, ideal=True) if servo else real
    t = asked.t
    live = np.ones(len(t), bool)
    joints = {}
    for j, d in drives.items():
        mot, c = d.at_joint, asked.col(j)
        if d.mode == "servo":
            need, w = asked.tau_need[:, c], asked.v[:, c]
            lo, hi = mot.limits(w)
            avail = np.where(need >= 0, hi, -lo)
            gives = np.minimum(np.abs(need), np.maximum(avail, 0))
        else:
            w, need = real.v[:, c], real.tau[:, c]
            avail = np.full(len(need), mot.stall_torque)
            gives = np.abs(need)
            usual = float(np.median(np.abs(w)))
            on = np.nonzero(np.abs(w) >= START_SHARE * usual)[0]
            live &= np.arange(len(w)) >= (on[0] if len(on) else 0)
        share = np.where(np.abs(need) < 1e-12, 0.0, np.abs(need) / np.maximum(avail, 1e-12))
        joints[j] = {"need": need, "avail": avail, "gives": gives, "share": share, "w": w, "motor": mot,
                     "mode": d.mode, "rms": float(np.sqrt(np.mean(need[live] ** 2))), "rated": mot.rated}
    worst = np.max([x["share"] for x in joints.values()], axis=0) if joints else np.zeros(len(t))
    k = int(np.argmax(np.where(live, worst, -1)))
    P = asked.poses
    i = int(np.argmin(np.abs(P["t"] - t[k])))
    pos, quat = _geom_pose(P, asked.model_info["geoms"], i)
    frame = {"value": float(v), "t_s": float(t[k]), "pos": pos.tolist(), "quat": quat.tolist(), "joints": {},
             "paths": {}, "numbers": {}}
    for j, x in joints.items():
        frame["joints"][j] = {"needed": float(abs(x["need"][k])), "limit": float(max(x["avail"][k], 0.0)),
                              "gives": float(x["gives"][k]), "share": float(x["share"][k]),
                              "peak_share": float(np.max(x["share"][live])), "rms": x["rms"], "rated": x["rated"],
                              "speed": float(x["w"][k]), "mode": x["mode"], "motor": x["motor"].name}
    carried = [k for k in real.points if "payload" in k] or list(real.points)
    for name in carried:
        frame["paths"][name] = np.round(_thin(np.asarray(real.points[name])), 4).tolist()
        if asked is not real:
            frame["paths"][f"{name} asked"] = np.round(_thin(np.asarray(asked.points[name])), 4).tolist()
    nums = frame["numbers"]
    if servo:
        lag = [float(np.nanmax(np.abs(np.degrees(real.q[:, real.col(j)] - real.q_des[:, real.col(j)]))))
               for j in servo]
        nums["falls behind"] = max(lag)
    for j, x in joints.items():
        if x["mode"] != "servo":
            c = real.col(j)
            wl = np.abs(real.v[live, c]) * 60 / (2 * math.pi)
            nums[f"{j} speed"] = float(np.mean(wl)) if len(wl) else 0.0
            nums[f"{j} slowest"] = float(np.min(wl)) if len(wl) else 0.0
    hits = [c for c in real.contacts.values() if not c["support"]]
    nums["contact force"] = max([c["peak_N"] for c in hits] + [0.0])
    tilt = real.free["base_free"]["tilt"] if "base_free" in real.free else np.zeros(1)
    nums["base tilt"] = float(np.max(tilt))
    # how far each link pinned to the ground swings in the run with the real motors, as a wiper's blade sweeps
    grounded = {bd.name for bd in m.bodies if bd.parent in ("world", "") and
                any(j.body == bd.name and j.kind == "hinge" for j in m.joints)}
    Pr = real.poses
    t0 = float(t[live][0]) if live.any() else 0.0
    idx = _thin(np.nonzero(Pr["t"] >= t0)[0], 48)
    frame["swing"] = {g["name"]: np.round(Pr["quat"][idx, n], 4).tolist()
                      for n, g in enumerate(real.model_info["geoms"]) if g["name"] in grounded}
    frame["geoms"] = asked.model_info["geoms"]
    return frame


def _solve_all(kind: str, params: dict, inp: Input) -> list[dict]:
    """Every value solved, side by side on the machine's cores when it has several, since each run stands alone."""
    import os

    workers = min(len(inp.values), max(1, (os.cpu_count() or 1) - 1))
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers) as pool:
                return list(pool.map(solve, *zip(*[(kind, params, inp, v) for v in inp.values])))
        except (OSError, RuntimeError, ImportError):    # no processes to spare here, so one after another
            pass
    return [solve(kind, params, inp, v) for v in inp.values]


def run(kind: str, params: dict, question: str = "", say=None) -> dict | None:
    """Every value of the chosen input solved, in the shape of playback.json with the input in place of time."""
    inp = choose(kind, params, question)
    if inp is None:
        return None
    if say:
        say("sweep", f"{len(inp.values)} runs over the {inp.label}, {inp.values[0]:g} to "
                     f"{inp.param(inp.values[-1])}")
    frames = _solve_all(kind, params, inp)
    geoms = frames[0]["geoms"]
    pos = np.array([f["pos"] for f in frames])          # values x geoms x 3
    quat = np.array([f["quat"] for f in frames])

    def r(a, k=5):
        return np.round(np.asarray(a, float), k).tolist()

    bodies = []
    for g, info in enumerate(geoms):
        fixed = bool(np.ptp(pos[:, g], 0).max() < 1e-7 and np.ptp(quat[:, g], 0).max() < 1e-7)
        bodies.append({"name": info["name"], "shape": info["type"], "size": r(info["size"]),
                       "color": "#" + "".join(f"{int(round(255 * c)):02x}" for c in info["rgba"][:3]),
                       "static": fixed, **({"bounds": True} if info.get("bounds") else {}),
                       "pos": r(pos[0, g]) if fixed else r(pos[:, g]),
                       "quat": r(quat[0, g], 4) if fixed else r(quat[:, g], 4)})
    joints = {}
    for j in frames[0]["joints"]:
        col = {key: [f["joints"][j][key] for f in frames] for key in
               ("needed", "limit", "gives", "share", "peak_share", "rms", "speed")}
        joints[j] = {"needed": {"unit": "N m", "values": r(col["needed"], 4)},
                     "limit": {"unit": "N m", "values": r(col["limit"], 4)},
                     "torque": {"unit": "N m", "values": r(col["gives"], 4)},
                     "share": r(col["share"], 4), "peak_share": r(col["peak_share"], 4),
                     "rms": {"unit": "N m", "values": r(col["rms"], 4)},
                     "rated": frames[0]["joints"][j]["rated"], "mode": frames[0]["joints"][j]["mode"],
                     "motor": frames[0]["joints"][j]["motor"]}
    series = {k: {"values": r([f["numbers"][k] for f in frames], 4)} for k in frames[0]["numbers"]}
    paths = {k: [f["paths"][k] for f in frames] for k in frames[0]["paths"]}
    swing = {k: [f["swing"][k] for f in frames] for k in frames[0]["swing"]}
    events = [{"t_s": float(v), "name": name, "detail": ""} for v, name in inp.marks]
    for j, x in joints.items():           # where each joint first runs short of its motor
        s = np.asarray(x["peak_share"])
        over = np.nonzero(s > 1)[0]
        if len(over) and over[0] > 0:
            a, c = over[0] - 1, over[0]
            v = inp.values[a] + (1 - s[a]) / (s[c] - s[a]) * (inp.values[c] - inp.values[a])
            events.append({"t_s": round(float(v), 4), "name": f"{j} motor runs short", "detail": "", "kind": "fail"})
    return {"manifest": {"format": "wobbl-sweep", "kind": "mechanism", "version": 1,
                         "input": {"key": inp.key, "label": inp.label, "unit": inp.unit, "design": inp.design,
                                   "note": inp.note},
                         "source": "Each value of the input is a full MuJoCo run. The mechanism stands at its "
                                   "hardest instant for that value, where a driven joint asks the most of its motor."},
            "t": [float(v) for v in inp.values], "bodies": bodies, "joints": joints, "series": series,
            "paths": paths, "swing": swing, "hardest_s": [f["t_s"] for f in frames], "events": events}


def write(out: str | Path, kind: str, params: dict, question: str = "", say=None) -> Path | None:
    """out/sweep.json for a ready made mechanism. Never raises, since a sweep that fails must not cost the run."""
    try:
        doc = run(kind, params, question, say)
    except Exception:  # noqa: BLE001
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise
        return None
    if doc is None:
        return None
    p = Path(out) / "sweep.json"
    p.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return p
