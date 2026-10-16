"""A mechanism run as a scene: every MuJoCo geom moving with its recorded pose, a pin at every joint and a motor at
every driven one, colored by the torque it is asked for against what it can give at that speed, and red when it is
asked for more. Readouts tick the torque, speed and current of the hardest worked motors.

A ready made mechanism also writes sweep.json, the mechanism solved again over the one input its question turns on,
such as an arm's payload or the drag a wiper's blade works against. Its scene is a model to work with. Each value
of that input shows the mechanism at its hardest instant, where a motor is asked the most, with every motor colored
by its share of what it can give there, and the user moves a slider over the input. Without a sweep the scene
plays the run's own poses, recorded at evenly spaced steps. Joint positions come from the run's model.xml and those
poses (mech_joints.py). Around the mechanism stand the things
the question names. A wiper gets its windshield, with an arm and blade on the output link sweeping the glass, and
an arm on a desk gets the desk, with its pick and place spots and its payload named.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import plain, read, verdict, words
from .mech_joints import joints as joint_poses, masses, rotate

FAIL_WORDS = ("limit", "stall", "slip", "tip", "bind", "collide", "hits", "lift off", "separat", "touches")
LOAD = "torque against limit"
GLASS = "#9fc6e0"
TRACK_DEG = 2.0       # a run further than this behind its move did not make it, as the arm analysis judges


def build(out: Path, pb: dict | None = None) -> Path | None:
    sw = read(out, "sweep.json")
    if sw and sw.get("manifest", {}).get("kind") == "mechanism":
        return build_sweep(out, sw)
    pb = pb or read(out, "playback.json")
    if not pb or pb.get("manifest", {}).get("kind") != "mechanism":
        return None
    m = pb["manifest"]
    t = np.asarray(pb["t"], float)
    o = read(out, "outcome.json") or {}
    o = {**o, "question": o.get("question") or m.get("question", "")}
    s = Scene(m.get("title", "Mechanism").capitalize(), domain="mechanism", question=o["question"], length="m",
              source=m.get("source", ""))
    s.timeline(t, unit="s", label="time", seconds=float(m.get("suggested_seconds") or max(t[-1], 3)),
               between="interpolated")
    reach, centre, drawn, plane = draw_bodies(s, pb)
    w = words(o)
    J = joint_poses(out, pb)
    pins(s, pb, J, reach)
    motors = drive_motors(s, pb, J, reach, drawn)
    setting(s, out, pb, J, w, plane, reach, drawn)
    readouts(s, pb, motors)
    for e in pb.get("events", []):
        bad = any(x in e["name"].lower() for x in FAIL_WORDS)
        s.event(float(e["t_s"]), e["name"], e.get("detail", ""), kind="fail" if bad else "info")
    verdict(s, {**o, "verdict": o.get("verdict") or m.get("verdict"),
                "headline": o.get("headline") or m.get("headline")})
    if not s.cam:
        s.camera(target=centre, direction=(0.55, -1.0, 0.6), distance=reach * 2.6 + 0.2)
    return s.save(out)


def build_sweep(out: Path, sw: dict) -> Path:
    """The mechanism over the input its question turns on: every value is its own run, shown at its hardest
    instant, and nothing plays on its own."""
    o = read(out, "outcome.json") or {}
    pb = read(out, "playback.json") or {}
    inp = sw["manifest"]["input"]
    title = (pb.get("manifest", {}).get("title") or "Mechanism").capitalize()
    s = Scene(f"{title} over its {inp['label']}", domain="mechanism", question=o.get("question", ""), length="m",
              source=sw["manifest"].get("source", ""))
    v = np.asarray(sw["t"], float)
    s.timeline(v, unit=inp["unit"], label=inp["label"], between="step", kind="input", design=inp["design"],
               note=inp.get("note", ""))
    reach, centre, drawn, plane = draw_bodies(s, sw)
    w = words(o)
    J = joint_poses(out, sw)
    pins(s, sw, J, reach)
    motors = drive_motors(s, sw, J, reach, drawn)
    setting(s, out, sw, J, w, plane, reach, drawn)
    paths(s, sw, v, reach)
    sweep_readouts(s, sw, motors)
    for e in sw.get("events", []):
        s.event(float(e["t_s"]), e["name"], e.get("detail", ""), kind=e.get("kind", "info"))
    s.note("Each value of the slider is its own run, shown at the instant its motors are asked the most.")
    verdict(s, o)
    c = s.cam or {}
    target = c.get("target", centre.tolist())
    dist = c.get("dist", reach * 2.6 + 0.2)
    s.camera(target=target, direction=c.get("dir", (0.55, -1.0, 0.6)), distance=dist,
             presets=[{"name": "Front", "target": target, "dir": (0.0, -1.0, 0.25), "dist": dist},
                      {"name": "Top", "target": target, "dir": (0.0, -0.05, 1.0), "dist": dist},
                      {"name": "Side", "target": target, "dir": (1.0, 0.0, 0.25), "dist": dist},
                      {"name": "Mechanism", "target": centre.tolist(), "dir": (0.55, -1.0, 0.6),
                       "dist": reach * 2.6 + 0.2}])
    return s.save(out)


def draw_bodies(s, pb):
    """Every geom the run recorded as a primitive with its pose at each frame. Returns the mechanism's reach and
    centre, the bodies drawn by name and the floor plane, if any."""
    moving = [np.asarray(b["pos"], float) for b in pb["bodies"] if not b.get("static")]
    allp = np.concatenate([p.reshape(-1, 3) for p in moving] or [np.zeros((1, 3))])
    reach = float(np.ptp(allp, 0).max()) or 0.5
    centre = allp.mean(0)
    thin = reach * 0.018
    drawn = {}
    plane = None
    for b in pb["bodies"]:
        shape, size = b["shape"], [float(x) for x in b["size"]]
        if shape == "plane":
            plane = b
            continue
        if shape not in ("sphere", "box", "cylinder", "capsule", "ellipsoid"):
            s.note(f"The {b['name']} is a {shape} geom, which this view does not draw.")
            continue
        if shape == "sphere":
            size = size[:1]
        if shape in ("capsule", "cylinder") and size[0] < thin:
            size[0] = thin
            s.note("Thin links are drawn a little thicker than the model's so they show.")
        drawn[b["name"]] = s.body(shape, size, pos=b["pos"], quat=b["quat"], color=b.get("color", "#9aa0a6"),
                                  name=b["name"])
    return reach, centre, drawn, plane


def pins(s, pb, J, reach):
    """A steel pin through every joint that is not driven, square to its axis."""
    driven = {k for k, j in pb.get("joints", {}).items() if "needed" in j}
    for name, j in J.items():
        if name in driven or j["type"] != "hinge":
            continue
        r = reach * 0.018
        s.body("cylinder", [r, r * 1.4], pos=j["anchor"], quat=_along(j["axis"]), color="#c3c7cd",
               name=f"{name} pin")


def drive_motors(s, pb, J, reach, drawn=None) -> list[dict]:
    """A motor at every driven joint, behind it on its axis, colored by its torque against its limit. The link the
    joint moves takes the same color, so the joint that runs short shows from across the room."""
    out = []
    for name, j in pb.get("joints", {}).items():
        if "needed" not in j:
            continue
        need = np.abs(np.asarray(j["needed"]["values"], float))
        lim = np.abs(np.asarray(j.get("limit", {}).get("values", need), float))
        share = np.asarray(j["share"], float) if "share" in j else need / np.maximum(lim, 1e-12)
        d = {"name": name, "need": need, "lim": lim, "share": share, "unit": j["needed"].get("unit", "N m"),
             "joint": j, "body": None}
        g = J.get(name)
        if g is not None:
            ax = np.asarray(g["axis"], float)
            side = -ax if float(np.mean(ax[:, 2])) > -0.5 else ax           # behind the link, below it when upright
            if abs(float(np.mean(ax[:, 2]))) < 0.5:
                side = -ax
            r, h = reach * 0.04, reach * 0.05
            pos = g["anchor"] + side * (h + reach * 0.012)
            body = s.body("cylinder", [r, h], pos=pos, quat=_along(side), color="#2b2e33", name=f"{name} motor")
            body.color_by(LOAD, share * 100, colormap="heat", range=(0, 150), unit="%", digits=3, limit=100,
                          limit_label="motor limit")
            s.body("cylinder", [reach * 0.012, reach * 0.014], pos=g["anchor"], quat=_along(ax), color="#c3c7cd",
                   name=f"{name} shaft")
            d["body"] = body
            link = (drawn or {}).get(g.get("body"))
            if link is not None and link.fld is None:
                link.color_by(LOAD, share * 100, colormap="heat", range=(0, 150), unit="%", digits=3, limit=100,
                              limit_label="motor limit")
            s.label(f"{name} {{{name} torque}}", [0, 0, reach * 0.09], follow=body, kind="note")
        out.append(d)
    out.sort(key=lambda d: -float(d["share"].max()))
    return out


def readouts(s, pb, motors):
    series = pb.get("series", {})
    for k, d in enumerate(motors[:4]):
        name = d["name"]
        # the torque the motor gives, red where the joint asks for more than the motor has at that speed
        gives = np.abs(np.asarray(d["joint"].get("torque", {}).get("values", d["need"]), float))
        short = d["need"] > d["lim"] * 1.02
        s.readout(f"{name} torque", gives, unit=d["unit"], digits=3,
                  limit=np.where(short, np.minimum(gives, d["lim"]) * 0.99, np.maximum(d["lim"], gives) * 1.01))
        if k >= 2:
            continue
        sp = series.get(f"{name} speed")
        if sp:
            s.readout(f"{name} speed", np.abs(np.asarray(sp["values"], float)), unit=sp.get("unit", ""), digits=3)
        elif "speed" in d["joint"]:
            v = np.abs(np.asarray(d["joint"]["speed"]["values"], float))
            s.readout(f"{name} speed", v * 60 / (2 * np.pi) if d["unit"] == "N m" else v * 1000,
                      unit="rpm" if d["unit"] == "N m" else "mm/s", digits=3)
        cur = series.get(f"{name} current")
        if cur:
            s.readout(f"{name} current", np.abs(np.asarray(cur["values"], float)), unit="A", digits=3)
    if not motors:
        for name, j in list(pb.get("joints", {}).items())[:3]:
            v = np.asarray(j["angle"]["values"], float)
            if np.ptp(v) > 1e-3:
                s.readout(f"{name} angle", np.degrees(v) if j["angle"].get("unit") != "m" else v * 1000,
                          unit="deg" if j["angle"].get("unit") != "m" else "mm", digits=3)
    for name, ser in series.items():
        if name.endswith(("speed", "current", "electrical power")) or len(s.readouts) >= 9:
            continue
        s.readout(name, ser["values"], unit=ser.get("unit", ""), digits=3)


def windows(v) -> list[tuple]:
    """The stretch of the input each saved value stands for, halfway to its neighbours, open at either end."""
    v = np.asarray(v, float)
    mid = (v[1:] + v[:-1]) / 2
    return [(None if i == 0 else float(mid[i - 1]), None if i == len(v) - 1 else float(mid[i])) for i in range(len(v))]


def paths(s, sw, v, reach):
    """Where the payload goes through the move with the real motors at each value, green when the mechanism keeps
    to the move and red when it falls behind, over the path the move asks for in yellow."""
    P = sw.get("paths") or {}
    keys = [k for k in P if not k.endswith(" asked")]
    keys = [k for k in keys if "payload" in k] or keys[:2]
    behind = (sw.get("series", {}).get("falls behind") or {}).get("values")
    design = float(sw["manifest"]["input"].get("design", v[0]))
    win = windows(v)
    for k in keys:
        asked = P.get(f"{k} asked")
        if asked:
            a = np.asarray(asked[int(np.argmin(np.abs(v - design)))], float)
            s.lines([a], color="#ffd400", width=1.5, name=f"{k} path asked")
            top = a[int(np.argmax(a[:, 2]))]
            s.label("the move as asked", top + [0, 0, reach * 0.06], kind="note")
        for i, p in enumerate(P[k]):
            off = behind is not None and behind[i] > TRACK_DEG
            s.lines([np.asarray(p, float)], color="#ff6b5b" if off else "#3dcf7a", width=2.5,
                    name=f"{k} path").show(*win[i])


def sweep_readouts(s, sw, motors):
    """The hardest worked motor's torque first, against what it can give at that instant, then the others, then
    what the run with the real motors shows."""
    n = len(sw["t"])
    series = sw.get("series", {})
    if motors:
        share = np.array([d["share"] for d in motors])
        k = share.argmax(0)
        hardest = {motors[i]["name"] for i in k}
        if len(hardest) == 1:
            motors.sort(key=lambda d: d["name"] not in hardest)
        else:
            rows = np.arange(n)
            need = np.array([d["need"] for d in motors])[k, rows]
            lim = np.array([d["lim"] for d in motors])[k, rows]
            s.readout("hardest motor torque", need, unit="N m", digits=3, limit=lim)
    for d in motors[:4]:
        name, j = d["name"], d["joint"]
        stall = j.get("mode") == "voltage"
        # a motor run open loop gives what it is asked until it stalls, so it reads red only at its stall
        s.readout(f"{name} torque", d["need"], unit=d["unit"], digits=3, limit=d["lim"] * (0.999 if stall else 1))
        sp = series.get(f"{name} speed")
        if sp:
            s.readout(f"{name} speed", sp["values"], unit="rpm", digits=3)
    for d in motors[:1] + [x for x in motors[1:4] if x["joint"].get("mode") == "voltage"]:
        rms = d["joint"].get("rms")
        if rms:
            s.readout(f"{d['name']} RMS torque", rms["values"], unit="N m", digits=3, limit=d["joint"]["rated"])
    if "falls behind" in series:
        s.readout("falls behind the move", series["falls behind"]["values"], unit="deg", digits=3, limit=TRACK_DEG)
    hit = series.get("contact force")
    if hit and max(hit["values"]) > 0.5:
        s.readout("parts touching", hit["values"], unit="N", digits=3, limit=0.5)
    tilt = series.get("base tilt")
    if tilt and max(tilt["values"]) > 0.5:
        s.readout("base tilt", tilt["values"], unit="deg", digits=3, limit=5.0)


def setting(s, out, pb, J, w, plane, reach, drawn):
    """What the question puts around the mechanism: a windshield for a wiper, a desk for an arm, else the floor."""
    driven = {k for k, j in pb.get("joints", {}).items() if "needed" in j}
    if w & {"wiper", "wipers", "windshield", "windscreen", "glass"}:
        out_j = [(k, j) for k, j in J.items() if k not in driven and j["type"] == "hinge" and
                 np.ptp(j["anchor"], 0).max() < 2e-3 * reach]
        if out_j:
            windshield(s, pb, out_j[0][0], out_j[0][1], J, reach)
            return
    if plane is not None:
        z = float(np.asarray(plane["pos"])[2])
        if w & {"desk", "desktop", "table", "bench", "workbench"}:
            desk(s, out, pb, z, reach, drawn)
        else:
            s.ground(z, size=reach * 5, grid=_grid(reach), color="#1d1f23", center=np.asarray(plane["pos"])[:2])


def windshield(s, pb, name, j, J, reach):
    """Glass over the linkage, its normal along the output joint's axis, and a wiper arm and blade fixed on the
    output link's angle, sweeping the glass. The arm and blade are drawn, not simulated."""
    ax = np.asarray(j["axis"], float)
    n = P.unit(ax.mean(0))
    if n[2] < 0:
        n = -n
    piv = np.asarray(j["anchor"], float)[0]
    body = next((b for b in pb["bodies"] if b["name"] == J[name]["body"] or b["name"] == name), None)
    if body is None or body.get("static"):
        return
    q = np.asarray(body["quat"], float)
    d = -rotate(q, [0, 0, 1])                          # along the link, from the pivot outward
    d = d - np.outer(d @ n, n)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    arm, blade = 1.5 * reach, 1.6 * reach
    lift = reach * 0.12
    base = piv + n * lift
    side = np.cross(n, d)
    quats = np.array([P.quat(a, b, n) for a, b in zip(d, side)])
    s.body("cylinder", [reach * 0.02, lift / 2], pos=piv + n * lift / 2, quat=P.z_to(n), color="#2b2e33",
           name="wiper spindle", prop=True)
    s.body("box", [arm / 2, reach * 0.018, reach * 0.01], pos=base + n * reach * 0.03 + d * arm / 2, quat=quats,
           color="#1f2124", name="wiper arm", prop=True)
    s.body("box", [blade / 2, reach * 0.012, reach * 0.016], pos=base + n * reach * 0.012 + d * arm * 0.98,
           quat=quats, color="#111214", name="wiper blade", prop=True)
    t = np.asarray(pb["t"], float)
    tip, root = base + d * (arm * 0.98 + blade / 2), base + d * (arm * 0.98 - blade / 2)
    swing = (pb.get("swing") or {}).get(body["name"])
    if swing:       # a sweep: the arc the blade's ends cover in each value's own run
        for (lo, hi), qs in zip(windows(t), swing):
            e = -rotate(np.asarray(qs, float), [0, 0, 1])
            e = e - np.outer(e @ n, n)
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            s.lines([base + e * (arm * 0.98 + blade / 2) + n * 0.002, base + e * (arm * 0.98 - blade / 2) + n * 0.002],
                    color="#e8f4ff", width=1.5, name="swept edge").show(lo, hi)
    else:
        s.lines([tip + n * 0.002, root + n * 0.002], color="#e8f4ff", width=1.5, name="swept edge",
                drawn=[t, t])
    span = np.concatenate([tip, root, base[None]])
    lo, hi = span.min(0), span.max(0)
    c = (lo + hi) / 2
    u, v, _ = P.basis(n, (0, 1, 0) if abs(n[1]) < 0.9 else (1, 0, 0))
    hu = float(np.abs((span - c) @ u).max()) * 1.12
    hv = float(np.abs((span - c) @ v).max()) * 1.2
    s.body("box", [hu, hv, reach * 0.006], pos=base - n * reach * 0.004 + (c - base) - n * ((c - base) @ n),
           quat=P.quat(u, v, n), color=GLASS, name="windshield", prop=True, opacity=0.3)
    s.label("windshield glass", base + (c - base) - n * ((c - base) @ n) + u * hu * 0.7, kind="note")
    s.note("The windshield, wiper arm and blade are drawn on the output link's solved angle.")
    s.camera(target=c - n * reach * 0.1, direction=n * 0.7 - v * 0.8 + u * 0.2, distance=max(hu, hv) * 3.2)


def desk(s, out, pb, z, reach, drawn):
    """A desk top under the arm, discs where it picks and places, and the payload named with its mass."""
    allp = np.concatenate([np.asarray(b["pos"], float).reshape(-1, 3) for b in pb["bodies"] if not b.get("static")])
    c = allp.mean(0)
    half = max(reach * 1.6, 0.25)
    s.body("box", [half * 1.3, half, reach * 0.06], pos=[c[0], c[1], z - reach * 0.06], color=P.WOOD, name="desk",
           prop=True)
    mass = masses(out)
    t = np.asarray(pb["t"], float)
    pay = next((b for b in pb["bodies"] if "payload" in b["name"].lower() and not b.get("static")), None)
    if pay is not None:
        pp = np.asarray(pay["pos"], float)
        kg = mass.get(pay["name"])
        swept = pb.get("manifest", {}).get("input", {}).get("key") == "payload"
        what = "payload" if swept or not kg else f"{kg * 1000:.0f} g payload"
        r = float(pay["size"][0])
        if pay["name"] in drawn:
            s.label(what, [0, 0, r * 2.2], follow=drawn[pay["name"]], kind="note")
        asked = (pb.get("paths") or {}).get(f"{pay['name']} asked")
        if asked:       # a sweep: the move as asked starts at the pick and ends at the place
            spots = [("pick", np.asarray(asked[0][0])), ("placed", np.asarray(asked[0][-1]))]
        else:
            spots = []
            for e in pb.get("events", []):
                nm = e["name"].lower()
                if nm in ("pick", "placed", "place"):
                    k = int(np.argmin(np.abs(t - float(e["t_s"]))))
                    spots.append((nm, pp[k] if pp.ndim == 2 else pp))
        for nm, at in spots:
            s.body("cylinder", [r * 1.6, reach * 0.002], pos=[at[0], at[1], z + reach * 0.002],
                   color="#ffd400" if nm == "pick" else "#3dcf7a", name=f"{nm} spot", prop=True)
            s.label("pick" if nm == "pick" else "place", [at[0], at[1], z + r * 0.4], kind="note")
    s.camera(target=[c[0], c[1], z + reach * 0.35], direction=(0.6, -1.0, 0.65), distance=reach * 3.2 + 0.15)


def _along(ax) -> np.ndarray | list:
    a = np.asarray(ax, float)
    if a.ndim == 1:
        return P.z_to(a)
    return np.array([P.z_to(x) for x in a])


def _grid(reach: float) -> float:
    return float(10 ** np.floor(np.log10(max(reach, 1e-3) / 3)))


__all__ = ["build", "plain"]
