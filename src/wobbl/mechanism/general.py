"""Any mechanism, any question: pick the physics the question asks for, set the model up for it, and combine what
each analysis finds into one answer.

The physics are torque (whether the motors hold and move it, against their torque speed limits), kinematics (how
far it reaches and through what range), collision (whether its parts run into each other or the floor),
stability (whether it tips), contact (what each foot or wheel carries) and dynamics (the model run as written, or
along the move a spec gives). The same code serves a URDF, an MJCF file, a spec with its own bodies or a CAD
assembly turned into one, with nothing tuned to a particular robot.
"""
from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from . import quantities as Q
from .findings import Finding, Setup, motor_from_limits, worst_verdict
from .model import describe, end_effector, end_of, loaded, start_qpos
from .motors import get_motor
from .spec import Drive, Mechanism, SpecError
from .study import Answer, energy_check, event, model_info, step_check, typical_size, contact_check, warnings_check

# which words in a question ask for which physics; domain.py registers the same patterns with wobbl simulate
KEYWORDS = {
    "torque": (r"\btorques?\b", r"\bmotors?\b", r"\bservos?\b", r"\bactuators?\b", r"\bhold\b", r"\blift",
               r"strong enough", r"\bstall", r"\beffort\b", r"\bpayload\b", r"force (?:holds|to hold)",
               r"\bdrives?\b", r"\bcarr(?:y|ies) (?:a |an )?\d"),
    "kinematics": (r"\breach", r"\brange\b", r"\bworkspace\b", r"how far", r"\btravel", r"\bstroke\b",
                   r"\bextend", r"joint limits?", r"\bslide\b(?! to)", r"dead points?", r"transmission angle"),
    "collision": (r"\bcollid\w*", r"\bcollision", r"\binterfer\w*", r"\bhits?\b", r"\bhitting\b", r"\bclash",
                  r"\bclearance\b", r"run into", r"self.?collision"),
    "stability": (r"\btip(?:s|ped|ping)? over", r"\btipping\b",
                  r"\b(?:will|would|does|could|might|can) (?:it|this|that|the \w+) tip\b",
                  r"\btopple", r"fall(?:s|ing)? over", r"\bstable\b", r"\bstability\b",
                  r"\bbalance", r"\bstand(?:s|ing)? up\b"),
    "contact": (r"\bfeet\b", r"\bfoot\b", r"\bwheels?\b", r"\bcontact force", r"\breaction", r"\bground force",
                r"\bstands?\b", r"\bstanding\b", r"what (?:does|do) (?:each|the) .* carry", r"\bweight on\b"),
    "dynamics": (r"\bswing", r"\bsimulate\b", r"what happens", r"\bhow fast\b", r"\bperiod\b",
                 r"\bfall(?:s)?\b(?! over)", r"\bmotion\b"),
}
PASSIVE = r"nothing (?:holds|drives|powers|supports)|unpowered|\bpassive\b|\blimp\b|\bno motors?\b|motors? off|" \
          r"switched off"
ORDER = ("torque", "contact", "stability", "kinematics", "collision", "dynamics")


def asked(question: str) -> list[str]:
    q = (question or "").lower()
    return [n for n in ORDER if any(re.search(rx, q) for rx in KEYWORDS[n])]


def choose(question: str, info, has_move: bool, passive: bool) -> list[str]:
    """The analyses a question asks for, or the usual ones for this kind of model when it names none."""
    names = asked(question)
    if has_move and not names:
        names = ["dynamics"]
    if not names:
        names = ["contact", "stability", "torque"] if info.floating else ["torque", "kinematics", "collision"]
        if not info.joints:
            names = ["dynamics"]
    if passive:
        names = [n for n in names if n != "torque"]
        if not info.floating and "stability" in names:     # a fixed base held still answers by running it
            names = [n if n != "stability" else "dynamics" for n in names]
    if not info.joints:
        names = [n for n in names if n not in ("torque", "kinematics", "collision")] or ["dynamics"]
    return list(dict.fromkeys(names))


# --- setting the model up for the question ---------------------------------------------------------------------
def _servo_in(question: str) -> str | None:
    from .tools import _servo_names

    names = _servo_names((question or "").lower())
    return names[0] if names else None


def _payload_in(question: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|grams?|lbs?|oz)\b", (question or "").lower())
    if not m:
        return None
    unit = {"grams": "g", "gram": "g", "lbs": "lb"}.get(m.group(2), m.group(2))
    return Q.mass(f"{m.group(1)} {unit}")


def setup(mech: Mechanism, question: str, params: dict, img: Path, say=None) -> Setup:
    """The model loaded, its drives named and any payload attached, ready for the analyses."""
    params = dict(params or {})
    ld = loaded(mech)
    model = ld.compile()
    info = describe(model)
    src = ld.path.name
    motors = {j: motor_from_limits(j, e, s, src) for j, (e, s) in ld.limits.items()}
    for d in mech.drives:
        if d.motor is not None:
            motors[d.joint] = d.at_joint
    servo = params.get("servo") or params.get("motor") or _servo_in(question)
    notes = list(ld.notes)
    if servo:
        mot = get_motor(str(servo))
        targets = [j["name"] for j in info.joints if j["name"] in motors] or [j["name"] for j in info.joints]
        motors = {**motors, **{n: mot for n in targets}}
        notes.append(f"Every driven joint has a {mot.describe()}.")
    passive = bool(params.get("passive")) or bool(re.search(PASSIVE, (question or "").lower()))
    edits = dict(mech.edits)
    if str(params.get("base", "")).lower() == "free":
        edits["free_base"] = True
    kg = Q.mass(params["payload"]) if params.get("payload") else None
    if kg is None and not info.floating and set(asked(question)) & {"torque", "stability", "kinematics"}:
        kg = _payload_in(question)
    payload = None
    if kg:
        if params.get("payload_at"):
            body = str(params["payload_at"])
            if body not in info.names["body"]:
                raise SpecError(f"no body named '{body}' to carry the payload")
            b0, at0 = end_effector(model, info)
            at = at0 if info.names["body"][b0] == body else end_of(model, info.names["body"].index(body))
        else:
            b, at = end_effector(model, info)
            body = info.names["body"][b]
        payload = (kg, body, at)
        edits["payload"] = payload
        notes.append(f"The {fmt(kg * 1000)} g payload is a small solid ball held at the {body}.")
    mech = dataclasses.replace(mech, edits=edits)
    if payload or edits.get("free_base"):
        ld = loaded(mech)
        model = ld.compile()
        info = describe(model)
    ld.notes = notes + [n for n in ld.notes if n not in notes]
    su = Setup(mech, ld, model, info, motors, question, payload, passive, bool(params.get("quick")), Path(img), say)
    if payload:
        su.cache["rebuild"] = lambda m_kg: loaded(dataclasses.replace(
            mech, edits={**edits, "payload": (m_kg, payload[1], payload[2])}))
    return su


# --- one analysis -------------------------------------------------------------------------------------------
def finding(su: Setup, name: str) -> Finding:
    from . import holding, stance, workspace

    if name == "torque":
        if su.mech.drives and (su.mech.move or su.mech.spin):     # a spec's own move is the duty to judge
            return dynamics(su)
        return stance.stance_torque(su) if su.info.floating else holding.holding(su)
    if name == "kinematics":
        return workspace.reach(su)
    if name == "collision":
        return workspace.interference(su)
    if name == "stability":
        return stance.stability(su)
    if name == "contact":
        return stance.contact(su)
    if name == "dynamics":
        return dynamics(su)
    raise SpecError(f"no mechanism analysis called '{name}' (use {', '.join(ORDER)})")


def balance(su: Setup) -> dict | None:
    """Whether the starting pose is a balance point and of which kind. Near a pose where gravity is balanced,
    M q'' = -K q with K the stiffness of gravity and the joint springs, so the eigenvalues of K against M say
    whether a nudge swings back (positive, with that frequency) or grows (negative, at that rate)."""
    from scipy.linalg import eigh

    from .holding import needed, quiet
    from .install import require
    from .poses import _eq_rows, set_pose, space

    mujoco = require()
    model = quiet(su.ld.compile())
    sp = space(model, su.info)
    if not sp.n:
        return None
    d = mujoco.MjData(model)
    x0 = np.array([sp.q0[j["qadr"]] for j in sp.free])
    set_pose(sp, d, x0)
    f0 = needed(sp, d)
    g = float(np.linalg.norm(model.opt.gravity))
    scale = max(su.info.mass * g * float(model.stat.extent), 1e-9)
    if float(np.max(np.abs(f0))) > 1e-4 * scale:
        k = int(np.argmax(np.abs(f0)))
        return {"equilibrium": False, "joint": sp.free[k]["name"]}
    h = 1e-5
    K = np.zeros((sp.n, sp.n))
    for k in range(sp.n):
        for sgn in (1, -1):
            x = x0.copy()
            x[k] += sgn * h
            set_pose(sp, d, x)
            K[:, k] += sgn * needed(sp, d) / (2 * h)
    set_pose(sp, d, x0)
    mujoco.mj_forward(model, d)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, d, M)
    s = [j["dof"] for j in sp.free]
    if sp.dep:
        J, _ = _eq_rows(model, d)
        dd = [j["dof"] for j in sp.dep]
        T = np.vstack([np.eye(sp.n), -np.linalg.pinv(J[:, dd]) @ J[:, s]])
        cols = s + dd
        Meff = T.T @ M[np.ix_(cols, cols)] @ T
    else:
        Meff = M[np.ix_(s, s)]
    lam, vec = eigh(0.5 * (K + K.T), Meff)
    i = int(np.argmin(lam))
    v = vec[:, i] / max(np.max(np.abs(vec[:, i])), 1e-12)
    amp = np.array([math.radians(0.5) if j["kind"] == "hinge" else 1e-3 for j in sp.free])
    nudge = {j["name"]: float(a * x) for j, a, x in zip(sp.free, amp, v) if abs(x) > 1e-6}
    tol = 1e-6 * max(abs(lam).max(), 1e-12)
    out = {"equilibrium": True, "lambda": lam.tolist(), "nudge": nudge, "joints": [j["name"] for j in sp.free]}
    if lam[i] < -tol:
        out.update(unstable=True, rate=math.sqrt(-lam[i]))
    else:
        pos = lam[lam > tol]
        out.update(unstable=False, period=2 * math.pi / math.sqrt(pos.min()) if len(pos) else math.inf)
    return out


def dynamics(su: Setup) -> Finding:
    """The model run as written, or along the move its spec gives: what each drive needs against its motor,
    parts that touch, anything that tips, and the energy account. Left to itself, a model is first checked for
    whether its starting pose balances, and an unstable balance is nudged so the run shows it fall."""
    from .sim import simulate

    m = su.mech
    bal = None
    if m.drives and (m.move or m.spin):
        m = dataclasses.replace(m, drives=[d if d.motor is not None or d.joint not in su.motors else
                                           dataclasses.replace(d, motor=su.motors[d.joint]) for d in m.drives])
    else:
        if not m.end_time:
            m = dataclasses.replace(m, duration=2.0)
        if su.info.joints and not su.info.floating and not m.drives:
            bal = balance(su)
            if bal and bal.get("unstable"):
                t_fall = math.acosh(30) / bal["rate"]
                m = dataclasses.replace(m, edits={**m.edits, "nudge": bal["nudge"]},
                                        duration=min(max(1.5 * t_fall, m.duration or 0.0, 1.0), 10.0))
    driven = [d for d in m.drives if d.motor is not None]
    su.log("solve", "running the mechanism")
    run = simulate(m, frames=9)
    ideal = simulate(m, ideal=True, frames=0, poses=0) if driven else None
    su.log("verify", "again at half the time step")
    run2 = simulate(m, timestep=run.dt / 2, frames=0, poses=0)
    checks, events, fixes, usage = [], [], [], {}
    passive = not m.drives and not m.loads
    checks.append(energy_check(run, passive=passive and not su.info.floor))
    for d in driven:
        c = ideal.col(d.joint)
        mot = d.at_joint
        u = mot.usage(ideal.tau_need[:, c], ideal.v[:, c])
        k = int(np.argmax(u))
        rms = float(np.sqrt(np.mean(ideal.tau_need[:, c] ** 2)))
        usage[d.joint] = {"peak_usage": float(u[k]), "at_s": float(ideal.t[k]),
                          "torque_Nm": float(ideal.tau_need[k, c]),
                          "speed_rpm": float(ideal.v[k, c] * 60 / (2 * math.pi)), "rms_Nm": rms, "rated_Nm": mot.rated,
                          "motor": mot.name}
        if u[k] > 1:
            events.append(event(f"{d.joint} stalls", ideal.t[k], f"needs {Q.fmt_torque(abs(ideal.tau_need[k, c]))}"))
    hits = {k: v for k, v in run.contacts.items() if not v["support"] and "world" not in v["bodies"]}
    for k, v in hits.items():
        events.append(event(f"{k} touch", v["first_s"], f"peak {fmt(v['peak_N'])} N"))
    tipped = {k: float(np.max(v["tilt"])) for k, v in run.free.items() if np.max(v["tilt"]) > 10}
    for k in tipped:
        i = int(np.argmax(run.free[k]["tilt"] > 10))
        events.append(event(f"{run.free[k]['body']} tips", run.t[i]))
    moved = {}
    if driven:
        j = max(usage, key=lambda x: usage[x]["peak_usage"])
        c1, c2 = run.col(j), run2.col(j)
        pinned = np.any(np.abs(run.tau_need[:, c1]) > np.abs(run.tau[:, c1]) + 1e-6)
        if pinned and np.isfinite(run.q_des[:, c1]).any():
            lag = [float(np.nanmax(np.abs(np.degrees(r.q[:, c] - r.q_des[:, c])))) for r, c in ((run, c1), (run2, c2))]
            checks.append(step_check(lag[0], lag[1], f"how far {j} falls behind", " deg"))
        else:
            checks.append(step_check(float(np.max(np.abs(run.tau[:, c1]))),
                                     float(np.max(np.abs(run2.tau[:, c2]))), f"peak {j} torque", " N m"))
    elif run.joints:
        ex = np.abs(run.q - run.q[0])
        ex2 = np.abs(run2.q - run2.q[0])
        for c, jn in enumerate(run.joints):
            moved[jn] = float(np.max(ex[:, c]))
        checks.append(step_check(float(np.max(ex)) + 1.0, float(np.max(ex2)) + 1.0, "largest joint excursion plus 1",
                                 ""))
    elif run.free:
        k = next(iter(run.free))
        checks.append(step_check(float(run.free[k]["z"][-1]) + 1.0, float(run2.free[k]["z"][-1]) + 1.0,
                                 "final height plus 1 m", " m"))
    size = typical_size(run)
    checks += [c for c in (contact_check(run, size), warnings_check(run)) if c]
    stalls = [j for j, u in usage.items() if u["peak_usage"] > 1]
    hot = [j for j, u in usage.items() if u["rms_Nm"] > u["rated_Nm"]]
    worst = max((u["peak_usage"] for u in usage.values()), default=0.0)
    fell = {j: v for j, v in moved.items() if v > math.radians(5)}
    if stalls or hot or hits or tipped:
        verdict, parts = "fails", []
        if stalls:
            u = usage[stalls[0]]
            parts.append(f"The {stalls[0]} drive stalls because it needs {Q.fmt_torque(abs(u['torque_Nm']))} at "
                         f"{fmt(abs(u['speed_rpm']))} rpm, {u['peak_usage']:.2g}x what its motor gives there.")
        if hot:
            u = usage[hot[0]]
            parts.append(f"The {hot[0]} drive runs hot, at {fmt(u['rms_Nm'])} N m RMS against about "
                         f"{fmt(u['rated_Nm'])} N m continuous.")
        if hits:
            k, v = next(iter(hits.items()))
            parts.append(f"{k.replace(' / ', ' and ').capitalize()} touch at {fmt(v['first_s'])} s with up to "
                         f"{fmt(v['peak_N'])} N.")
        if tipped:
            parts.append(f"{next(iter(run.free.values()))['body'].capitalize()} tips over.")
        sentence = " ".join(parts)
    elif worst > 0.8:
        verdict = "marginal"
        sentence = f"Every drive keeps up, the busiest at {100 * worst:.0f}% of what its motor gives."
    elif usage:
        verdict, sentence = "ok", f"Every drive keeps up, the busiest at {100 * worst:.0f}% of its motor."
    elif fell:
        j = max(fell, key=fell.get)
        verdict = "fails" if su.passive else "info"
        unit = "degrees" if any(x["name"] == j and x["kind"] == "hinge" for x in su.info.joints) else "m"
        amt = math.degrees(fell[j]) if unit == "degrees" else fell[j]
        sentence = (f"Left to itself it does not stay put: the {j} moves {fmt(amt)} {unit} from where it started "
                    f"within {fmt(run.t[-1])} s.")
    else:
        verdict = "ok"
        sentence = f"Run for {fmt(run.t[-1])} s it stays put, with no parts touching and nothing tipping."
    quiet_run = not (stalls or hot or hits or tipped or usage)
    if bal and bal.get("unstable") and quiet_run:
        hinge = {x["name"]: x["kind"] == "hinge" for x in su.info.joints}
        js = max(bal["nudge"], key=lambda n: abs(bal["nudge"][n]) / (math.radians(0.5) if hinge.get(n) else 1e-3))
        c, n0 = run.col(js), bal["nudge"][js]
        dev = np.abs(run.q[:, c] - (run.q[0, c] - n0)) / abs(n0)
        pred = math.acosh(10) / bal["rate"]
        t10 = float(run.t[int(np.argmax(dev >= 10))]) if dev.max() >= 10 else math.nan
        err = abs(t10 - pred) / pred if math.isfinite(t10) else math.inf
        checks.append(Check("linear stability", "pass" if err < 0.1 else ("warn" if err < 0.25 else "fail"),
                            f"the stiffness of gravity against the mass matrix says a nudge grows tenfold in "
                            f"{fmt(pred, 3)} s, and the nudged run takes {fmt(t10, 3)} s ({100 * err:.2g}% apart)",
                            {"predicted_s": pred, "run_s": t10}))
        verdict = "fails"
        sentence = (f"Its starting pose is a balance it cannot keep: nudged by half a degree, the {js} falls away, the "
                    f"nudge growing tenfold in {fmt(t10 if math.isfinite(t10) else pred, 3)} s, so left alone it "
                    "falls over.")
        events.append(event("nudged half a degree", 0.0))
        if math.isfinite(t10):
            events.append(event("ten times the nudge", t10))
    elif bal and bal.get("equilibrium") and quiet_run:
        verdict = "ok"
        per = bal.get("period", math.inf)
        sentence = ("Its starting pose is a stable balance: nudged, it swings back" +
                    (f" with a period of {fmt(per, 3)} s." if math.isfinite(per) else "."))
    for j in stalls + [h for h in hot if h not in stalls]:
        from .motors import stronger

        u = usage[j]
        dd = m.drive(j)
        pick = stronger(dd.at_joint, abs(u["torque_Nm"]), abs(u["speed_rpm"]) * 2 * math.pi / 60, u["rms_Nm"])
        need = 1.25 * max(abs(u["torque_Nm"]), u["rms_Nm"] * 3)
        fixes.append({"kind": "motor", "title": f"Give {j} a motor with at least {Q.fmt_torque(need)} of stall torque",
                      "summary": (f"The {pick.name} clears it. " if pick else "") +
                                 "A gear or belt reduction does the same while the joint stays slow enough.",
                      "auto": False})
    su.img.mkdir(parents=True, exist_ok=True)
    images = {"frames": plots.strobe(su.img / "frames.png", run, (("x", "z"), ("x", "y")))}
    figs = [["frames", "The mechanism at several instants"]]
    if driven:
        images["torque_speed"] = plots.torque_speed(su.img / "torque_speed.png", [
            {"name": d.joint, "motor": d.at_joint, "w": ideal.v[:, ideal.col(d.joint)],
             "tau": ideal.tau_need[:, ideal.col(d.joint)]} for d in driven])
        figs.insert(0, ["torque_speed", "What each drive needs against its motor's line"])
    a = [f"Run for {fmt(run.t[-1])} s at a {fmt(run.dt * 1000)} ms step."]
    if m.drives:
        a.append("Servos follow the move exactly while their motor allows, by computed torque.")
    metrics = [(f"{j} use", f"{100 * u['peak_usage']:.0f}%") for j, u in list(usage.items())[:3]]
    metrics.append(("contacts", str(len(hits))))
    return Finding("dynamics", verdict, sentence, metrics, checks,
                   {"drives": usage, "contacts": {k: v for k, v in run.contacts.items() if not v["support"]},
                    "part_contacts": hits, "excursion_rad": moved}, fixes, events, figs, images, a,
                   run=run)


# --- one answer ---------------------------------------------------------------------------------------------
def answer(mech: Mechanism, question: str, params: dict, img: Path, say=None, names: list[str] | None = None,
           quick: bool = False) -> Answer:
    su = setup(mech, question, params, img, say)
    names = names or choose(question, su.info, bool(mech.move or mech.spin), su.passive)
    found = [finding(su, n) for n in names]
    return combine(su, found)


def combine(su: Setup, found: list[Finding]) -> Answer:
    order = sorted(found, key=lambda f: -{"fails": 2, "marginal": 1}.get(f.verdict, 0))
    verdict = worst_verdict(found)
    head = " ".join(f.sentence for f in order)
    many = len(found) > 1
    checks = []
    for f in found:
        for c in f.checks:
            checks.append(Check(f"{f.name}, {c.name}", c.status, c.summary, c.detail) if many else c)
    metrics = [m for f in found for m in f.metrics[:3]][:8]
    values = {"analyses": [f.name for f in found]}
    for f in found:
        values[f.name] = f.values
    if not many and found:
        values.update(found[0].values)
    run = next((f.run for f in found if f.name == "dynamics" and f.run is not None), None) or \
        next((f.run for f in found if f.run is not None), None)
    if run is None and su.info.joints and not su.info.floating:
        run = tour(su, found)
    extra = [x for f in found for x in f.extra]
    if su.ld.massless:
        extra.append(("medium", f"the mass of {', '.join(su.ld.massless[:3])} came from mesh files that are missing"))
    assumptions = su.ld.notes + [a for f in found for a in f.assumptions]
    return Answer(verdict, head, metrics, checks, assumptions, values, [x for f in found for x in f.fixes],
                  sorted([e for f in found for e in f.events], key=lambda e: e["t_s"]),
                  [x for f in found for x in f.figures], {k: v for f in found for k, v in f.images.items()}, run,
                  extra, model=model_info(su.mech, run) if run is not None else {})


def tour(su: Setup, found: list[Finding]):
    """A short run that drives the mechanism from its starting pose through each analysis's worst pose and back,
    with each drive held to its motor's limits, so the result can be played back with the torques live. A joint
    whose motor falls short sags in it, as it would."""
    from .sim import simulate

    stops = [p for f in found for p in f.values.get("_poses", {}).values()]
    if not stops:
        return None
    joints = sorted({k for p in stops for k in p if k in su.motors or not su.motors})
    stops = [{k: v for k, v in p.items() if k in joints} for p in stops]
    if not joints:
        return None
    start = start_qpos(su.model)
    q0 = {j["name"]: float(start[j["qadr"]]) for j in su.info.joints if j["name"] in joints}
    move = [(0.0, q0)]
    for p in stops[:4]:
        move.append((move[-1][0] + 1.5, {**q0, **p}))
    move.append((move[-1][0] + 1.5, q0))
    drives = [Drive(j, su.motors.get(j), "servo") for j in joints]
    m = dataclasses.replace(su.mech, drives=drives, move=move, hold=0.3, spin={})
    try:
        return simulate(m, frames=9)
    except Exception:  # noqa: BLE001  a tour is only for watching; the findings stand without it
        return None
