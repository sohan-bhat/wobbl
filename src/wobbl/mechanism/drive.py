"""Whether a motor can drive a mechanism: the torque it takes, the speed it turns at, the current and power it draws,
and whether it stalls or runs hot. It reads any mechanism with a driven joint, a four bar linkage, a slider crank, a
spec that describes a linkage, or a URDF or MJCF joint behind a motor.

Two runs answer it. A slow turn (or a slow sweep through the joint's range) by an ideal drive gives the torque the
mechanism needs with its inertia out of the way, which virtual work predicts from the weights and loads alone:
T w = -(sum of F . v over every weight, load, damper and spring). The same line of working gives a hand value
at every sample for MuJoCo's torque to be checked against. A run with the real motor at full voltage, or a servo
following a swing at a chosen speed, then gives the torque, speed and current it draws, from the motor's torque speed
line and the torque constant of a brushed motor.
"""
from __future__ import annotations

import dataclasses
import math
import re

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from . import quantities as Q
from . import theory as T
from .findings import Finding
from .motors import Motor, stronger
from .quantities import fmt_torque
from .sim import DRAG_SPEED, Run, simulate
from .spec import Drive, Load, Mechanism, SpecError
from .study import constraint_check, energy_check, event, step_check, typical_size, warnings_check

RPM = 2 * math.pi / 60
RAMP_S = 0.4                 # a slow turn ramps up to speed over this long
SLOW_TURN_S = 2.5            # a slow turn lasts this long, slow enough that the links' own inertia stays out of it
SWEEP_S = 4.5                # seconds per leg of a slow sweep through a joint's range
HOT = 1.0                    # RMS torque over the motor's continuous rating that counts as running hot
BUSY = 0.8                   # share of the motor's torque at that speed that leaves little in hand


# --- virtual work ------------------------------------------------------------------------------------------
def others_power(mech: Mechanism, run: Run) -> np.ndarray | None:
    """Power every force except the drive does on the mechanism at each sample of ``run``, W: the weights, the loads,
    joint dampers, dry friction and springs. The drive's torque times its speed has to cancel it when the motion is
    slow. None when something else acts, such as contacts or a gear mesh that loses torque."""
    from .install import require
    from .mjcf import build

    if mech.couplings or any(not c["support"] for c in run.contacts.values()) or run.free:
        return None
    mujoco = require()
    model = build(mech)
    d = mujoco.MjData(model)
    adr = {}
    for name in run.joints:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr[name] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), j)
    heavy = [b for b in range(1, model.nbody) if model.body_mass[b] > 0]
    gvec = np.asarray(model.opt.gravity, float)
    jacp = np.zeros((3, model.nv))
    out = np.zeros(len(run.t))
    start = d.qpos.copy()
    bnames = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(model.nbody)}
    for k in range(len(run.t)):
        d.qpos[:] = start
        d.qvel[:] = 0
        for c, name in enumerate(run.joints):
            qa, da, _ = adr[name]
            d.qpos[qa], d.qvel[da] = run.q[k, c], run.v[k, c]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        p = 0.0
        for b in heavy:
            mujoco.mj_jacBodyCom(model, d, jacp, None, b)
            p += float(model.body_mass[b] * gvec @ (jacp @ d.qvel))
        for name, (qa, da, j) in adr.items():
            v = float(d.qvel[da])
            p -= float(model.dof_damping[da]) * v * v + float(model.dof_frictionloss[da]) * abs(v)
            if model.jnt_stiffness[j]:
                p -= float(model.jnt_stiffness[j] * (d.qpos[qa] - model.qpos_spring[qa]) * v)
        for ld in mech.loads:
            ramp = min(1.0, run.t[k] / ld.ramp) if ld.ramp > 0 else 1.0
            if ld.joint:
                v = float(d.qvel[adr[ld.joint][1]])
                p += ld.torque * ramp * v if ld.kind == "constant" else -ld.torque * ramp * math.tanh(v / DRAG_SPEED) * v
            else:
                b = bnames[ld.body]
                pt = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(ld.point)
                mujoco.mj_jac(model, d, jacp, None, pt, b)
                p += float(np.asarray(ld.force) * ramp @ (jacp @ d.qvel))
        out[k] = p
    return out


def hand_check(mech: Mechanism, run: Run, joint: str, label: str = "virtual work") -> Check | None:
    """MuJoCo's torque at the driven joint against the one virtual work gives, T = -sum(F . v) / w, over the part of
    the run where the joint is turning at a good share of its top speed."""
    c = run.col(joint)
    w = run.v[:, c]
    P = others_power(mech, run)
    if P is None:
        return None
    top = float(np.max(np.abs(w)))
    steady = np.abs(w) > 0.3 * top
    if steady.sum() < 5:
        return None
    hand = -P[steady] / w[steady]
    sim = run.tau[steady, c]
    scale = max(float(np.max(np.abs(hand))), 1e-9)
    err = float(np.max(np.abs(sim - hand)) / scale)
    unit = "N" if _is_slide(mech, joint) else "N m"
    return Check(label, "pass" if err < 0.02 else ("warn" if err < 0.05 else "fail"),
                 f"turning {joint} slowly, the weights, loads and dampers do {fmt(float(np.max(np.abs(P))))} W at most, so "
                 f"T = sum(F . v) / w asks {fmt(float(np.max(np.abs(hand))))} {unit} at peak, and MuJoCo's drive gives "
                 f"{fmt(float(np.max(np.abs(sim))))} {unit} ({100 * err:.2g}% of the peak apart)",
                 {"error": err, "hand_peak": float(np.max(np.abs(hand))), "sim_peak": float(np.max(np.abs(sim)))})


def _is_slide(mech: Mechanism, joint: str) -> bool:
    j = next((x for x in mech.joints if x.name == joint), None)
    return j is not None and j.kind == "slide"


# --- what the motor does -----------------------------------------------------------------------------------
def electrics(motor: Motor | None, run: Run, joint: str, live: np.ndarray) -> dict:
    """Speed, torque, current and power over the live part of a run with the real motor."""
    c = run.col(joint)
    tau, w = run.tau[:, c], run.v[:, c]
    out = {"peak_torque_Nm": float(np.max(np.abs(tau[live]))),
           "rms_torque_Nm": float(np.sqrt(np.mean(tau[live] ** 2))),
           "mean_rpm": float(np.mean(np.abs(w[live])) / RPM), "peak_rpm": float(np.max(np.abs(w[live])) / RPM),
           "peak_mech_power_W": float(np.max(tau[live] * w[live])),
           "mean_mech_power_W": float(np.mean(np.maximum(tau[live] * w[live], 0.0)))}
    if motor is None:
        return out
    i = motor.current(tau)
    if i is not None:
        _, _, basis = motor.currents()
        out.update(peak_current_A=float(np.max(i[live])), mean_current_A=float(np.mean(i[live])),
                   rms_current_A=float(np.sqrt(np.mean(i[live] ** 2))), current_basis=basis,
                   stall_current_A=motor.currents()[1])
        if motor.voltage > 0:
            pe = motor.voltage * i
            out.update(supply_V=motor.voltage, peak_elec_power_W=float(np.max(pe[live])),
                       mean_elec_power_W=float(np.mean(pe[live])))
    return out


def time_ratio(t: np.ndarray, v: np.ndarray) -> dict | None:
    """How long an output spends going each way over a steady turn of the input: forward time over return time
    (above 1 means the return is the quick stroke), as a quick return mechanism is rated."""
    if len(t) < 10:
        return None
    dt = np.gradient(t)
    top = float(np.max(np.abs(v)))
    moving = np.abs(v) > 0.02 * top
    fwd, back = float(np.sum(dt[moving & (v > 0)])), float(np.sum(dt[moving & (v < 0)]))
    if fwd <= 0 or back <= 0:
        return None
    return {"slow_s": max(fwd, back), "quick_s": min(fwd, back), "ratio": max(fwd, back) / min(fwd, back)}


# --- the slow run ------------------------------------------------------------------------------------------
def _held(mech: Mechanism, drives: list, move: list, hold: dict | None):
    """Add ideal servos that keep every other joint where it started, so a joint that nothing drives does not fall."""
    if not hold:
        return drives, move
    drives = drives + [Drive(j, None, "servo") for j in hold if not any(d.joint == j for d in drives)]
    move = [(t, {**hold, **q}) for t, q in (move or [(0.0, {}), (max(mech.end_time, 1.0), {})])]
    return drives, move


def slow_mech(mech: Mechanism, joint: str, rng: tuple | None, start: float, hold: dict | None = None) -> Mechanism:
    """A copy of ``mech`` whose ``joint`` is turned (or swept through ``rng``) slowly by an ideal drive, with the
    other joints held and no rotor to spin up."""
    bare = [dataclasses.replace(j, armature=0.0) for j in mech.joints]
    drives = [Drive(joint, None, "servo")]
    if rng is None:
        w = 2 * math.pi / SLOW_TURN_S
        end = RAMP_S + SLOW_TURN_S + 0.05
        drives, move = _held(mech, drives, [(0.0, {}), (end, {})] if hold else [], hold)
        return dataclasses.replace(mech, joints=bare, drives=drives, spin={joint: (w, RAMP_S)}, move=move,
                                   duration=end, hold=0.0)
    lo, hi = rng
    span = hi - lo
    lo, hi = lo + 0.02 * span, hi - 0.02 * span
    move = [(0.0, {joint: start}), (SWEEP_S, {joint: lo}), (3 * SWEEP_S, {joint: hi}), (4 * SWEEP_S, {joint: start})]
    drives, move = _held(mech, drives, move, hold)
    return dataclasses.replace(mech, joints=bare, drives=drives, spin={}, move=move, hold=0.0, duration=0.0)


def swing_mech(mech: Mechanism, joint: str, motor: Motor, rng: tuple, start: float, rpm: float | None,
               hold: dict | None = None) -> Mechanism:
    """A copy of ``mech`` whose servo takes ``joint`` out to each end of its range and back, peaking at ``rpm``
    (default 60% of the motor's no load speed)."""
    lo, hi = rng
    span = hi - lo
    lo, hi = lo + 0.02 * span, hi - 0.02 * span
    w_top = rpm * RPM if rpm else 0.6 * (motor.no_load_speed if math.isfinite(motor.no_load_speed) else 2 * math.pi)
    leg = 1.875 * (hi - lo) / max(w_top, 1e-6)      # a minimum jerk leg peaks at 1.875 times its mean speed
    move = [(0.0, {joint: start}), (leg / 2, {joint: lo}), (leg / 2 + leg, {joint: hi}), (2 * leg + leg / 2, {joint: start})]
    drives = [d for d in mech.drives if d.joint != joint] + [Drive(joint, motor, "servo")]
    drives, move = _held(mech, drives, move, hold)
    return dataclasses.replace(mech, drives=drives, move=move, spin={}, hold=0.2, duration=0.0)


def spin_mech(mech: Mechanism, joint: str, motor: Motor, turns: float = 2.0, hold: dict | None = None) -> Mechanism:
    """A copy of ``mech`` whose motor runs ``joint`` open loop at full voltage for a couple of turns."""
    drives = [d for d in mech.drives if d.joint != joint] + [Drive(joint, motor, "voltage")]
    w = motor.no_load_speed if math.isfinite(motor.no_load_speed) else 2 * math.pi
    joints = [dataclasses.replace(j, armature=motor.rotor_inertia()) if j.name == joint else j for j in mech.joints]
    end = turns * 2 * math.pi / (0.7 * w) * 1.3 + 0.3
    drives, move = _held(mech, drives, [(0.0, {}), (end, {})] if hold else [], hold)
    return dataclasses.replace(mech, drives=drives, joints=joints, spin={}, move=move, hold=0.0, duration=end)


# --- the whole analysis ------------------------------------------------------------------------------------
def pick_joint(info, question: str, mech: Mechanism, motors: dict, model=None) -> dict:
    """The joint a drive question is about: one the question names, the joint of the spec's own drive, one a model
    gives a motor, or the joint that carries the most mass beyond it."""
    low = (question or "").lower().replace("_", " ")
    for j in sorted(info.joints, key=lambda x: -len(x["name"])):
        if re.search(rf"(?<![a-z0-9]){re.escape(j['name'].lower().replace('_', ' '))}(?![a-z0-9])", low):
            return j
    for d in mech.drives:
        for j in info.joints:
            if j["name"] == d.joint:
                return j
    for j in info.joints:
        if j["name"] in motors:
            return j
    if model is not None:
        return max(info.joints, key=lambda j: (float(model.body_subtreemass[j["body"]]), -j["dof"]))
    return info.joints[0]


def _range_of(j: dict, mech: Mechanism, start: float) -> tuple | None:
    """The limits of a joint a sweep stays within, or None for a joint that turns all the way round."""
    spec = next((x for x in mech.joints if x.name == j["name"]), None)
    if spec is not None and spec.range is not None:
        return tuple(spec.range)
    if j["limited"] and j["range"][0] < j["range"][1]:
        return tuple(j["range"])
    return None


def _speed_in(question: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*rpm", (question or "").lower())
    return float(m.group(1)) if m else None


def analyse(mech: Mechanism, joint: dict, motor: Motor | None, img, say, question: str = "", start: float = 0.0,
            hold: dict | None = None) -> Finding:
    """The drive analysis for one joint of any mechanism, through a slow ideal run, a run with the real motor and a
    run again at half the time step."""
    name = joint["name"]
    slide = joint["kind"] == "slide"
    unit = "N" if slide else "N m"
    rng = _range_of(joint, mech, start)
    if rng is None and slide:
        raise SpecError(f"{name} is a slide with no range, so give it one to sweep through")
    log = say or (lambda a, b: None)
    log("solve", f"driving {name} slowly with an ideal drive")
    kin_m = slow_mech(mech, name, rng, start, hold)
    kr = simulate(kin_m, samples=1500, frames=9, poses=0)
    checks = []
    hc = hand_check(kin_m, kr, name)
    if hc:
        checks.append(hc)
    full = None
    if motor is not None:
        log("solve", "running with the motor")
        rpm = _speed_in(question)
        full_m = (spin_mech(mech, name, motor, hold=hold) if rng is None
                  else swing_mech(mech, name, motor, rng, start, rpm, hold))
        full = full_m
        run = simulate(full_m, frames=9)
        run2 = simulate(full_m, timestep=(full_m.timestep or run.dt) / 2, frames=0, poses=0)
    else:
        run, run2 = kr, None
    c = run.col(name)
    t_on = 0.3 if motor is not None and rng is None else 0.0
    live = run.t > t_on
    el = electrics(motor, run, name, live)
    stalled = False
    usage = None
    ck = kr.col(name)
    need_slow = float(np.max(np.abs(kr.tau[np.abs(kr.v[:, ck]) > 0.3 * float(np.max(np.abs(kr.v[:, ck]))), ck])))
    if motor is not None:
        servo = rng is not None
        need = run.tau_need[:, c] if servo else run.tau[:, c]
        usage = motor.usage(need, run.v[:, c])
        # run open loop the motor sits on its line by definition, so its use is the peak torque over the stall torque
        peak_use = float(np.max(usage[live])) if servo else max(el["peak_torque_Nm"], need_slow) / motor.stall_torque
        if servo:
            stalled = peak_use > 1.02
        else:
            turns = float(run.q[-1, c] / (2 * math.pi)) if not slide else 2
            stalled = float(np.min(np.abs(run.v[live, c]))) < 0.05 * motor.no_load_speed or turns < 0.9 * 2
        hot = el["rms_torque_Nm"] > HOT * motor.rated
    else:
        peak_use, hot = 0.0, False
    if run2 is not None:
        c2 = run2.col(name)
        live2 = run2.t > t_on
        if rng is not None and stalled:
            lag = [float(np.nanmax(np.abs(r.q[:, k] - r.q_des[:, k]))) for r, k in ((run, c), (run2, c2))]
            checks.append(step_check(lag[0] + 1.0, lag[1] + 1.0, "how far the joint falls behind plus 1", ""))
        else:
            checks.append(step_check(el["peak_torque_Nm"], float(np.max(np.abs(run2.tau[live2, c2]))),
                                     "peak drive torque", f" {unit}"))
    checks.append(energy_check(run))
    checks += [x for x in (constraint_check(run, max(typical_size(run), 1e-3), "mm", "linkage loop"),
                           warnings_check(run)) if x]

    def spd(rpm: float) -> str:
        return f"{fmt(rpm * RPM * 1000)} mm/s" if slide else f"{fmt(rpm)} rpm"

    # what it says
    events = [event("drive on" if motor else "drive starts", 0.0)]
    if usage is not None and stalled and rng is not None:
        k = int(np.argmax(np.where(live, usage, 0)))
        events.append(event(f"{name} stalls", run.t[k], f"needs {fmt(abs(need[k]))} {unit}"))
    verdict, parts = "ok", []
    peak, rms = el["peak_torque_Nm"], el["rms_torque_Nm"]
    if motor is None:
        parts.append(f"Driving {name} takes up to {fmt(peak)} {unit} and {fmt(el['peak_mech_power_W'])} W at the joint, "
                     f"at {spd(el['mean_rpm'])} on average.")
        verdict = "info"
    elif stalled:
        verdict = "fails"
        if rng is None:
            parts.append(f"The motor stalls. " + (
                f"{name.capitalize()} takes {fmt_torque(need_slow)} even turning slowly, more than the "
                f"{fmt_torque(motor.stall_torque)} the motor gives at rest." if need_slow > motor.stall_torque else
                f"Against this load it cannot carry {name} through the hardest part of the turn, where the speed it "
                "needs is more than the motor gives at that torque."))
        else:
            k = int(np.argmax(np.where(live, usage, 0)))
            parts.append(f"The motor stalls. {name.capitalize()} needs {fmt_torque(abs(need[k]))} at {spd(abs(run.v[k, c]) / RPM)}, "
                         f"{peak_use:.2g}x the {fmt_torque(float(motor.limit(float(run.v[k, c]))[1]))} the motor gives there.")
    elif hot:
        verdict = "fails"
        parts.append(f"The motor drives it but runs hot, drawing {fmt(rms)} {unit} RMS against about "
                     f"{fmt(motor.rated)} {unit} it can carry continuously.")
    else:
        verdict = "marginal" if peak_use > BUSY else "ok"
        parts.append(f"The motor drives {name} at {spd(el['mean_rpm'])} on average, peaking at "
                     f"{fmt_torque(peak) if not slide else fmt(peak) + ' N'}, {100 * peak_use:.0f}% of what it gives at that speed.")
    if "peak_current_A" in el:
        parts.append(f"It draws {fmt(el['peak_current_A'])} A at the peak and {fmt(el['mean_current_A'])} A on average"
                     + (f" from {fmt(el['supply_V'])} V, up to {fmt(el['peak_elec_power_W'])} W." if "supply_V" in el else "."))
    fixes = []
    if motor is not None and (stalled or hot):
        pk = max(peak, el["peak_torque_Nm"])
        pick = stronger(motor, pk, el["peak_rpm"] * RPM, rms)
        need_ts = 1.25 * max(pk, rms * 3)
        fixes.append({"kind": "motor", "title": f"Use a motor with at least {fmt_torque(need_ts)} of stall torque",
                      "summary": (f"The {pick.name} clears it. " if pick else "") +
                                 "A gear or belt reduction does the same while the joint stays slow enough.",
                      "auto": False})
    # pictures
    img.mkdir(parents=True, exist_ok=True)
    series_ = {"drive torque": run.tau[:, c]}
    if motor is not None:
        series_["speed (rpm / 10)" if not slide else "speed (mm/s / 100)"] = run.v[:, c] * (RPM ** -1 / 10 if not slide else 10)
        if "peak_current_A" in el:
            series_["current (A)"] = motor.current(run.tau[:, c])
    images = {"drive": plots.series(img / "drive.png", run.t, series_, unit,
                                    {"stall torque": motor.stall_torque} if motor is not None else None),
              "frames": plots.strobe(img / "frames.png", run if run.poses or run.frames else kr, (("x", "y"),))}
    figs = [["drive", "Drive torque, speed and current through the run"], ["frames", "The mechanism at several instants"]]
    if motor is not None and not slide:
        images["torque_speed"] = plots.torque_speed(img / "torque_speed.png", [
            {"name": name, "motor": motor, "w": run.v[:, c][live], "tau": (need if rng is not None else run.tau[:, c])[live]}])
        figs.insert(1, ["torque_speed", "What the joint needs against the motor's line"])
    metrics = [("peak torque", f"{fmt(peak)} {unit}"), ("peak speed", spd(el["peak_rpm"]))]
    if "peak_current_A" in el:
        metrics.append(("peak current", f"{fmt(el['peak_current_A'])} A"))
    if motor is not None:
        metrics.append(("motor use", f"{100 * peak_use:.0f}%"))
    values = {"joint": name, "drive_Nm": peak, "needed_slow_Nm": need_slow, **el, "stalled": stalled, "hot": hot, "peak_usage": peak_use,
              "worst_usage": peak_use, "motor": motor.name if motor else None}
    extra = []
    if hc is None:
        extra.append(("medium", "no virtual work check was possible, because contacts or a lossy gear mesh also act"))
    assume = []
    if hold:
        assume.append("Every other joint is held where it starts by an ideal servo, so the drive's torque is the one "
                      f"for {name} alone.")
    assume.append(f"{name.capitalize()} is turned slowly by an ideal drive for the virtual work check, then driven "
                  + ("by the motor at full voltage for two turns." if rng is None else
                     "by the motor's servo out to each end of its range and back."))
    if motor is not None and "current_basis" in el:
        assume.append(f"Current follows the brushed motor's straight line in torque, from {el['current_basis']}.")
    elif motor is not None:
        assume.append("The motor's current is not known, so none is reported. Give stall_current or a supply voltage.")
    return Finding("drive", verdict, " ".join(parts), metrics, checks, values, fixes, events, figs, images, assume,
                   extra, run, (kin_m, kr))


def finding(su) -> Finding:
    """The drive analysis for a model file or spec."""
    info = su.info
    if not info.joints:
        return Finding("drive", "info", "The mechanism has no joint a motor could drive.")
    j = pick_joint(info, su.question, su.mech, su.motors, su.model)
    motor = su.motors.get(j["name"])
    from .model import start_qpos

    q0 = start_qpos(su.model)
    hold = None
    if su.mech.model_file and not su.model.neq:      # a serial chain's other joints stay put, a closed loop's cannot
        hold = {o["name"]: float(q0[o["qadr"]]) for o in info.joints if o["name"] != j["name"]}
    return analyse(su.mech, j, motor, su.img, su.say, su.question, float(q0[j["qadr"]]), hold)


# --- templates --------------------------------------------------------------------------------------------
def extend(a, b, joint: str, out: str) -> None:
    """Add the drive analysis to a template's answer: virtual work from the weights and loads checked at every sample of
    the slow run, the current and power the motor draws, and the output's time ratio. ``a.kin`` holds the slow run
    the template made and the mechanism it ran."""
    motor = b.info.get("motor")
    if a.kin is None:
        return
    kin_m, kr = a.kin
    hc = hand_check(kin_m, kr, joint, "virtual work from the weights and loads")
    if hc:
        a.checks.append(hc)
    if out in kr.joints and b.info.get("full_turn", True):
        one = (kr.t >= RAMP_S) & (kr.t <= RAMP_S + SLOW_TURN_S)      # exactly one turn at a steady speed
        tr = time_ratio(kr.t[one], kr.v[one, kr.col(out)])
        if tr:
            a.values["quick_return_ratio"] = tr["ratio"]
            a.metrics.append(("time ratio", f"{fmt(tr['ratio'])} : 1"))
    run = a.run
    if motor is not None and run is not None and joint in run.joints and run is not kr:
        live = run.t > 0.3
        el = electrics(motor, run, joint, live)
        a.values.update({k: v for k, v in el.items() if k not in a.values})
        if "peak_current_A" in el:
            a.metrics.append(("peak current", f"{fmt(el['peak_current_A'])} A"))
            a.headline += (f" At that torque the motor draws {fmt(el['peak_current_A'])} A at the peak and "
                           f"{fmt(el['mean_current_A'])} A on average"
                           + (f" from {fmt(el['supply_V'])} V." if "supply_V" in el else "."))
            a.assumptions.append(f"Current follows the brushed motor's straight line in torque, from {el['current_basis']}.")
        else:
            a.assumptions.append("The motor's current is not known, so none is reported. Give stall_current or a supply "
                                 "voltage with the motor's values to see it.")
        a.values["peak_usage"] = a.values["worst_usage"] = el["peak_torque_Nm"] / motor.stall_torque


def slider_answer(b, img, say, drive: bool = True):
    """A slider crank: the slide's stroke and dead points, the torque at the crank against the closed form
    -F dx/dtheta, and the motor that drives it."""
    from .study import Answer, model_info

    info, m = b.info, b.mech
    a_, l_, e_ = info["lengths"]
    t0 = info["start"][0]
    motor, load, kind = info["motor"], info["load"], info["load_kind"]
    jn = {"name": "crank", "kind": "hinge", "body": 0, "range": (0.0, 0.0), "limited": False}
    f = analyse(m, jn, motor, img, say, "", t0)
    say("verify", "against the closed form")
    kin_m, kr = f.kin
    qc = kr.q[:, kr.col("crank")]
    th = t0 + qc
    turn = (qc >= 0) & (qc <= 2 * math.pi)
    xs = kr.q[:, kr.col("slider")]
    x_hand = T.slider_x(a_, l_, e_, th)
    xerr = float(np.nanmax(np.abs(xs + info["start"][2] - x_hand)[turn]))
    wc = kr.v[:, kr.col("crank")]
    ratio = T.slider_ratio(a_, l_, e_, th)
    steady = turn & (np.abs(wc) > 0.3 * float(np.max(np.abs(wc))))
    vs = kr.v[:, kr.col("slider")]
    # the slider's drag opposes its motion, a constant load pushes toward the crank
    force = (-load * np.tanh(vs / DRAG_SPEED)) if kind == "drag" else np.full_like(vs, load)
    tau_cf = -force * ratio                           # T w + F v = 0 with v = (dx/dtheta) w
    tau_sim = kr.tau[:, kr.col("crank")]
    scale = max(float(np.max(np.abs(tau_cf[steady]))), 1e-9)
    terr = float(np.max(np.abs(tau_sim - tau_cf)[steady]) / scale) if load else 0.0
    f.checks.insert(0, Check("closed form position", "pass" if xerr < 1e-4 else ("warn" if xerr < 1e-3 else "fail"),
                             f"r cos(t) + sqrt(l^2 - (r sin(t) - e)^2) puts the slider within {fmt(xerr * 1000)} mm of "
                             f"MuJoCo's at every crank angle", {"max_error_m": xerr}))
    if load:
        f.checks.insert(1, Check("closed form torque", "pass" if terr < 0.02 else ("warn" if terr < 0.05 else "fail"),
                                 f"against {fmt(load)} N on the slider, the torque at the crank is -F dx/dtheta, "
                                 f"peaking at {fmt(float(np.max(np.abs(tau_cf[steady]))))} N m, and MuJoCo's matches it to "
                                 f"{100 * terr:.2g}% of that peak", {"error": terr}))
    stroke = float(np.nanmax(x_hand[turn]) - np.nanmin(x_hand[turn]))
    one = (kr.t >= RAMP_S) & (kr.t <= RAMP_S + SLOW_TURN_S)
    tr = time_ratio(kr.t[one], vs[one])
    values = dict(f.values, stroke_m=stroke, closed_form_position_error_m=xerr, closed_form_torque_error=terr,
                  lengths_m=[a_, l_, e_], quick_return_ratio=tr["ratio"] if tr else None, load_N=load)
    metrics = [("stroke", f"{fmt(stroke * 1000)} mm")] + f.metrics
    if tr:
        metrics.append(("time ratio", f"{fmt(tr['ratio'])} : 1"))
    head = (f"The slider travels {fmt(stroke * 1000)} mm for each turn of the crank"
            + (f", with a {fmt(tr['ratio'])} : 1 time ratio between its strokes" if tr and tr["ratio"] > 1.02 else "")
            + f". {f.sentence}")
    assumptions = [f"A {fmt(a_ * 1000)} mm crank and a {fmt(l_ * 1000)} mm rod push a slider on a line {fmt(e_ * 1000)} mm "
                   "off the crank's pivot, flat and stacked so nothing touches.",
                   (f"The slider's load is a {kind} force of {fmt(load)} N" + (" against its motion." if kind == "drag"
                    else " toward the crank.") if load else "The slider carries no load besides its own mass."),
                   "Pins and the slide are frictionless."] + f.assumptions
    return Answer(f.verdict, head, metrics, f.checks, assumptions, values, f.fixes, f.events, f.figures, f.images,
                  f.run, f.extra, model=model_info(m, f.run), kin=f.kin)
