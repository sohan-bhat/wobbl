"""A gear train driven by a DC motor: the output speed and torque with mesh losses, how long it takes to come up to
speed, and whether the motor stalls, runs hot or sits far from its efficient range.

MuJoCo spins each shaft with the meshes as constraints and takes a share of the torque each mesh carries as loss.
At steady state the motor's line and the load have to balance through the ratio and the efficiency, which gives
the hand check.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from .quantities import fmt_torque
from .sim import simulate
from .study import Answer, constraint_check, energy_check, event, model_info, step_check, warnings_check

RPM = 60 / (2 * math.pi)


def _steady(run, n_shafts: int):
    """Mean speeds and motor torque over the last tenth of the run, and when the output first reached 95% of it."""
    last = run.t >= 0.9 * run.t[-1]
    w_in = float(np.mean(run.v[last, run.col("shaft0")]))
    w_out = float(np.mean(run.v[last, run.col(f"shaft{n_shafts}")]))
    tau_m = float(np.mean(run.tau[last, run.col("shaft0")]))
    wo = np.abs(run.v[:, run.col(f"shaft{n_shafts}")])
    t95 = float(run.t[int(np.argmax(wo >= 0.95 * abs(w_out)))]) if abs(w_out) > 1e-9 else float("nan")
    return w_in, w_out, tau_m, t95


def answer(b, img: Path, say) -> Answer:
    info, m = b.info, b.mech
    stages, ratio, eff = info["stages"], info["ratio"], info["efficiency"]
    motor, load = info["motor"], info["load"]
    n = len(stages)
    say("solve", "spinning the train up from rest")
    run = simulate(m, frames=0)
    say("verify", "again at half the time step")
    run2 = simulate(m, timestep=m.timestep / 2, frames=0)
    w_in, w_out, tau_m, t95 = _steady(run, n)
    w_in2, w_out2, _, t95b = _steady(run2, n)
    stall_out = motor.stall_torque * ratio * eff
    stalls = load >= stall_out
    checks = []
    ratio_sim = abs(w_in / w_out) if abs(w_out) > 1e-9 else float("inf")
    w_hand, t_hand = info["hand_motor_speed"], info["hand_motor_torque"]
    if not stalls:
        err_r = abs(ratio_sim - ratio) / ratio
        err_w = abs(w_in - w_hand) / max(abs(w_hand), 1e-9)
        checks.append(Check("hand calculation", "pass" if max(err_r, err_w) < 0.005 else
                            ("warn" if max(err_r, err_w) < 0.02 else "fail"),
                            f"the teeth give a {fmt(ratio)}:1 ratio and MuJoCo's shafts turn {fmt(ratio_sim, 5)}:1, "
                            f"and the motor line balances the load through {100 * eff:.3g}% efficiency at "
                            f"{fmt(w_hand * RPM)} rpm, and MuJoCo settles at {fmt(w_in * RPM)} rpm "
                            f"({100 * err_w:.2g}% apart)", {"ratio": ratio_sim, "motor_rpm": w_in * RPM,
                                                           "hand_motor_rpm": w_hand * RPM}))
        if load > 0:
            eff_sim = load * abs(w_out) / max(tau_m * abs(w_in), 1e-12)
            e_err = abs(eff_sim - eff) / eff
            checks.append(Check("mesh losses", "pass" if e_err < 0.01 else ("warn" if e_err < 0.03 else "fail"),
                                f"output power over motor power is {100 * eff_sim:.3g}%, against "
                                f"{100 * info['mesh_eff']:.3g}% per mesh over {n} mesh{'es' if n > 1 else ''}, "
                                f"{100 * eff:.3g}%", {"efficiency": eff_sim}))
        checks.append(step_check(abs(w_out), abs(w_out2), "steady output speed", " rad/s"))
    else:
        checks.append(Check("hand calculation", "pass" if abs(w_out) < 0.02 * motor.no_load_speed / ratio else "warn",
                            f"the load of {fmt(load)} N m is above the {fmt(stall_out)} N m the motor can hold through "
                            f"the train, and MuJoCo's output turns at {fmt(abs(w_out) * RPM)} rpm", {}))
    checks.append(energy_check(run))
    checks += [c for c in (constraint_check(run, 2 * math.pi, "degrees", "gear meshes"), warnings_check(run)) if c]

    t_me = motor.max_efficiency_torque()
    out_rpm = abs(w_out) * RPM
    want = info["want_speed"]
    slow = want is not None and abs(w_out) < want
    hot = tau_m > motor.rated
    far = not stalls and not (0.5 * t_me <= tau_m <= 2 * t_me)
    if stalls:
        verdict = "fails"
        head = (f"The motor stalls because the {fmt(load)} N m load is more than the {fmt(stall_out)} N m it can "
                f"hold through the {fmt(ratio)}:1 train at {100 * eff:.3g}% efficiency.")
    elif hot or slow:
        verdict = "fails"
        head = (f"The output turns at {fmt(out_rpm)} rpm with {fmt_torque(load)}"
                + (f", short of the {fmt(want * RPM)} rpm wanted" if slow else "")
                + (f", and the motor runs hot at {fmt(tau_m * 1000)} mN m against about {fmt(motor.rated * 1000)} mN m "
                   "it can carry continuously" if hot else "") + ".")
    else:
        verdict = "marginal" if far else "ok"
        head = (f"The output turns at {fmt(out_rpm)} rpm with {fmt_torque(load)} through the {fmt(ratio)}:1 train, "
                f"{100 * eff:.3g}% efficient, and reaches 95% of that speed in {fmt(t95 * 1000)} ms.")
        if far:
            head += (f" The motor works at {100 * tau_m / motor.stall_torque:.0f}% of its stall torque, well away from "
                     f"its most efficient {100 * t_me / motor.stall_torque:.0f}%.")
    fixes = []
    if verdict != "ok":
        fixes = _fixes(info, tau_m, t_me, stalls, hot, slow)
    metrics = [("output", f"{fmt(out_rpm)} rpm"), ("ratio", f"{fmt(ratio)}:1"), ("efficiency", f"{100 * eff:.3g}%"),
               ("motor load", f"{100 * tau_m / motor.stall_torque:.0f}% of stall")]
    img.mkdir(parents=True, exist_ok=True)
    images = {"spinup": plots.series(img / "spinup.png", run.t * 1000,
                                     {"output rpm": run.v[:, run.col(f"shaft{n}")] * RPM,
                                      "motor rpm / ratio": run.v[:, run.col("shaft0")] * RPM / ratio},
                                     "rpm", xlabel="ms"),
              "torque_speed": plots.torque_speed(img / "torque_speed.png", [
                  {"name": "motor", "motor": motor, "w": run.v[:, run.col("shaft0")],
                   "tau": run.tau[:, run.col("shaft0")]}])}
    events = [event("motor on", 0.0)]
    if math.isfinite(t95):
        events.append(event("95% of full speed", t95, f"{fmt(out_rpm)} rpm at the output"))
    assumptions = [
        f"Stages {', '.join(f'{a}:{bb}' for a, bb in stages)} teeth, each mesh {100 * info['mesh_eff']:.3g}% "
        "efficient, with the loss taken in proportion to the torque the mesh carries.",
        f"The {motor.describe()}, runs at full voltage from rest"
        + (f", with a rotor of {fmt(info['j_rotor'] * 1e9)} g mm2 chosen to give the bare motor a 20 ms mechanical "
           "time constant, typical of small brushed motors" if info["rotor_auto"] else "") + ".",
        f"The load is a drag of {fmt(load)} N m at the output" + (" plus the given inertia." if
                                                                   info.get("j_eq") else "."),
        "Gears are rigid with no backlash, and the gear blanks are solid discs.",
    ]
    values = {"ratio": ratio, "ratio_simulated": ratio_sim, "efficiency": eff, "output_rpm": out_rpm,
              "motor_rpm": abs(w_in) * RPM, "motor_torque_Nm": tau_m, "output_torque_Nm": load, "t95_s": t95,
              "stall_output_torque_Nm": stall_out, "max_efficiency_torque_Nm": t_me}
    return Answer(verdict, head, metrics, checks, assumptions, values, fixes, events,
                  [["spinup", "Output speed coming up from rest"],
                   ["torque_speed", "The motor's operating points against its line"]], images, run,
                  model=model_info(m, run))


def _fixes(info, tau_m, t_me, stalls, hot, slow) -> list[dict]:
    motor, load, eff = info["motor"], info["load"], info["efficiency"]
    n = len(info["stages"])
    out = []
    if load > 0:
        best = load / (eff * t_me)
        per = best ** (1 / n)
        teeth = [(12, max(12, round(12 * per))) for _ in range(n)]
        real = math.prod(bb / a for a, bb in teeth)
        w_out = motor.no_load_speed * (1 - load / (real * eff) / motor.stall_torque) / real
        out.append({"kind": "ratio", "title": f"Change the ratio to about {fmt(best)}:1, for example "
                                              f"{', '.join(f'{a}:{bb}' for a, bb in teeth)}",
                    "summary": f"The motor then works near its most efficient torque, {fmt(t_me * 1000)} mN m, and the "
                               f"output turns at about {fmt(w_out * RPM)} rpm. Run it again with the new teeth to "
                               "check.", "auto": False, "after": {"ratio": real, "output_rpm": w_out * RPM}})
    if stalls or hot or slow:
        need = 1.25 * max(load / (info["ratio"] * eff), 3 * tau_m)
        out.append({"kind": "motor", "title": f"Use a motor with at least {fmt(need * 1000)} mN m of stall torque",
                    "summary": "Through the same train it carries the load at under a third of stall, where it stays "
                               "cool.", "auto": False, "after": {"stall_torque_Nm": need}})
    return out
