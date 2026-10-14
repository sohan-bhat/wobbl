"""A four bar linkage: how far the rocker swings, where the linkage binds, its transmission angle and mechanical
advantage through the turn, and whether the motor can drive it against its load.

Two runs answer it. A slow turn of the crank by an ideal drive against a steady load on the rocker gives the
geometry, which Freudenstein's equation predicts exactly, and the torque at the crank, which virtual work predicts
exactly. A run at full speed with the real motor and the real drag then gives the speeds and the torque drawn.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from . import plots
from . import theory as T
from .analyses import crossings, with_
from .quantities import fmt_torque
from .sim import simulate
from .spec import Drive, Load
from .study import Answer, constraint_check, energy_check, event, model_info, step_check, warnings_check

MU_GOOD = (40.0, 140.0)       # the usual rule for a transmission angle that drives smoothly
MU_BIND = (20.0, 160.0)       # past these the linkage all but locks under load


def _wrap(x):
    return np.mod(np.asarray(x) + math.pi, 2 * math.pi) - math.pi


def answer(b, img: Path, say) -> Answer:
    info, m = b.info, b.mech
    d, a, bl, c = info["lengths"]
    t2s, t3s, t4s = info["start"]
    full = info["full_turn"]
    ref = info["load"] or 1.0
    motor = info["motor"]
    say("solve", "turning the crank slowly against a steady load")
    w_slow = 2 * math.pi / 4
    bare = [dataclasses.replace(j, armature=0.0) for j in m.joints]      # an ideal drive has no rotor to spin up
    if full:
        kin = with_(m, joints=bare, drives=[Drive("crank", None, "servo")], spin={"crank": (w_slow, 0.4)}, move=[],
                    loads=[Load("rocker", ref, "constant")], duration=0.4 + 2 * math.pi / w_slow + 0.05)
    else:
        lo, hi = _reach(d, a, bl, c, t2s)
        kin = with_(m, joints=bare, drives=[Drive("crank", None, "servo")], spin={},
                    loads=[Load("rocker", ref, "constant")],
                    move=[(0.0, {"crank": 0.0}), (2.0, {"crank": lo - t2s}), (6.0, {"crank": hi - t2s})],
                    duration=0.0)
    kr = simulate(kin, samples=1500, frames=9)
    qc, qk, qr = (kr.q[:, kr.col(j)] for j in ("crank", "coupler", "rocker"))
    th2, th3, th4 = t2s + qc, t3s + qc + qk, t4s + qr
    turn = (qc >= 0) & (qc <= 2 * math.pi) if full else np.ones(len(qc), bool)
    t4f, t3f = T.freudenstein(d, a, bl, c, th2)
    perr = float(np.nanmax(np.abs(np.degrees(_wrap(th4 - t4f)))[turn]))
    mu = np.degrees(T.transmission_angle(th3, th4))[turn]
    th2d = np.degrees(th2[turn] - t2s) + math.degrees(t2s)
    th4u = np.degrees(np.unwrap(th4[turn]))
    th4u += 360 * round(-float(np.mean(th4u)) / 360)      # rocker angles read from -180 to 180 degrees
    r_lo, r_hi = float(th4u.min()), float(th4u.max())
    sweep = r_hi - r_lo
    toggles = (float(th2d[np.argmin(th4u)]) % 360, float(th2d[np.argmax(th4u)]) % 360)
    vc, vr = kr.v[:, kr.col("crank")], kr.v[:, kr.col("rocker")]
    steady = turn & (np.abs(vc) > 0.3 * float(np.max(np.abs(vc))))       # past the ramp, where inertia is slight
    ratio_sim = vr[steady] / vc[steady]
    ratio_hand = T.velocity_ratio(a, c, th2[steady], t3f[steady], t4f[steady])
    tau_vw = -ref * ratio_hand                       # virtual work: T_crank w2 + T_load w4 = 0
    tau_sim = kr.tau[steady, kr.col("crank")]
    verr = float(np.max(np.abs(tau_sim - tau_vw)) / max(float(np.max(np.abs(tau_vw))), 1e-9))
    ma = 1 / np.maximum(np.abs(ratio_sim), 1e-9)
    k_ma = int(np.argmin(ma))
    checks = [
        Check("hand calculation", "pass" if perr < 0.1 else ("warn" if perr < 1 else "fail"),
              f"Freudenstein's equation puts the rocker within {fmt(perr)} degrees of MuJoCo's at every crank angle "
              f"through the {'turn' if full else 'travel'}", {"max_error_deg": perr}),
        Check("virtual work", "pass" if verr < 0.02 else ("warn" if verr < 0.05 else "fail"),
              f"against a steady {fmt(ref)} N m on the rocker, the torque MuJoCo needs at the crank matches "
              f"load times rocker speed over crank speed to {100 * verr:.2g}% of its peak", {"error": verr}),
    ]
    checks += [x for x in (None if full else constraint_check(kr, min(d, a, bl, c), "mm", "linkage loop"),
                           warnings_check(kr)) if x]

    run, peak, rms, w_mean, stalled = kr, None, None, None, False
    events = []
    if full:
        say("solve", "running at full speed with the motor and the load")
        run = simulate(m, frames=9)
        run2 = simulate(m, timestep=m.timestep / 2, frames=0)
        t_on = 0.3
        live = run.t > t_on
        cw = run.v[:, run.col("crank")]
        tq = run.tau[:, run.col("crank")]
        w_mean = float(np.mean(cw[live]))
        peak, rms = float(np.max(np.abs(tq[live]))), float(np.sqrt(np.mean(tq[live] ** 2)))
        turns = float(run.q[-1, run.col("crank")] / (2 * math.pi))
        if motor is not None:
            stalled = float(np.min(np.abs(cw[live]))) < 0.05 * motor.no_load_speed or turns < 0.9 * info["turns"]
        live2 = run2.t > t_on
        checks.append(step_check(peak, float(np.max(np.abs(run2.tau[live2, run2.col("crank")]))), "peak crank torque",
                                 " N m"))
        checks.append(energy_check(run))
        checks += [x for x in (constraint_check(run, min(d, a, bl, c), "mm", "linkage loop"), warnings_check(run))
                   if x]
        events.append(event("motor on" if motor else "crank starts", 0.0))
        qcr = run.q[:, run.col("crank")]
        for n in range(1, int(turns) + 1):
            i = int(np.argmax(qcr >= n * 2 * math.pi))
            events.append(event(f"crank turn {n}", run.t[i]))
        rv = run.v[:, run.col("rocker")]
        for tt in crossings(run.t, rv)[:6]:
            i = int(np.argmin(np.abs(run.t - tt)))
            events.append(event("rocker at an end", tt, f"{fmt(math.degrees(t4s + run.q[i, run.col('rocker')]))} deg"))
        events.sort(key=lambda e: e["t_s"])
    else:
        events = [event("start", 0.0), event("crank binds", 2.0, f"{fmt(math.degrees(lo))} deg"),
                  event("crank binds", 6.0, f"{fmt(math.degrees(hi))} deg")]

    mu_lo, mu_hi = float(mu.min()), float(mu.max())
    target = info["sweep"]
    hot = motor is not None and rms is not None and rms > motor.rated
    if not full:
        verdict = "fails"
        head = (f"The crank cannot turn all the way round. With these lengths the linkage is {info['grashof']}, so the "
                f"crank binds at {fmt(math.degrees(lo))} and {fmt(math.degrees(hi))} degrees and the rocker only "
                f"swings {fmt(sweep)} degrees.")
    elif stalled:
        verdict = "fails"
        head = (f"The motor stalls. Against {fmt(info['load'])} N m on the rocker it cannot carry the crank through "
                f"the hardest part of the turn, where the linkage needs more torque than its line gives.")
    elif mu_lo < MU_BIND[0] or mu_hi > MU_BIND[1]:
        verdict = "fails"
        worst = mu_lo if mu_lo < MU_BIND[0] else mu_hi
        head = (f"The linkage nearly locks, as its transmission angle {'falls' if worst == mu_lo else 'rises'} to "
                f"{fmt(worst)} degrees, where the coupler pushes almost along the rocker and friction can jam it.")
    elif hot:
        verdict = "fails"
        head = (f"The linkage moves, but the motor runs hot, drawing {fmt(rms)} N m RMS against about "
                f"{fmt(motor.rated)} N m it can carry continuously.")
    else:
        off = target is not None and abs(sweep - math.degrees(target)) > 3
        poor = mu_lo < MU_GOOD[0] or mu_hi > MU_GOOD[1]
        verdict = "marginal" if off or poor else "ok"
        head = (f"The rocker swings {fmt(sweep)} degrees, from {fmt(r_lo)} to {fmt(r_hi)}, as the crank turns, and the "
                f"transmission angle stays between {fmt(mu_lo)} and {fmt(mu_hi)} degrees.")
        if off:
            want = math.degrees(target)
            head += f" That is {fmt(abs(sweep - want))} degrees off the {fmt(want)} wanted."
        elif poor:
            head += f" Outside {MU_GOOD[0]:g} to {MU_GOOD[1]:g} degrees it drives stiffly and wears its pins."
        if motor is not None and w_mean is not None:
            head += (f" The motor turns the crank at {fmt(w_mean * 60 / (2 * math.pi))} rpm on average and peaks at "
                     f"{fmt_torque(peak)}.")
        elif peak is not None:
            head += f" Driving it at {fmt(info['speed'] * 60 / (2 * math.pi))} rpm takes up to {fmt_torque(peak)}."
    fixes = _fixes(info, sweep, mu_lo, mu_hi, peak, stalled, hot, full, m) if verdict != "ok" else []

    metrics = [("sweep", f"{fmt(sweep)} deg"), ("transmission", f"{fmt(mu_lo)} to {fmt(mu_hi)} deg"),
               ("worst advantage", f"{fmt(float(ma[k_ma]))}x")]
    if peak is not None:
        metrics.append(("peak torque", f"{fmt(peak)} N m"))
    img.mkdir(parents=True, exist_ok=True)
    order = np.argsort(th2d % 360)
    images = {
        "angles": plots.series(img / "angles.png", (th2d % 360)[order],
                               {"rocker angle": th4u[order] - r_lo, "transmission angle": mu[order]}, "degrees",
                               {f"{MU_GOOD[0]:g} degrees": MU_GOOD[0], f"{MU_GOOD[1]:g} degrees": MU_GOOD[1]},
                               xlabel="crank angle (degrees)"),
        "frames": plots.strobe(img / "frames.png", kr, (("x", "y"),)),
    }
    figs = [["angles", "Rocker swing (from its low end) and transmission angle through the crank's turn"],
            ["frames", "The linkage at several crank angles, from above"]]
    if full:
        series = {"crank torque": run.tau[:, run.col("crank")]}
        lines = {}
        if motor is not None:
            series["crank speed (rpm / 10)"] = run.v[:, run.col("crank")] * 6 / (2 * math.pi)
            lines = {"stall torque": motor.stall_torque}
        images["torque"] = plots.series(img / "torque.png", run.t, series, "N m", lines)
        figs.insert(1, ["torque", "Torque at the crank through the run at full speed"])
    assumptions = [
        f"Links {fmt(d * 1000)} mm ground, {fmt(a * 1000)} mm crank, {fmt(bl * 1000)} mm coupler and "
        f"{fmt(c * 1000)} mm rocker, a {info['grashof']} linkage, flat and stacked so the links never touch.",
        (f"The rocker's load is a drag of {fmt(info['load'])} N m against its motion, as from a wiper blade or a "
         "door." if info["load"] else "The rocker carries no load besides the links' own inertia."),
        (f"The {motor.describe()}, drives the crank at full voltage, with its rotor felt through the gearbox as "
         f"{fmt(motor.rotor_inertia() * 1e4)} kg cm2, a 20 ms mechanical time constant." if motor else
         f"The crank turns at a steady {fmt(info['speed'] * 60 / (2 * math.pi))} rpm whatever it takes."),
        "Pins are frictionless and have no play.",
    ]
    values = {"sweep_deg": sweep, "rocker_range_deg": [r_lo, r_hi], "toggle_crank_deg": list(toggles),
              "transmission_deg": [mu_lo, mu_hi], "min_mechanical_advantage": float(ma[k_ma]),
              "grashof": info["grashof"], "full_turn": full, "freudenstein_error_deg": perr,
              "peak_torque_Nm": peak, "rms_torque_Nm": rms,
              "mean_crank_rpm": w_mean * 60 / (2 * math.pi) if w_mean is not None else None, "stalled": stalled}
    return Answer(verdict, head, metrics, checks, assumptions, values, fixes, events, figs, images, run,
                  model=model_info(m, run))


def _reach(d, a, b, c, t2s) -> tuple[float, float]:
    """Crank angles either side of the start where a linkage that cannot turn fully binds, less a degree."""
    grid = t2s + np.radians(np.arange(0, 360.5, 0.25))
    ok = np.isfinite(T.freudenstein(d, a, b, c, grid)[0])
    up = int(np.argmax(~ok)) if not ok.all() else len(grid) - 1
    back = t2s - np.radians(np.arange(0, 360.5, 0.25))
    okb = np.isfinite(T.freudenstein(d, a, b, c, back)[0])
    down = int(np.argmax(~okb)) if not okb.all() else len(back) - 1
    return float(back[max(down - 4, 0)]), float(grid[max(up - 4, 0)])


def _fixes(info, sweep, mu_lo, mu_hi, peak, stalled, hot, full, m) -> list[dict]:
    d, a, b, c = info["lengths"]
    out = []
    target = math.degrees(info["sweep"]) if info["sweep"] is not None else sweep
    if not full or mu_lo < MU_GOOD[0] or mu_hi > MU_GOOD[1] or abs(sweep - target) > 3:
        best = _better(d, a, b, c, target)
        if best:
            bb, cc, sw, lo, hi = best
            out.append({"kind": "lengths", "title": f"Make the coupler {fmt(bb * 1000)} mm and the rocker "
                                                    f"{fmt(cc * 1000)} mm",
                        "summary": f"Freudenstein's equation gives a {fmt(sw)} degree sweep with the transmission "
                                   f"angle between {fmt(lo)} and {fmt(hi)} degrees and a crank that turns fully. Run "
                                   "it again with these lengths to check the torque.", "auto": False,
                        "after": {"coupler_mm": bb * 1000, "rocker_mm": cc * 1000, "sweep_deg": sw,
                                  "transmission_deg": [lo, hi]}})
    motor = info["motor"]
    if (stalled or hot) and motor is not None:
        w = 0.7 * motor.no_load_speed
        need = with_(m, drives=[Drive("crank", None, "servo")], spin={"crank": (w, 0.3)},
                     duration=2 * 2 * math.pi / w + 0.3)
        r = simulate(need, frames=0)
        live = r.t > 0.3
        pk = float(np.max(np.abs(r.tau[live, r.col("crank")])))
        rms = float(np.sqrt(np.mean(r.tau[live, r.col("crank")] ** 2)))
        ts = max(pk / 0.3, 3 * rms) * 1.1
        out.append({"kind": "motor", "title": f"Use a motor with at least {fmt_torque(ts)} of stall torque at the "
                                              "crank",
                    "summary": f"Turning the crank at {fmt(w * 60 / (2 * math.pi))} rpm takes up to {fmt_torque(pk)} "
                               f"and {fmt(rms)} N m RMS. A motor with that stall torque runs there at 70% of its no "
                               "load speed and stays cool, and more reduction on this motor does the same and turns "
                               "slower.",
                    "auto": False, "after": {"stall_torque_Nm": ts, "peak_Nm": pk, "rms_Nm": rms}})
    return out


def _better(d, a, b, c, target_deg: float):
    """Coupler and rocker lengths, within -40% and +60% of today's, that keep a fully turning crank and the wanted
    sweep within 1.5 degrees while holding the transmission angle closest to 90 degrees."""
    th2 = np.linspace(0, 2 * np.pi, 361)
    best, score = None, -1.0
    for bb in np.linspace(0.6 * b, 1.6 * b, 41):
        for cc in np.linspace(0.6 * c, 1.6 * c, 41):
            if T.grashof(d, a, bb, cc)[0] not in ("crank rocker",):
                continue
            t4, t3 = T.freudenstein(d, a, bb, cc, th2)
            if not np.all(np.isfinite(t4)):
                continue
            sw = float(np.degrees(np.ptp(np.unwrap(t4))))
            if abs(sw - target_deg) > 1.5:
                continue
            mu = np.degrees(T.transmission_angle(t3, t4))
            s = min(float(mu.min()), 180 - float(mu.max()))
            if s > score:
                best, score = (float(bb), float(cc), sw, float(mu.min()), float(mu.max())), s
    return best
