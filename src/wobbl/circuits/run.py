"""Run a circuit check with ngspice and verify it: textbook comparison, time step convergence, power balance."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..solvers import base as solvers
from ..units import fmt
from ..verify.checks import Check, confidence
from .draw import FIGURES, write_drawings
from .templates import TEMPLATES, Template, e24_above


def _ngspice():
    a = solvers.for_domain("circuit")
    a.require()
    return a


def simulate(t: Template, work: Path, step_factor: float = 1.0) -> dict[str, np.ndarray]:
    net = t.netlist.rstrip()
    if net.lower().endswith(".end"):
        net = net[:-4].rstrip()
    if t.analysis == "tran":
        step, stop = t.tran
        s = step * step_factor
        net += f"\n.tran {s:.6e} {stop:.6e} 0 {s:.6e} uic\n"
    data, log, secs = _ngspice().run(net + "\n.end\n", work, t.vectors, f"{t.name}_{step_factor:g}")
    return data


def _interp_at(x, y, x0):
    return float(np.interp(x0, x, y))


def run_circuit(template: str, params: dict, out_dir: str | Path, question: str = "") -> GenericOutcome:
    from ..report.render import line_plot

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if template not in TEMPLATES:
        raise ValueError(f"unknown circuit template '{template}' (use {', '.join(TEMPLATES)}) or pass a netlist")
    tpl = TEMPLATES[template](params)
    work = out / "work"
    data = simulate(tpl, work)
    checks: list[Check] = []
    extra = []
    fixes = []
    imgs = {}
    p = tpl.params
    sp = _ngspice()
    solver = f"ngspice {sp.version()}"
    if template == "led":
        i = -float(data["i(v1)"][-1])
        va = float(data["v(a)"][-1])
        vs, r = p["supply"], p["resistor"]
        exact = tpl.expect["i_led"]
        err = abs(i - exact) / max(exact, 1e-12)
        checks.append(Check("hand calculation", "pass" if err < 0.01 else "warn",
                            f"diode equation solved by hand gives {fmt(exact * 1000)} mA, ngspice {fmt(i * 1000)} mA "
                            f"({100 * err:.2g}% apart); the quick estimate (Vs - Vf)/R is "
                            f"{fmt(tpl.expect['i_simple'] * 1000)} mA", {"exact_A": exact}))
        p_src, p_r, p_led = vs * i, i * i * r, va * i
        bal = abs(p_src - p_r - p_led) / max(p_src, 1e-12)
        checks.append(Check("power balance", "pass" if bal < 0.005 else "warn",
                            f"supply gives {fmt(p_src * 1000)} mW = resistor {fmt(p_r * 1000)} mW + LED "
                            f"{fmt(p_led * 1000)} mW ({100 * bal:.2g}% apart)"))
        imax, rating = p["imax"], p["rating"]
        if vs <= p["vf"] * 0.9:
            verdict, head = "fails", f"The {fmt(vs)} V supply is below the LED's {fmt(p['vf'])} V forward voltage: " \
                                     "it will barely glow."
        elif i > imax:
            verdict = "fails"
            head = (f"The LED fries: {fmt(i * 1000)} mA through it, above its {fmt(imax * 1000)} mA rating "
                    f"({i / imax:.2g}x).")
        elif i > 0.8 * imax or p_r > 0.6 * rating:
            verdict = "warm"
            head = (f"{fmt(i * 1000)} mA through the LED ({100 * i / imax:.0f}% of its rating); resistor dissipates "
                    f"{fmt(p_r * 1000)} mW of its {fmt(rating * 1000)} mW rating. Works, little margin.")
        else:
            verdict = "ok"
            head = (f"Safe: {fmt(i * 1000)} mA through the LED ({100 * i / imax:.0f}% of its {fmt(imax * 1000)} mA "
                    f"rating); the resistor runs at {fmt(p_r * 1000)} mW.")
        checks.append(Check("operating limits", "info",
                            f"LED {fmt(i * 1000)} mA of {fmt(imax * 1000)} mA; resistor {fmt(p_r * 1000)} mW of "
                            f"{fmt(rating * 1000)} mW"))
        target = 0.020 if imax <= 0.05 else 0.7 * imax
        if verdict != "ok" or i < 0.3 * imax:
            rn = e24_above((vs - p["vf"]) / target) if vs > p["vf"] else None
            if rn:
                pr = (vs - p["vf"]) ** 2 / rn
                fixes.append({"kind": "resistor", "title": f"Use a {fmt(rn)} ohm resistor",
                              "summary": f"Gives about {fmt((vs - p['vf']) / rn * 1000)} mA; it dissipates "
                                         f"{fmt(pr * 1000)} mW, so a {'0.25' if pr < 0.12 else '0.5' if pr < 0.25 else '1'}"
                                         f" W part is enough.", "auto": False})
        metrics = [("LED current", f"{fmt(i * 1000)} mA"), ("rating", f"{fmt(imax * 1000)} mA"),
                   ("resistor power", f"{fmt(p_r * 1000)} mW"), ("LED power", f"{fmt(p_led * 1000)} mW")]
        values = {"i_led_A": i, "v_led_V": va, "p_resistor_W": p_r}
        assumptions = [f"{p['led']} LED modeled with the diode equation: {fmt(p['vf'])} V at "
                       f"{'20 mA' if imax <= 0.05 else '350 mA'}, ideality {p['n']:.2g}, series resistance 1.5 ohm.",
                       f"Resistor {fmt(r)} ohm exact, rated {fmt(rating)} W.", "Room temperature, steady DC."]
    elif template == "rc":
        tt, v = data["time"], data["v(out)"]
        vt = _interp_at(tt, v, p["tau"])
        exp_v = tpl.expect["v_at_tau"]
        err = abs(vt - exp_v) / exp_v
        checks.append(Check("hand calculation", "pass" if err < 0.005 else "warn",
                            f"V(t = RC) is {fmt(vt)} V; theory says 63.2% of {fmt(p['v'])} V = {fmt(exp_v)} V "
                            f"({100 * err:.2g}% apart)"))
        d2 = simulate(tpl, work, 0.5)
        v2 = _interp_at(d2["time"], d2["v(out)"], p["tau"])
        checks.append(_step_check(vt, v2, "V(t = RC)"))
        reach = tt[np.argmax(v >= 0.99 * p["v"])] if (v >= 0.99 * p["v"]).any() else float("nan")
        verdict, head = "ok", (f"Time constant {fmt(p['tau'] * 1000)} ms: the output reaches 63% in "
                               f"{fmt(p['tau'] * 1000)} ms and 99% in {fmt(reach * 1000)} ms.")
        metrics = [("time constant", f"{fmt(p['tau'] * 1000)} ms"), ("V at RC", f"{fmt(vt)} V"),
                   ("99% after", f"{fmt(reach * 1000)} ms")]
        values = {"v_at_tau": vt, "t99_s": float(reach)}
        assumptions = ["Ideal resistor and capacitor, step input at t = 0."]
        imgs["waveform"] = line_plot(out / "images" / "waveform.png", tt * 1000, {"v(out)": v}, "ms", "V")
    elif template == "buck":
        tt, vo, il = data["time"], data["v(out)"], data["i(l1)"]
        period = 1 / p["frequency"]
        tw = tt[-1] - 20 * period
        m = tt >= tw
        rip_v = float(vo[m].max() - vo[m].min())
        rip_i = float(il[m].max() - il[m].min())
        vavg = float(np.mean(vo[m]))
        m2 = (tt >= tw - 20 * period) & (tt < tw)
        drift = abs(vavg - float(np.mean(vo[m2]))) / max(vavg, 1e-12)
        ev, ei = tpl.expect["ripple_v"], tpl.expect["ripple_i"]
        errv = abs(rip_v - ev) / ev
        checks.append(Check("hand calculation", "pass" if errv < 0.15 else "warn",
                            f"ideal waveform math (triangular inductor ripple dI = {fmt(ei)} A through C and ESR) gives "
                            f"{fmt(ev * 1000)} mV; ngspice {fmt(rip_v * 1000)} mV and {fmt(rip_i)} A "
                            f"({100 * errv:.0f}% apart; the math ignores switch and diode drops)"))
        checks.append(Check("steady state", "pass" if drift < 0.005 else "warn",
                            f"average output moved {100 * drift:.2g}% between the last two 20 cycle windows"))
        d2 = simulate(tpl, work, 0.5)
        m_2 = d2["time"] >= d2["time"][-1] - 20 * period
        rip2 = float(d2["v(out)"][m_2].max() - d2["v(out)"][m_2].min())
        checks.append(_step_check(rip_v, rip2, "output ripple"))
        if not tpl.expect["ccm"]:
            checks.append(Check("operating mode", "warn", "inductor current reaches zero each cycle "
                                                          "(discontinuous mode); ripple formulas do not apply"))
        spec = float(p.get("max_ripple", 0.01 * p["vout"]))
        spec = spec if spec > 0 else 0.01 * p["vout"]
        verdict = "ok" if rip_v <= spec else "fails"
        head = (f"Output ripple {fmt(rip_v * 1000)} mV peak to peak on {fmt(vavg)} V "
                f"({'within' if verdict == 'ok' else 'over'} the {fmt(spec * 1000)} mV limit); inductor ripple "
                f"{fmt(rip_i)} A, peak {fmt(float(il[m].max()))} A.")
        if verdict != "ok":
            cnew = rip_i / (8 * p["frequency"] * max(spec * 0.8 - rip_i * p["esr"], 1e-6))
            fixes.append({"kind": "capacitor", "title": f"Raise output capacitance to about {fmt(cnew * 1e6)} uF",
                          "summary": "Ripple falls with 1/(8 f C) until ESR dominates; use low ESR ceramics.",
                          "auto": False})
        metrics = [("ripple", f"{fmt(rip_v * 1000)} mV"), ("output", f"{fmt(vavg)} V"),
                   ("inductor ripple", f"{fmt(rip_i)} A"), ("duty", f"{100 * p['duty']:.1f}%")]
        values = {"ripple_V": rip_v, "ripple_A": rip_i, "vout_avg": vavg, "il_peak": float(il[m].max())}
        assumptions = [f"Open loop at the steady state duty cycle {100 * p['duty']:.1f}%; real controllers add "
                       "their own ripple and transients.", f"Switch on resistance {fmt(p.get('rds', 0.05))} ohm, "
                       "Schottky catch diode.", f"Capacitor ESR {fmt(p['esr'] * 1000)} mohm, no ESL."]
        zoom = tt >= tt[-1] - 3 * period
        imgs["waveform"] = line_plot(out / "images" / "waveform.png", (tt[zoom] - tt[zoom][0]) * 1e6,
                                     {"v(out)": vo[zoom]}, "us", "V")
        from ..fields import ringing
        if ringing.wanted(p):                    # board parasitics in the hot loop: simulate the switch node too
            verdict, head = ringing.attach(p, work, out, checks, values, imgs, fixes, assumptions, metrics, verdict,
                                           head)
    elif template == "mosfet":
        tt, vd, idr = data["time"], data["v(d)"], data["i(vs)"]
        f = p["frequency"]
        tw = tt[-1] - (10 / f if f > 0 else tt[-1] / 2)
        m = tt >= tw
        pc_sim = float(np.mean(vd[m] * idr[m]))
        i_rms = float(np.sqrt(np.mean(idr[m] ** 2)))
        rds25 = p["rds_on"]
        pc_hand = i_rms ** 2 * rds25
        err = abs(pc_sim - pc_hand) / max(pc_hand, 1e-12)
        checks.append(Check("hand calculation", "pass" if err < 0.02 else "warn",
                            f"conduction loss I_rms^2 R = {fmt(pc_hand)} W; ngspice average of V x I = {fmt(pc_sim)} W "
                            f"({100 * err:.2g}% apart)"))
        trf = float(p.get("switching_time", 40e-9))
        psw = 0.5 * p["vbus"] * p["current"] * trf * f * 2 if f > 0 else 0.0
        ta = float(p.get("ambient", 25))
        theta = float(p.get("theta_ja", 62))
        tjmax = float(p.get("tj_max", 150))
        tc = float(p.get("rds_tempco", 0.006))
        tj = ta
        for _ in range(50):
            ptot = pc_sim * (1 + tc * (tj - 25)) + psw
            tj_new = ta + ptot * theta
            if abs(tj_new - tj) < 1e-3:
                break
            tj = tj_new if tj_new < 1000 else 1000
        runaway = tj >= 999
        checks.append(Check("thermal loop", "fail" if runaway else "pass",
                            "on resistance rises with temperature faster than the heat can leave: thermal runaway"
                            if runaway else f"junction settles at {fmt(tj)} C after the on resistance warms up "
                                            f"({fmt(rds25 * (1 + tc * (tj - 25)) * 1000)} mohm hot)"))
        verdict = "ok" if tj <= tjmax - 25 else ("warm" if tj <= tjmax else "hot")
        head = (f"Junction reaches about {fmt(tj)} C ({fmt(ptot)} W: {fmt(ptot - psw)} W conduction, {fmt(psw)} W "
                f"switching) with {fmt(theta)} C/W to {fmt(ta)} C air; limit {fmt(tjmax)} C.")
        if verdict != "ok":
            th_need = (tjmax - 25 - ta) / max(ptot, 1e-9)
            fixes.append({"kind": "heatsink", "title": f"Get the junction to air resistance under {fmt(th_need)} C/W",
                          "summary": "Add copper area or a heatsink; for TO-220, a clip on sink of about "
                                     f"{fmt(max(th_need - 2, 1))} C/W plus pad is typical.", "auto": False})
            fixes.append({"kind": "part", "title": f"Pick a part with under {fmt(rds25 * (tjmax - 25 - ta) / max(tj - ta, 1e-9) * 1000)} mohm",
                          "summary": "Conduction loss scales with on resistance.", "auto": False})
        metrics = [("junction", f"{fmt(tj)} C"), ("limit", f"{fmt(tjmax)} C"), ("loss", f"{fmt(ptot)} W"),
                   ("theta JA", f"{fmt(theta)} C/W")]
        values = {"tj_C": tj, "p_total_W": ptot, "p_cond_W": ptot - psw, "p_sw_W": psw}
        assumptions = [f"On resistance {fmt(rds25 * 1000)} mohm at 25 C rising {tc * 100:g}% per C.",
                       f"Switching loss from {fmt(trf * 1e9)} ns edges: 0.5 V I (tr + tf) f.",
                       f"Junction to air {fmt(theta)} C/W (datasheet, no heatsink unless included)."]
    imgs.update(write_drawings(out / "images", template, p, data, values, verdict, tpl.netlist))
    try:      # every part of the template's netlist against its ratings, and playback.json, like any circuit
        from .pictures import template_parts
        values["parts"] = template_parts(tpl.netlist, tpl.analysis, tpl.tran, out, float(p.get("ambient", 25)),
                                         verdict)
    except Exception:  # noqa: BLE001  the template's own answer stands without them
        pass
    out_o = GenericOutcome("circuit", template, question, verdict, head, metrics, "high", [], checks, assumptions,
                           {**values, "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                                 for k, v in p.items()},
                            "_figures": FIGURES + [["waveform", "Simulated waveform"]]}, imgs, fixes, solver=solver,
                           t_total=time.time() - t0)
    out_o.confidence, out_o.confidence_reasons = confidence(checks, extra)
    if "ringing" in imgs:
        out_o.values["_figures"].append(["ringing", "Switch node with and without the board's parasitics"])
    (out / "netlist.cir").write_text(tpl.netlist)
    return out_o.write(out)


def _step_check(a: float, b: float, label: str) -> Check:
    d = abs(a - b) / max(abs(b), 1e-15)
    st = "pass" if d < 0.01 else ("warn" if d < 0.05 else "fail")
    return Check("time step convergence", st, f"{label} {fmt(a)} with the normal step, {fmt(b)} with half the step "
                                              f"({100 * d:.2g}% apart)")


def run_netlist(path: str, probes: list[str], out_dir: str | Path, question: str = "") -> GenericOutcome:
    """Any user netlist, as the general circuit study runs it, with the probed signals reported."""
    from .study import simulate_circuit
    return simulate_circuit(str(path), question, {}, out_dir, list(probes))
