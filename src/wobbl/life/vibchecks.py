"""The checks, verdict, pictures, assumptions and fixes of a vibration answer."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..structural.outcome import process_caveat
from ..units import fmt
from ..verify.checks import Check, convergence_check
from . import fatigue as F
from . import modal
from .outcome import verdict_for

ORDER = {"ok": 0, "holds": 1, "marginal": 2, "fails": 3}


def relevant_modes(modes: modal.Modes, frac: np.ndarray) -> np.ndarray:
    """Modes a real excitation can drive: those that move at least 1% of the mass along an axis, and those that
    swing an attached mass, which is where a motor or rotor pushes."""
    keep = frac.max(1) >= 0.01
    for nodes in modes.mass_nodes.values():
        mot = np.linalg.norm(modes.shapes[:, nodes].astype(float), axis=2).max(1)
        top = np.linalg.norm(modes.shapes.astype(float), axis=2).max(1)
        keep |= mot >= 0.2 * np.maximum(top, 1e-30)
    return keep


def separation(f: float, lo: float, hi: float) -> float:
    """How far a natural frequency sits from an excitation, as a share of it: zero inside a band."""
    if lo <= f <= hi:
        return 0.0
    return (lo - f) / lo if f < lo else (f - hi) / hi


def resonance(modes, frac, excite, margin: float, zeta: float) -> tuple[list[dict], str]:
    """Each relevant mode against each excitation, and the verdict: fails when a mode sits inside an excitation or
    within half the margin of it, marginal within the margin, holds when every mode is clear."""
    rows, worst = [], "holds"
    rel = relevant_modes(modes, frac)
    for e in excite:
        mid = math.sqrt(e.lo * e.hi)
        nearest = int(np.argmin([abs(math.log(x / mid)) for x in modes.freq]))
        for i, f in enumerate(modes.freq):
            sep = separation(float(f), e.lo, e.hi)
            if sep > 3 * margin and i != nearest:
                continue
            r = mid / f if sep > 0 else 1.0
            amp = 1.0 / math.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2)
            status = "resonant" if sep <= 0.5 * margin else ("close" if sep <= margin else "clear")
            if not rel[i]:
                status = "barely excited" if status != "clear" else "clear"
            rows.append({"mode": i + 1, "frequency": float(f), "excitation": e.describe(), "lo": e.lo, "hi": e.hi,
                         "separation": sep, "status": status, "amplification": amp,
                         "mass_share": float(frac[i].max())})
            v = {"resonant": "fails", "close": "marginal"}.get(status, "holds")
            if ORDER[v] > ORDER[worst]:
                worst = v
    return rows, worst


def evaluate(study, mat, modes, mesh, built, levels, stopped_by, tol, zeta, excite, sine, psd, axis, resp, harmonic,
             rayleigh, dur, p):
    checks: list[Check] = []
    extra: list[tuple[str, str]] = []
    total = float(modes.total_mass[:3].mean())
    frac = modes.effective_mass[:, :3] / max(total, 1e-300)
    target = study.target_sf or mat.recommended_sf
    watch = levels[-1]["watch_mode"]
    checks.append(convergence_check([L["watch_Hz"] for L in levels], tol, f"frequency of mode {watch}", "Hz",
                                    stopped_by, min_levels=2))
    if resp is not None:
        c = convergence_check([L["stress"] for L in levels], 2.5 * tol, "governing response stress", "MPa",
                              stopped_by, min_levels=2)
        c.name = "stress convergence"
        checks.append(c)
    checks.append(_mass_check(mesh, built, modes, total))
    checks.append(_participation(modes, frac, sine, psd, axis, excite))
    checks.append(_rayleigh_check(rayleigh))
    values = {"analysis": "vibration", "frequencies": modes.freq.tolist(),
              "mass_fraction": frac.round(4).tolist(), "total_mass_kg": total * 1000, "target_sf": target,
              "modes": modes.count, "masses_g": [float(np.sum(v)) * 1e6 for v in modes.mass_values.values()],
              "_figures": [["modes", "Mode shapes that the excitation can drive, motion exaggerated"],
                           ["frequencies", "Natural frequencies against the excitation, as tall as the share of mass "
                                           "each moves"],
                           ["response", "Sine sweep response at the governing point"],
                           ["stress", "Response stress with the governing point marked"],
                           ["random", "3 sigma stress of the random input"]]}
    verdicts, heads, sf = [], [], None
    if excite:
        rows, v = resonance(modes, frac, excite, study.margin, zeta)
        values["resonance"] = rows
        values["excitations"] = [e.describe() for e in excite]
        values["margin"] = study.margin
        verdicts.append(v)
        heads.append(_resonance_head(rows, v, excite, study.margin))
        values["_tables"] = [["Natural frequencies against the excitation",
                              ["mode", "Hz", "excitation", "separation", "amplification", "status"],
                              [[r["mode"], fmt(r["frequency"]), r["excitation"], f"{100 * r['separation']:.0f}%",
                                f"{r['amplification']:.2g}x", r["status"]] for r in rows],
                              f"Modes within {100 * study.margin:.0f}% of an excitation are close, within half that "
                              "resonant. Amplification is the steady response of that mode at the middle of the "
                              f"excitation for {100 * zeta:g}% damping."]]
    allow = mat.allowable
    if sine is not None and resp is not None:
        h = resp["harmonic"]
        ok = harmonic is not None and "error" not in harmonic
        peak = harmonic["peak"] if ok else h["peak"]
        xyz = harmonic["xyz"] if ok else h["xyz"]
        f_at = harmonic["f"] if ok else h["f_peak"]
        sf_h = allow / peak if peak > 0 else math.inf
        verdicts.append(verdict_for(sf_h, target))
        sf = sf_h if sf is None else min(sf, sf_h)
        values["harmonic"] = {"peak_MPa": peak, "frequency_Hz": f_at, "xyz": xyz, "sf": sf_h,
                              "input": sine.text, "solver": "CalculiX" if ok else "Wobbl's modal sum"}
        heads.append(f"The {sine.text} sweep peaks at {fmt(f_at)} Hz with {fmt(peak)} MPa at "
                     f"({', '.join(fmt(v) for v in xyz)}), safety factor {sf_h:.2f} against "
                     f"{mat.criterion.split(' vs ')[1]}.")
        checks.append(_sdof_check(modes, h, sine, zeta))
        checks.append(_cross_check(harmonic, h))
        checks.append(_singular(h["raw_max"], h["peak"], h["exclusion_radius"], "sine peak"))
    if psd is not None and resp is not None:
        r = resp["random"]
        s3 = 3 * r["sigma_rms"]
        sf_r = allow / s3 if s3 > 0 else math.inf
        verdicts.append(verdict_for(sf_r, target))
        sf = sf_r if sf is None else min(sf, sf_r)
        rnd = {"sigma_rms_MPa": r["sigma_rms"], "three_sigma_MPa": s3, "rate_Hz": r["rate"], "xyz": r["xyz"],
               "sf": sf_r, "psd": psd.describe(), "grms": psd.grms(), "axis": list(map(float, axis))}
        head = (f"Under {psd.name} random input ({psd.grms():.3g} grms) along {_axis_word(axis)} the 3 sigma stress is "
                f"{fmt(s3)} MPa at ({', '.join(fmt(v) for v in r['xyz'])}), safety factor {sf_r:.2f}.")
        if dur > 0:
            fat = _random_fatigue(mat, r, dur)
            rnd["fatigue"] = fat
            if fat is not None:
                verdicts.append(verdict_for(fat["sf"], target))
                if fat["damage"] <= 0:
                    head += (f" Over {fmt(dur / 60)} min even the 3 sigma peaks stay under the endurance limit, so "
                             f"Steinberg's method finds no fatigue damage; the stress could rise {fat['sf']:.2f} times "
                             "before cracks would start.")
                else:
                    head += (f" Over {fmt(dur / 60)} min Steinberg's method gives fatigue damage {fat['damage']:.2g}, "
                             f"and the stress could rise {fat['sf']:.2f} times before cracks would start.")
                if fat["estimated"]:
                    extra.append(("medium", "the fatigue estimate uses an S-N rule of thumb for this material, "
                                            "not test data"))
        values["random"] = rnd
        heads.append(head)
        checks.append(_miles_check(modes, r, psd, axis, zeta, frac))
        checks.append(_singular(r["raw_max"], r["sigma_rms"], r["exclusion_radius"], "rms stress"))
    checks.append(_sanity(modes, total))
    verdict = max(verdicts, key=lambda v: ORDER[v]) if verdicts else "ok"
    if not heads:
        d = int(np.argmax(frac[0]))
        heads.append(f"First natural frequency {fmt(modes.freq[0])} Hz, where mode 1 moves {100 * frac[0, d]:.0f}% of "
                     f"the mass along {'xyz'[d]}. No excitation was given, so Wobbl lists the modes without a "
                     "verdict; pass --rpm, --frequency, --sine or --psd to judge them.")
    values["sf"] = sf
    return checks, extra, verdict, sf, " ".join(heads), values


def _axis_word(axis) -> str:
    a = np.asarray(axis, float)
    k = int(np.argmax(np.abs(a)))
    return "xyz"[k] if abs(a[k]) > 0.999 else ", ".join(f"{v:.2g}" for v in a)


def _resonance_head(rows, v, excite, margin) -> str:
    bad = [r for r in rows if r["status"] in ("resonant", "close")]
    if bad:
        r = min(bad, key=lambda r: r["separation"])
        where = "inside" if r["separation"] == 0 else f"{100 * r['separation']:.0f}% from"
        return (f"Mode {r['mode']} at {fmt(r['frequency'])} Hz sits {where} the {r['excitation']}, where the response "
                f"grows about {r['amplification']:.2g} times.")
    near = min(rows, key=lambda r: r["separation"]) if rows else None
    s = f"Every mode that the excitation can drive stays at least {100 * margin:.0f}% clear of it"
    return s + (f"; the nearest is mode {near['mode']} at {fmt(near['frequency'])} Hz, {100 * near['separation']:.0f}% "
                f"from the {near['excitation']}." if near else ".")


def _mass_check(mesh, built, modes, total) -> Check:
    pm = modal.part_mass(mesh, built.model.density, modes.mass_values)
    err = abs(total - pm) / pm if pm > 0 else 1.0
    k = min(6, modes.count)
    mm = modal.modal_masses(mesh, built.model.density, modal.Modes(
        modes.freq[:k], modes.shapes[:k], None, modes.participation[:k], modes.effective_mass[:k], modes.total_mass,
        modes.n, modes.mass_nodes, modes.mass_values))
    norm = float(np.abs(mm - 1).max())
    s = (f"CalculiX counts {fmt(total * 1e6)} g against {fmt(pm * 1e6)} g from the volume, the density and the "
         f"attached masses ({100 * err:.2g}% apart), and Wobbl's own mass matrix gives the first {k} modes a modal "
         f"mass of one to {100 * norm:.2g}%")
    worst = max(err, norm)
    status = "pass" if worst <= 0.01 else ("warn" if worst <= 0.03 else "fail")
    return Check("mass balance", status, s, {"calculix_kg": total * 1000, "wobbl_kg": pm * 1000,
                                             "modal_mass": mm.tolist()})


def _participation(modes, frac, sine, psd, axis, excite) -> Check:
    if sine is None and psd is None:
        cap = frac.sum(0)
        return Check("mass participation", "info", "the modes found move " + ", ".join(
            f"{100 * c:.0f}% of the mass along {a}" for c, a in zip(cap, "xyz")), {"captured": cap.tolist()})
    d = np.asarray(sine.direction if sine is not None else axis, float)
    cap = modes.mass_fraction(d)
    top = max([sine.hi if sine is not None else 0.0, float(psd.f[-1]) if psd is not None else 0.0])
    reach = float(modes.freq[-1])
    s = (f"the {modes.count} modes found, up to {fmt(reach)} Hz, move {100 * cap:.0f}% of the mass along the input "
         f"{_axis_word(d)}")
    if cap >= 0.9 or reach >= 1.5 * top:
        return Check("mass participation", "pass", s, {"captured": cap, "highest_Hz": reach})
    return Check("mass participation", "warn", s + f", and the input reaches {fmt(top)} Hz. The rest of the mass moves "
                                                   "in higher modes, so the response may be low; run with more --modes",
                 {"captured": cap, "highest_Hz": reach})


def _rayleigh_check(r) -> Check:
    if not r or "error" in r or not np.isfinite(r.get("frequency", float("nan"))):
        return Check("hand calculation", "skip", "Rayleigh's quotient could not be formed"
                     + (f" ({r['error'][:120]})" if r and "error" in r else ""))
    fr, fm, f1 = r["frequency"], r["mode_frequency"], r["first"]
    err = fr / fm - 1
    s = (f"Rayleigh's quotient with the static sag under 1 g along {r['direction']} ({fmt(r['sag_mm'])} mm) gives "
         f"{fmt(fr)} Hz, {100 * err:+.1f}% from mode {r['mode']} at {fmt(fm)} Hz")
    if fr < f1 * (1 - 2e-3):
        return Check("hand calculation", "fail", s + ". It should never fall below the first mode, so the modes and "
                                                     "the static solution disagree", r)
    if abs(err) <= 0.1:
        return Check("hand calculation", "pass", s, r)
    return Check("hand calculation", "info", s + ". The sag shape differs from that mode, which is usual when several "
                                                 "modes move mass along the same direction", r)


def _sdof_check(modes, h, sine, zeta) -> Check:
    """One mode alone, at the peak: q = Gamma a / (2 zeta w^2) through that mode's stress at the governing node."""
    f = h["f_peak"]
    i = int(np.argmin(np.abs(np.log(modes.freq / f))))
    g = modes.gamma(sine.direction)[i]
    w = 2 * math.pi * modes.freq[i]
    q = g * sine.accel / (2 * zeta * w * w)
    single = float(modal.vm_amplitude((q * modes.stress[i, h["node"]].astype(float))[None, :] + 0j)[0])
    ratio = single / h["peak"] if h["peak"] > 0 else 0.0
    s = (f"mode {i + 1} alone as a single degree of freedom, Gamma a / (2 zeta w^2), gives {fmt(single)} MPa at the "
         f"governing point, {100 * ratio:.0f}% of the full sum")
    detail = {"mode": i + 1, "single_mode_MPa": single, "ratio": ratio}
    if 0.9 <= ratio <= 1.1:
        return Check("single mode estimate", "pass", s, detail)
    return Check("single mode estimate", "info", s + ", so neighbouring modes add to the peak", detail)


def _cross_check(harmonic, h) -> Check:
    if harmonic is None or "error" in harmonic:
        why = harmonic["error"] if harmonic else "it was not run"
        return Check("solver cross-check", "warn", f"CalculiX's steady state dynamics did not run ({why[:120]}); the "
                                                   "peak comes from Wobbl's own modal sum")
    d = abs(harmonic["at_wobbl_node"] - h["peak"]) / max(h["peak"], 1e-30)
    s = (f"CalculiX's steady state dynamics, reading the same modes, gives {fmt(harmonic['at_wobbl_node'])} MPa where "
         f"Wobbl's modal sum gives {fmt(h['peak'])} MPa ({100 * d:.2g}% apart), and their motion of that point "
         f"across {harmonic['sweep_points']} sweep frequencies agrees to {100 * harmonic['sweep_error']:.2g}%")
    status = "pass" if max(d, harmonic["sweep_error"]) <= 0.02 else ("warn" if max(d, harmonic["sweep_error"]) <= 0.1
                                                                    else "fail")
    return Check("solver cross-check", status, s, {"calculix_MPa": harmonic["at_wobbl_node"], "wobbl_MPa": h["peak"],
                                                   "sweep_error": harmonic["sweep_error"]})


def _miles_check(modes, r, psd, axis, zeta, frac) -> Check:
    """The dominant mode alone under the PSD level at its frequency, the closed form behind Miles' equation."""
    d = np.asarray(axis, float)
    g = modes.gamma(d)
    inband = (modes.freq >= psd.f[0]) & (modes.freq <= psd.f[-1])
    if not inband.any():
        return Check("hand calculation (Miles)", "skip", "no mode lies inside the PSD band")
    share = np.where(inband, g ** 2, 0.0)
    i = int(np.argmax(share))
    level = float(psd(np.array([modes.freq[i]]))[0])
    q_rms = modal.single_mode_rms(g[i], float(modes.freq[i]), zeta, level)
    s_i = float(modal.vm_amplitude((modes.stress[i, r["node"]].astype(float))[None, :] + 0j)[0])
    single = q_rms * s_i
    accel = modal.miles_rms(float(modes.freq[i]), zeta, level)
    ratio = single / r["sigma_rms"] if r["sigma_rms"] > 0 else 0.0
    s = (f"mode {i + 1} at {fmt(modes.freq[i])} Hz alone, at {level:.3g} g2/Hz (Miles: {fmt(accel)} grms of response), "
         f"gives {fmt(single)} MPa rms at the governing point, {100 * ratio:.0f}% of the full sum of "
         f"{fmt(r['sigma_rms'])} MPa")
    detail = {"mode": i + 1, "single_mode_rms_MPa": single, "miles_grms": accel, "ratio": ratio}
    if 0.8 <= ratio <= 1.1:
        return Check("hand calculation (Miles)", "pass", s, detail)
    return Check("hand calculation (Miles)", "info", s + ", so other modes add to the response there", detail)


def _singular(raw: float, gov: float, radius: float, what: str) -> Check:
    if raw <= 1.05 * gov:
        return Check("singularities", "pass", f"no support or sharp corner peak is above the reported {what}")
    return Check("singularities", "info" if raw <= 1.5 * gov else "warn",
                 f"the highest {what} anywhere is {fmt(raw)} MPa, at a support, a rigid attachment or a sharp corner "
                 f"where its value depends on the idealization; Wobbl reports it {fmt(radius)} mm or more away from "
                 f"those spots ({fmt(gov)} MPa)", {"raw_max_MPa": raw})


def _sanity(modes, total) -> Check:
    f1 = float(modes.freq[0])
    s = f"first mode {fmt(f1)} Hz, mass {fmt(total * 1e6)} g"
    if not 0.1 <= f1 <= 1e6:
        return Check("units and sanity", "warn", s + "; a first mode this far out usually means a unit slip in the "
                                                     "file or the material")
    return Check("units and sanity", "pass", s + ": plausible")


def _random_fatigue(mat, r: dict, dur: float) -> dict | None:
    from ..materials import fatigue_data

    data = fatigue_data(mat)
    if data is None:
        return None
    surface = "machined" if mat.category == "metal" else "polished"
    curve = F.build_curve(mat, data, surface, None, "bending", 0.9)
    blocks = F.steinberg_blocks(r["sigma_rms"], r["rate"], dur)
    D, _ = F.damage(blocks, curve)
    return {"damage": D, "sf": F.factor_to_failure(blocks, curve), "cycles": r["rate"] * dur, "surface": surface,
            "reliability": 0.9, "curve": curve.describe(), "source": data.source, "estimated": data.estimated}


def metrics(values: dict) -> list[tuple[str, str]]:
    out = [("first mode", f"{fmt(values['frequencies'][0])} Hz")]
    res = values.get("resonance")
    if res:
        near = min(res, key=lambda r: r["separation"])
        out.append(("nearest excitation", f"mode {near['mode']}, {100 * near['separation']:.0f}% away"))
    if values.get("harmonic"):
        out.append(("sine peak", f"{fmt(values['harmonic']['peak_MPa'])} MPa"))
    if values.get("random"):
        out.append(("3 sigma stress", f"{fmt(values['random']['three_sigma_MPa'])} MPa"))
    if values.get("sf") is not None:
        out.append(("safety factor", f"{values['sf']:.2f}"))
    if len(out) < 4:
        out.append(("damping", f"{100 * values.get('damping', 0):g}%"))
    return out[:4]


def assumptions(mat, zeta_why, masses, deck, sine, psd, base) -> list[str]:
    a = ["Linear elastic, small vibrations about the unloaded shape. Supports are as rigid as modeled; real mounts "
         "flex, which lowers every natural frequency.",
         f"Damping is the same for every mode: {zeta_why}.",
         f"{mat.name}: E {fmt(mat.E)} MPa, Poisson {mat.nu:g}, density {fmt(mat.density)} kg/m3, {mat.criterion} "
         f"{fmt(mat.allowable)} MPa ({mat.source})."]
    for k, m in enumerate(masses):
        how = (f"a point at ({', '.join(fmt(v) for v in m.point)}) held rigidly by its faces, with no rotary inertia "
               "of its own" if m.rigid else "spread over its faces by area, adding no stiffness")
        a.append(f"Mass {k + 1}, {fmt(m.mass * 1e6)} g ({m.describe()}), is {how}.")
    if sine is not None:
        a.append(f"The sine sweep is a base acceleration of {fmt(sine.accel / modal.G0)} g along "
                 f"{_axis_word(sine.direction)} from {fmt(sine.lo)} to {fmt(sine.hi)} Hz, slow enough to reach "
                 "steady state at every frequency.")
    if psd is not None:
        a.append(f"The random input is a stationary Gaussian base acceleration, {psd.describe()}. CalculiX has no "
                 "random response step, so Wobbl sums it from CalculiX's modes, with the mean square von Mises stress "
                 "from Segalman's method. The 3 sigma stress is exceeded 0.27% of the time.")
    if sine is not None or psd is not None:
        a.append("The response counts the modes found; modes above them add a little that is not counted, and the "
                 "mass participation check says how much mass they carry.")
    made = process_caveat(mat, base.print_direction)
    if made:
        a.append(made)
    return a + list(base.assumptions)


def fixes(values: dict, mat, features) -> list[dict]:
    out = []
    margin = values.get("margin", 0.2)
    bad = [r for r in values.get("resonance", []) if r["status"] in ("resonant", "close")]
    if bad:
        r = min(bad, key=lambda r: r["separation"])
        f, excite, lo, hi = r["frequency"], r["excitation"], r["lo"], r["hi"]
        up = hi * (1 + margin) / f
        down = lo / (1 + margin) / f
        out.append({"kind": "stiffen", "title": f"Stiffen to move mode {r['mode']} above {fmt(hi * (1 + margin))} Hz",
                    "summary": f"A natural frequency grows with the square root of stiffness, so mode {r['mode']} at "
                               f"{fmt(f)} Hz needs about {up ** 2:.2g} times the stiffness to clear the {excite} by "
                               f"{100 * margin:.0f}%. For a beam carrying a heavy end mass that is a section about "
                               f"{up ** (2 / 3):.2g} times as deep in the bending direction, and when the beam's own "
                               f"mass dominates about {up:.2g} times as deep.",
                    "params": {"stiffness_factor": up ** 2, "target_Hz": hi * (1 + margin)}, "auto": False})
        if down > 0.2:
            out.append({"kind": "soften", "title": f"Or move mode {r['mode']} below {fmt(lo / (1 + margin))} Hz",
                        "summary": f"Softer mounts or isolators that bring it to {down ** 2:.2g} times the stiffness "
                                   "put the mode under the excitation, which works when the speed only passes through "
                                   "it on the way up.", "params": {"stiffness_factor": down ** 2}, "auto": False})
        if values.get("masses_g"):
            out.append({"kind": "lighten", "title": "Lighten the attached mass",
                        "summary": f"With an end mass dominating, the frequency goes as 1/sqrt(mass), so {up:.2g} times "
                                   f"the frequency needs about {1 / up ** 2:.2g} of the mass.", "auto": False})
        out.append({"kind": "damping", "title": "Add damping",
                    "summary": f"At resonance the response is about 1/(2 zeta) of the static one, "
                               f"{1 / (2 * values.get('damping', 0.02)):.3g} times now. A damping layer or a damped "
                               "mount that doubles the damping halves it.", "auto": False})
    sf, target = values.get("sf"), values.get("target_sf", 1.5)
    if sf is not None and sf < target:
        need = target / sf
        zeta = values.get("damping", 0.02)
        rnd = values.get("random") is not None and values["random"]["sf"] <= sf * 1.0001
        z_new = zeta * (need ** 2 if rnd else need)
        out.append({"kind": "damping", "title": f"Raise damping to about {100 * z_new:.2g}%",
                    "summary": ("Random response near a resonance falls as 1/sqrt(damping) (Miles)" if rnd else
                                "The peak of a sine sweep falls as 1/damping") +
                               f", so {100 * zeta:g}% becoming {100 * z_new:.2g}% gives the target factor of {target:g}."
                               , "params": {"damping": z_new}, "auto": False})
        t = features.thickness_typical
        out.append({"kind": "thicken", "title": "Thicken where the stress peaks, and raise the mode",
                    "summary": f"The response stress has to fall {need:.2g} times. Adding depth where the part bends "
                               f"lowers the stress and raises the frequency; for a flat input a single mode's random "
                               f"motion falls as f^-1.5. Start with the {fmt(t)} mm walls near the peak.",
                    "auto": False})
    return out


def images(img: Path, mesh, modes, excite, sine, psd, axis, resp, harmonic, margin) -> dict:
    from .render import frequency_map, render_field, render_modes, response_curve

    out = {}
    try:
        total = float(modes.total_mass[:3].mean())
        frac = modes.effective_mass[:, :3] / max(total, 1e-300)
        rel = np.nonzero(relevant_modes(modes, frac))[0][:3]
        pick = list(rel) if len(rel) else list(range(min(3, modes.count)))
        out["modes"] = render_modes(img / "modes.png", mesh.X, mesh.tets,
                                    [modes.shapes[i, :mesh.n_nodes].astype(float) for i in pick],
                                    [f"mode {i + 1}, {fmt(modes.freq[i])} Hz" for i in pick])
        d = np.asarray(sine.direction if sine is not None else axis, float)
        w = (modes.gamma(d) ** 2) / max(total, 1e-300) if (sine is not None or psd is not None) else frac.max(1)
        tones = [(e.lo, e.label) for e in excite if e.tone]
        bands = [(e.lo, e.hi, e.label) for e in excite if not e.tone]
        inputs = []
        if sine is not None:
            inputs.append((sine.lo, sine.hi, "sine sweep"))
        if psd is not None:
            inputs.append((float(psd.f[0]), float(psd.f[-1]), "random input"))
        out["frequencies"] = frequency_map(img / "frequencies.png", modes.freq, w, tones, bands, margin, inputs)
        if resp is not None and "harmonic" in resp:
            h = resp["harmonic"]
            marks = [(harmonic["f"], harmonic["at_wobbl_node"], "CalculiX at the peak")] \
                if harmonic and "error" not in harmonic else []
            out["response"] = response_curve(img / "response.png", h["curve_f"], {"von Mises amplitude": h["curve"]},
                                             "MPa at the governing point", marks)
            vm = harmonic["vm"] if harmonic and "error" not in harmonic else h["vm"]
            peak = harmonic["peak"] if harmonic and "error" not in harmonic else h["peak"]
            xyz = harmonic["xyz"] if harmonic and "error" not in harmonic else h["xyz"]
            out["stress"] = render_field(img / "stress.png", mesh.X, mesh.tets, mesh.tris, vm, "sine peak", "MPa",
                                         xyz, f"{fmt(peak)} MPa")
        if resp is not None and "random" in resp:
            r = resp["random"]
            key = "stress" if "stress" not in out else "random"
            out[key] = render_field(img / f"{key}.png", mesh.X, mesh.tets, mesh.tris, 3 * r["rms"], "3 sigma", "MPa",
                                    r["xyz"], f"{fmt(3 * r['sigma_rms'])} MPa")
    except Exception as e:  # noqa: BLE001  pictures are a convenience, never a reason to fail
        out["error"] = str(e)[:200]
    return out
