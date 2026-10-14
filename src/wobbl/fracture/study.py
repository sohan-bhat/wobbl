"""Crack growth: from a crack in a part to where it goes, when it gets there, and how lives spread over many parts.

The part is solved without the crack to find where the crack starts and which way it faces. The crack is cut
into the part and grown step by step, each step a fresh mesh and solve, so the path comes from the geometry and
loads. The K at each crack size along that path is the surrogate for the life: the growth law integrated over it
gives the nominal life, and a Monte Carlo over the starting flaw, the growth constants, the load and the toughness
gives B50, B10 and B1, the chance of failing before a target and inspection times. Every answer comes with checks:
K on a coarser front mesh, a handbook or weight function K, the force balance, the integration step and the
Monte Carlo sample count, and Wheeler retardation against a cycle by cycle reference when it is used.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..materials import CRACK_GROWTH, LIBRARY, resolve_material
from ..units import fmt
from ..verify.checks import Check, balance_check, confidence
from . import handbook as H
from .growth import GrowthLaw, law_for, parse_spectrum
from .life import CrackPath, LifeInputs, Scatter, growth_curve, inspection, lives, monte_carlo, wheeler_reference
from .spec import FractureStudy

P10 = 1.2816      # standard normal quantile of the smallest tenth
GROWTH_STEP = 0.3   # each step moves the fastest front point this fraction of the crack's size


def _counts(n: float) -> str:
    if not np.isfinite(n):
        return "no failure"
    return f"{n:,.0f}" if n < 1e5 else fmt(n)


def _pct(x: float) -> str:
    """A share as a percentage with up to three significant digits, 100% rather than 1e+02%."""
    return f"{100 * x:.3g}%"


def run_fracture(st: FractureStudy, out_dir: str | Path, progress=None, quick: bool = False):
    """Run a crack growth study. quick skips the pictures and the fixes, which rerun the Monte Carlo."""
    from ..mesh.mesher import auto_sizing
    from ..outcome_generic import GenericOutcome
    from ..solvers import base as solvers
    from ..structural.build import build_static
    from ..structural.run import describe_part, resolve_selections
    from .path import front_size, grow
    from .place import PartShape, auto_crack, place, section, solve_uncracked, stress_profile, structural_study

    t0 = time.time()
    say = progress or (lambda a, b: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    material = resolve_material(st.material)
    law = law_for(material, st.law, st.growth, st.kc)
    record = CRACK_GROWTH.get(material.key.rstrip("*"))
    spectrum = parse_spectrum(st.spectrum, st.r_ratio)
    adapter = solvers.for_domain("structural")
    adapter.require()
    say("geometry", f"reading {Path(st.part).name}")
    prep, features = describe_part(st.part, work, st.scale, None, st.params)
    spec = st.crack
    unc, origin = solve_uncracked(st, prep, features, material, adapter, work, spec, say)
    if spec.shape == "auto":
        spec = auto_crack(spec, unc, origin, features)
        st = replace(st, crack=spec)
    h = auto_sizing(prep, thickness=features.thickness_typical).h
    start = math.exp(-P10 * st.flaw_scatter)
    spec0 = replace(spec, a=spec.a * start, c=None if spec.c is None else spec.c * start)
    first = place(spec0, unc, origin, features, front_size(spec0.a, None, h, 0.0, st.quality))
    say("crack", f"{spec.shape} crack {first.where}")
    sst = structural_study(st)
    loads, fixtures, _ = resolve_selections(sst, features)

    def build_model(cm):
        return build_static(cm.mesh, loads, fixtures, material, features.faces)

    scatter = Scatter(st.flaw_scatter, st.growth_scatter if st.growth_scatter is not None else
                      (record.scatter if record else 0.25), st.load_scatter, 0.07, st.samples)
    flow = 0.5 * (material.yield_strength + material.tensile_strength)
    area0, force0 = section(unc, first)
    hi = math.exp(3 * scatter.load) * spectrum.peak
    kc_hi = law.kc * math.exp(3 * scatter.toughness) if law.kc > 0 else math.inf
    stop_k = kc_hi / hi

    def stop_net(size, area):
        return area0 > 0 and abs(force0) / max(area0 - area, 1e-9) * hi >= flow

    part = PartShape(unc.mesh)
    res_path = grow(prep, first.crack, build_model, material, law, spectrum, adapter, work, h, first.wall,
                    st.quality, st.steps, GROWTH_STEP, stop_k, stop_net, st.budget, say, st.keep, part)
    steps = res_path.steps
    s = np.array([x.size for x in steps])
    k = np.array([x.k_lead for x in steps])
    area = np.array([x.area for x in steps])
    x_prof, sig_prof = stress_profile(unc, first, max(spec.a, s[0]) * 1.05)
    small = _small_crack(x_prof, sig_prof, s[0], k[0])
    solved = len(s)
    end_fails = res_path.end_fails
    if res_path.stuck or res_path.dormant or res_path.cut:
        # past the last crack K follows the square root of size: a part whose crack is below threshold there
        # never fails, and one with a bigger flaw or load grows on this estimate
        s, k, area = _extrapolate(s, k, area, res_path, first, part)
        end_fails = True
    elif len(s) == 1:   # one crack and nothing after it: give the table a second point so it can be read
        s, k, area = np.append(s, s[0] * 1.001), np.append(k, k[0] * 1.0005), np.append(area, area[0])
    cpath = CrackPath(s, k, area, area0, force0, small, res_path.end, end_fails)
    cpath.solved = solved
    inp = LifeInputs(law, spectrum, flow, st.wheeler)
    say("life", f"Monte Carlo over {scatter.samples:,} parts")
    mc = monte_carlo(cpath, inp, spec.a, scatter, st.target_life)
    insp = inspection(cpath, inp, spec.a, st.detectable)
    checks = _checks(st, res_path, first, cpath, inp, mc, unc, x_prof, sig_prof, material, spectrum)
    level, reasons = confidence(checks)
    verdict, head = _verdict(st, mc, first, spec, spectrum)
    try:
        fixes = [] if quick else _fixes(st, cpath, inp, scatter, mc, law, material, insp, spectrum)
    except Exception as e:  # noqa: BLE001  fixes are advice, so a failed one never costs the answer
        fixes, res_path.notes = [], res_path.notes + [f"The fixes could not be worked out ({str(e)[:120]})."]
    say("render", "drawing the crack path and the lives")
    curve = growth_curve(cpath, inp, float(cpath.s[0]))
    n_at = np.interp([x.size for x in steps], curve[0], curve[1]) if len(curve[0]) else np.full(len(steps), np.nan)
    images = {} if quick else _images(out, unc, res_path, curve, n_at, mc, st, first)
    assumptions = _assumptions(st, law, record, material, spectrum, scatter, first, start, flow, cpath, res_path)
    p = mc.percentiles
    metrics = [("nominal life", f"{_counts(mc.nominal)} cycles"), ("B10", f"{_counts(p['B10'])} cycles"),
               ("B1", f"{_counts(p['B1'])} cycles")]
    if st.target_life:
        metrics.append(("fail before target", _pct(mc.pof)))
    if np.isfinite(insp["repeat"]):
        metrics.append(("inspect every", f"{_counts(insp['repeat'])} cycles"))
    values = {"nominal_cycles": mc.nominal, "percentiles": p, "b1_band": list(mc.b1_band), "pof": mc.pof,
              "pof_band": list(mc.pof_band), "inspection": insp, "path_end": res_path.end, "origin": first.origin,
              "normal": first.normal, "where": first.where, "crack": spec.describe(), "crack_reason": spec.reason,
              "start_size_mm": float(s[0]), "steps": len(steps),
              "k_table": {"size_mm": s.tolist(), "k_mpa_sqrt_m": k.tolist(), "area_mm2": area.tolist()},
              "section": {"area_mm2": area0, "force_N": force0, "flow_stress_MPa": flow},
              "law": law.to_dict(), "spectrum": spectrum.describe(), "samples": scatter.samples,
              "ran_out": mc.ran_out, "stress_at_origin_MPa": first.stress,
              "_figures": [["path", "The crack path, colored by the cycle count when the front passed"],
                           ["growth", "Crack size against cycles for the nominal part"],
                           ["life", "How lives spread over many parts"]]}
    values["playback"] = _playback(out, unc, res_path, curve, n_at, mc, insp, st, first, verdict, head, material,
                                   spectrum)
    o = GenericOutcome("fracture", str(Path(st.part).resolve()), st.question, verdict, head, metrics, level, reasons,
                       checks, assumptions, values, images, fixes, prep.notes + res_path.notes,
                       solver=unc.solver, t_total=time.time() - t0)
    o.write(out)
    if not st.keep:
        shutil.rmtree(work, ignore_errors=True)
    return o


def _extrapolate(s, k, area, res_path, first, part):
    """When a step cannot be meshed, carry K on past the last solved crack as K grows with the square root of size,
    the same shape the crack had, up to the far side of the wall ahead of the last front, where it counts as
    failed. The checks say how much of the life rests on this."""
    last = np.asarray(res_path.steps[-1].fronts[0]["xyz"]) if res_path.steps[-1].fronts else None
    mid = last[len(last) // 2] if last is not None else first.origin
    way = first.toward
    if last is not None and len(res_path.steps) > 1 and res_path.steps[-2].fronts:
        prev = np.asarray(res_path.steps[-2].fronts[0]["xyz"])
        d = mid - prev[len(prev) // 2]
        if np.linalg.norm(d) > 1e-9:
            way = d / np.linalg.norm(d)
    ahead = part.ahead(mid, way, 1e-6 * s[-1])
    if not np.isfinite(ahead) or ahead <= 0.02 * s[-1]:
        ahead = 0.02 * s[-1]
    ext = s[-1] + ahead * np.linspace(0.2, 1.0, 5)
    k_ext = k[-1] * np.sqrt(ext / s[-1])
    a_ext = area[-1] * (ext / s[-1]) ** 2
    return np.concatenate([s, ext]), np.concatenate([k, k_ext]), np.concatenate([area, a_ext])


def _small_crack(x, sig, s0: float, k0: float):
    """K below the smallest crack solved: the edge crack weight function over the stress the crack's line carried
    before the crack, scaled to the solved K at that size."""
    k_wf0 = H.edge_weight_function(x, sig, s0)
    if not (np.isfinite(k_wf0) and k_wf0 > 0):
        return None
    scale = k0 / k_wf0

    def small(s):
        s = np.atleast_1d(np.asarray(s, float))
        return np.array([scale * H.edge_weight_function(x, sig, max(v, 1e-6)) for v in s])
    return small


def _checks(st, res_path, first, cpath, inp, mc, unc, x_prof, sig_prof, material, spectrum) -> list[Check]:
    checks = []
    conv = res_path.convergence
    if len(conv) >= 2:
        vals = [c["k_lead"] for c in conv]
        ch = abs(vals[-1] - vals[-2]) / max(abs(vals[-1]), 1e-12)
        trend = " -> ".join(fmt(v) for v in vals)
        sizes = " and ".join(fmt(c["h_front"]) for c in conv)
        status = "pass" if ch <= 0.03 else ("warn" if ch <= 0.08 else "fail")
        checks.append(Check("SIF mesh convergence", status,
                            f"largest K at the starting crack {trend} MPa sqrt(m) with front elements of {sizes} mm, "
                            f"last change {100 * ch:.1f}%", {"levels": conv, "change": ch}))
    try:
        checks.append(_handbook(st, first, cpath, unc, x_prof, sig_prof))
    except Exception as e:  # noqa: BLE001  a hand check that cannot be made never stops the answer
        checks.append(Check("hand calculation", "skip", f"no handbook K could be formed here ({str(e)[:120]})"))
    if res_path.balance is not None:
        (fa, ma), (fr, mr), size, partial = res_path.balance
        checks.append(balance_check(fa, fr, ma, mr, size, partial))
    a0 = st.crack.a
    n1 = float(lives(cpath, inp, a0, 1.0, 1.0, inp.law.kc, n_grid=200)[0][0])
    n2 = float(lives(cpath, inp, a0, 1.0, 1.0, inp.law.kc, n_grid=400)[0][0])
    if np.isfinite(n2) and n2 > 0:
        ch = abs(n1 - n2) / n2
        checks.append(Check("integration step", "pass" if ch <= 0.005 else "warn",
                            f"nominal life {_counts(n1)} cycles with 200 steps in crack size and {_counts(n2)} with 400, "
                            f"{100 * ch:.2g}% apart", {"n200": n1, "n400": n2}))
    p = mc.percentiles
    lo, hi = mc.b1_band
    if p["B1"] > 0 and np.isfinite(p["B1"]):
        half = 0.5 * (hi - lo) / p["B1"]
        drift = abs(mc.half_b1 - p["B1"]) / p["B1"]
        status = "pass" if half <= 0.05 else ("warn" if half <= 0.15 else "fail")
        checks.append(Check("Monte Carlo samples", status,
                            f"B1 {_counts(p['B1'])} cycles from {mc.lives.size:,} parts, 90% sampling band "
                            f"{_counts(lo)} to {_counts(hi)} (within {100 * half:.2g}%), and {_counts(mc.half_b1)} from "
                            f"the first half of the parts ({100 * drift:.2g}% off)", {"band": [lo, hi], "half": half}))
    if inp.wheeler > 0 and not spectrum.constant:
        ref = wheeler_reference(cpath, inp, a0, material.yield_strength)
        if np.isfinite(ref) and ref > 0 and np.isfinite(mc.nominal):
            ch = abs(mc.nominal - ref) / ref
            status = "pass" if ch <= 0.05 else ("warn" if ch <= 0.15 else "fail")
            checks.append(Check("retardation", status,
                                f"Wheeler exponent {inp.wheeler:g}: the block factors give {_counts(mc.nominal)} cycles "
                                f"and following each overload zone cycle by cycle gives {_counts(ref)} "
                                f"({100 * ch:.2g}% apart)", {"block": mc.nominal, "cycle_by_cycle": ref}))
    s = cpath.s
    ran = mc.ran_out
    path_note = (f"{len(res_path.steps)} solved cracks from {fmt(s[0])} to {fmt(s[cpath.solved - 1])} mm, ended by "
                 f"{res_path.end}")
    if res_path.dormant:
        grows = float(np.isfinite(mc.lives).mean())
        status = "pass" if grows <= 0.01 else "warn"
        checks.append(Check("crack path", status,
                            path_note + f". The starting crack is below the growth threshold at these loads, so it "
                            f"does not grow in a typical part. {_pct(grows)} of the parts sampled, those with "
                            "larger flaws or loads, grow past it, on K carried on as the square root of size",
                            {"growing_share": grows}))
    elif res_path.stuck or res_path.cut:
        n_solved = float(lives(cpath, inp, a0, 1.0, 1.0, inp.law.kc, n_grid=400)[0][0])
        to_last = _life_to(cpath, inp, a0, float(s[cpath.solved - 1]))
        share = 1.0 - to_last / n_solved if np.isfinite(n_solved) and n_solved > 0 else 1.0
        if res_path.cut:   # the wall ahead is all that is left, so the estimate covers the crack's last stretch
            status = "pass" if share <= 0.25 else "warn"
        else:
            status = "warn" if share <= 0.25 else "fail"
        checks.append(Check("crack path", status,
                            path_note + f". Past {fmt(s[cpath.solved - 1])} mm K is carried on as the square root "
                            f"of size to the far side of the wall at {fmt(s[-1])} mm, and {_pct(share)} of the "
                            "nominal life rests on that estimate", {"extrapolated_share": share}))
    elif ran:
        share = ran / mc.lives.size
        status = "warn" if share <= 0.1 else "fail"
        text = (path_note + f". {_pct(share)} of the parts were still growing at its end and count their life "
                "there, which is short of their real life")
        if status == "fail":
            text += ", so the percentiles are lower bounds. More steps or a longer budget follow the crack further"
        checks.append(Check("crack path", status, text, {"ran_out": ran}))
    elif cpath.end_fails:
        checks.append(Check("crack path", "pass", path_note + ". Parts whose crack reaches its end count as failed "
                                                              "there, which leaves out the last cycles and errs short"))
    else:
        checks.append(Check("crack path", "pass", path_note + "; every part sampled fails on the solved path"))
    if inp.law.estimated:
        checks.append(Check("material data", "warn",
                            f"no crack growth data is built in for {material.name}, so its constants are estimated "
                            "from its stiffness and lives can be off by a factor of a few. Measured constants given "
                            "as growth 'C=..,n=..,dkth=..,kic=..' replace them", {"source": inp.law.source}))
    return checks


def _handbook(st, first, cpath, unc, x_prof, sig_prof) -> Check:
    """K at the starting crack against a handbook value from the stress the crack's faces carried before it."""
    shape, a0 = st.crack.shape, float(cpath.s[0])
    k_fe = float(cpath.k[0])
    if shape == "through":
        k_hb = H.edge_weight_function(x_prof, sig_prof, a0)
        what = "the edge crack weight function (Glinka and Shen) over the uncracked stress along the crack"
    elif shape == "surface":
        ratio = st.crack.a / (st.crack.c or st.crack.a)
        if ratio > 1.0 + 1e-6:
            return Check("hand calculation", "skip", "Newman and Raju's deepest point form covers a / c up to 1")
        t = _depth(unc, first)
        k_hb = H.newman_raju(first.stress, a0, a0 / min(ratio, 1.0), max(t, 1.25 * a0), 1e6)
        what = (f"Newman and Raju at the deepest point, with the {fmt(first.stress)} MPa at the origin taken as "
                "uniform tension")
    else:
        return Check("hand calculation", "skip", f"no handbook K for a {shape} crack in this geometry")
    if not (np.isfinite(k_hb) and k_hb > 0):
        return Check("hand calculation", "skip", "the uncracked stress along the crack could not be read")
    ratio = k_fe / k_hb
    status = "pass" if abs(ratio - 1) <= 0.10 else "warn"
    s = (f"{what} gives {fmt(k_hb)} MPa sqrt(m) for the {fmt(a0)} mm starting crack; the finite element K is "
         f"{fmt(k_fe)} ({100 * (ratio - 1):+.1f}%)")
    if status == "warn":
        s += ". The handbook case ignores the part's shape away from the crack, so a gap this size is expected " \
             "when the crack is large next to its feature"
    return Check("hand calculation", status, s, {"handbook": k_hb, "fea": k_fe})


def _life_to(cpath, inp, a0: float, size: float) -> float:
    """Cycles for the nominal part to grow from a0 to ``size``."""
    sizes, cycles = growth_curve(cpath, inp, a0)
    if not len(sizes):
        return 0.0
    return float(np.interp(size, sizes, cycles))


def _depth(unc, first) -> float:
    from .place import _first
    return _first(first.origin, first.toward, unc.mesh, skip=0.1 * first.crack.size)


def _verdict(st: FractureStudy, mc, first, spec, spectrum) -> tuple[str, str]:
    p = mc.percentiles
    where = first.where
    crack = f"A {fmt(spec.a)} mm {spec.shape} crack {where}"
    if np.isfinite(mc.nominal):
        blocks = "" if spectrum.constant else f", {fmt(mc.nominal / spectrum.cycles)} passes through the load block"
        life = f"{crack} fails after about {_counts(mc.nominal)} cycles in a typical part{blocks}."
    else:
        life = f"{crack} stops growing in a typical part, below the growth threshold."
    if np.isfinite(p["B10"]):
        life += f" One part in ten fails by {_counts(p['B10'])} cycles and one in a hundred by {_counts(p['B1'])}."
    elif np.isfinite(p["B1"]):
        life += f" One part in a hundred fails by {_counts(p['B1'])} cycles, and fewer than one in ten fail at all."
    else:
        life += " Fewer than one part in a hundred fails at all, because the crack stops below the threshold."
    if not st.target_life:
        return "life", life
    pof = mc.pof
    if pof <= st.max_pof:
        verdict = "ok"
    elif p["B10"] >= st.target_life:
        verdict = "marginal"
    else:
        verdict = "fails"
    tail = f" {_pct(pof)} of parts fail before the {_counts(st.target_life)} cycle target (allowed {_pct(st.max_pof)})."
    return verdict, life + tail


def _b1(cpath, inp, draws, scale: float = 1.0) -> float:
    n, _ = lives(cpath, inp, draws["a0"], draws["scale"] * scale, draws["cmult"], draws["kc"])
    return float(np.quantile(n, 0.01))


def _fixes(st, cpath, inp, scatter, mc, law, material, insp, spectrum) -> list[dict]:
    fixes = []
    small = replace(scatter, samples=min(scatter.samples, 4000))
    draws = small.draw(st.crack.a, law.kc)
    target = st.target_life
    if target and mc.pof > st.max_pof:
        lo, hi = 0.05, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if _b1(cpath, inp, draws, mid) >= target:
                lo = mid
            else:
                hi = mid
        cut = 1 - lo
        fixes.append({"kind": "stress", "title": f"Lower the stress at the crack's origin by {100 * cut:.0f}%",
                      "summary": f"Growth goes as the stress range to the power {law.n:g}, so lowering the stress "
                                 f"where the crack starts by {100 * cut:.0f}% (a larger radius, a thicker section or a "
                                 f"smaller load) brings B1 to the {_counts(target)} cycle target.", "auto": False})
    base = _b1(cpath, inp, draws)
    best = []
    for key, rec in CRACK_GROWTH.items():
        if key == material.key.rstrip("*"):
            continue
        other = law_for(key, law.kind)
        m = LIBRARY[key]
        alt = replace(inp, law=other, flow_stress=0.5 * (m.yield_strength + m.tensile_strength))
        d = dict(draws, kc=other.kc * np.exp(scatter.toughness * np.random.default_rng(scatter.seed + 1)
                                              .standard_normal(len(draws["a0"]))) if other.kc else
                 np.zeros(len(draws["a0"])))
        best.append((_b1(cpath, alt, d), key, m.name))
    best.sort(reverse=True)
    for b1, key, name in best[:2]:
        if np.isfinite(base) and b1 > 1.2 * base:
            fixes.append({"kind": "material", "title": f"Use {name}",
                          "summary": f"With {name}'s crack growth constants on the same path, B1 is {_counts(b1)} "
                                     f"cycles against {_counts(base)} now. Its stiffness does not change K under the "
                                     "same loads, but check its strength and weight with a strength study.",
                          "auto": False, "material": key})
    if not np.isfinite(insp["repeat"]):
        return fixes
    fixes.append({"kind": "inspection", "title": f"Inspect every {_counts(insp['repeat'])} cycles",
                  "summary": f"Damage tolerance practice: the first inspection at half the nominal life from the "
                             f"starting crack ({_counts(insp['first'])} cycles), then every half of the life from a "
                             f"{fmt(insp['detectable_mm'])} mm crack, the size an inspection is taken to find "
                             f"reliably, to failure ({_counts(insp['life_from_detectable'])} cycles).", "auto": False})
    return fixes


def _images(out, unc, res_path, curve, n_at, mc, st, first) -> dict:
    from .render import growth_image, life_image, path_image

    img = {}
    size, cycles = curve
    steps = res_path.steps
    sides = _sides(res_path, first)
    try:
        img["path"] = path_image(out / "images" / "path.png", unc.mesh.X, unc.mesh.tets,
                                 [s["fronts"] for s in sides], list(n_at), first.origin, "origin",
                                 [s["start"] for s in sides])
        if len(size):
            img["growth"] = growth_image(out / "images" / "growth.png", cycles, size, n_at,
                                         [x.size for x in steps])
        img["life"] = life_image(out / "images" / "life.png", mc.lives, mc.percentiles, st.target_life)
    except Exception as e:  # noqa: BLE001  pictures never stop an answer
        img["error"] = str(e)[:200]
    return img


def _sides(res_path, first) -> list[dict]:
    """Each side of the crack (one, or two for a crack with two tips) as its fronts in the order the crack grew
    through them, and the line on the part's surface or through its middle that the starting flaw grew from."""
    sides: dict[int, list[np.ndarray]] = {}
    for x in res_path.steps:
        for f in x.fronts:
            sides.setdefault(0 if f["section"] == 0 else 1, []).append(np.asarray(f["xyz"]))
    out = []
    for fronts in sides.values():
        f0 = fronts[0]
        rel = f0 - first.origin
        out.append({"fronts": fronts, "start": f0 - np.outer(rel @ first.toward, first.toward)})
    return out


def _playback(out, unc, res_path, curve, n_at, mc, insp, st, first, verdict, head, material, spectrum) -> str:
    """playback.json, the run in a form a viewer can animate."""
    from . import playback

    steps = [dict(x.to_dict(), cycles=float(n)) for x, n in zip(res_path.steps, n_at)]
    try:
        part = playback.part_surface(unc.mesh.X, unc.mesh.tris)
        playback.write(out / "playback.json", part=part, origin=first.origin, normal=first.normal, steps=steps,
                       starts=[s["start"] for s in _sides(res_path, first)],
                       growth=curve, lives=mc.lives, percentiles=mc.percentiles, target=st.target_life, pof=mc.pof,
                       inspection=insp, verdict=verdict, headline=head, material=material.name,
                       spectrum=spectrum.describe(), end=res_path.end)
    except Exception:  # noqa: BLE001  the playback is a convenience, never a reason to fail
        return ""
    return "playback.json"


def _assumptions(st, law: GrowthLaw, record, material, spectrum, scatter, first, start, flow, cpath,
                 res_path) -> list[str]:
    out = [f"{material.name}: {law.kind.upper() if law.kind == 'nasgro' else 'Paris'} growth with C "
           f"{law.C:.3g} mm/cycle and n {law.n:g}"
           + (f", threshold {law.dk1:g} MPa sqrt(m) at R = 0" if law.dk1 else "")
           + (f", toughness {law.kc:g} MPa sqrt(m)" if law.kc else ", no toughness, so failure is net section yield")
           + f". Source: {law.source or 'user supplied'}.",
           f"Loads cycle as {spectrum.describe()}, with the loads given as the peak.",
           *([st.crack.reason] if st.crack.reason else []),
           f"The crack plane at the start is normal to the largest principal stress at the origin, {fmt(first.stress)} "
           f"MPa before the crack.",
           f"The path starts from a crack {start:.2g} times the size given, the smallest tenth of the starting flaws "
           "sampled; smaller cracks follow the uncracked stress through the edge crack weight function, matched to "
           "the first solved K.",
           "K for growth combines the three modes by energy, sqrt(K_I^2 + K_II^2 + K_III^2 / (1 - nu)), and the "
           "turn uses K_I and K_II only (maximum tangential stress).",
           f"Failure is K_max reaching the toughness or the average stress over the remaining ligament reaching the "
           f"flow stress, {fmt(flow)} MPa, the mean of yield and tensile strength. Bending in the ligament is not "
           "counted, so net section yield is approximate.",
           f"Parts scatter as lognormal: starting flaw {scatter.flaw:g}, growth coefficient {scatter.growth:g}, load "
           f"{scatter.load:g}, toughness {scatter.toughness:g} (standard deviations of the logarithm). One part's "
           "failure cycle cannot be predicted; the percentiles describe many parts.",
           "The Monte Carlo keeps the solved path's shape and changes only how fast a part's crack travels along it.",
           "Linear elastic fracture mechanics: crack tip plasticity, crack face contact and environment are not "
           "modeled."]
    if res_path.end_fails:
        out.append("A crack that cuts through the wall it grows across counts as a failed part, even where another "
                   "load path remains, such as the far side of a lug, which errs short.")
    if st.wheeler > 0 and not spectrum.constant:
        out.append(f"Wheeler retardation with exponent {st.wheeler:g}, from the plane stress plastic zone.")
    elif not spectrum.constant:
        out.append("No retardation after large cycles, which is conservative; give --wheeler with an exponent from "
                   "tests on this material and spectrum to include it.")
    return out
