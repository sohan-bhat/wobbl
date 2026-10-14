"""A fatigue answer from a finished strength or vibration result, or from stresses you give directly.

From a strength result the peak stress cycles with the load: between R times the studied load and the load itself,
or through a duty cycle of blocks ('100% x 2e4', '60% x 1e6'). The beam hand calculation of that result gives the
stress concentration at the peak and the fillet or hole there gives the notch radius, so the notch counts through
Peterson's sensitivity rather than in full. From a vibration result the random input becomes Steinberg's three
bands and a sine sweep a dwell at its peak, both about a zero mean. Every answer gives the life, the safety factor
on stress at the required life, the S-N line with its source and each Marin factor.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..materials import FATIGUE, LIBRARY, fatigue_data, resolve_material
from ..units import fmt, parse_quantity
from ..verify.checks import Check, confidence
from . import fatigue as F
from .outcome import LifeOutcome, verdict_for


@dataclass
class FatigueStudy:
    source: str = ""              # a strength or vibration result folder; empty for stresses given directly
    material: str = ""            # needed for direct stresses, read from the source otherwise
    amplitude: str = ""           # direct: nominal stress amplitude, e.g. '80 MPa'
    mean: str = ""                # direct: nominal mean stress
    kt: float | None = None       # stress concentration at the notch (direct), or overrides the source's
    radius: str = ""              # notch radius, e.g. '1 mm', for the notch sensitivity
    ratio: float = 0.0            # load ratio R = min / max for a strength source (0 on and off, -1 reversed)
    cycles: str = "1e7"           # the life the part needs
    blocks: list[str] = field(default_factory=list)
    repeats: float = 1.0          # how many times the duty cycle of blocks runs
    duration: str = ""            # vibration source: time at the input
    surface: str = "auto"
    size: str = ""                # equivalent diameter of the section, e.g. '12 mm'
    loading: str = "auto"         # bending, axial or torsion
    reliability: float = 0.9
    mean_model: str = "goodman"
    target_sf: float | None = None
    question: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def count(text) -> float:
    """A number of cycles: '1e7', '2,000,000', '5 million'."""
    t = str(text).strip().lower().replace(",", "").replace("_", "")
    scale = 1.0
    for word, k in (("million", 1e6), ("thousand", 1e3), ("k", 1e3), ("m", 1e6)):
        if t.endswith(word):
            t, scale = t[: -len(word)].strip(), k
            break
    try:
        n = float(t) * scale
    except ValueError:
        raise ValueError(f"could not read the number of cycles '{text}'; write it like 1e7 or '2 million'") from None
    if n <= 0:
        raise ValueError("the number of cycles must be positive")
    return n


def cycles_text(n: float) -> str:
    if not math.isfinite(n):
        return "infinite"
    if n >= 1e9:
        e = int(math.floor(math.log10(n)))
        return f"{n / 10 ** e:.3g}e{e}"
    if n >= 1e6:
        return f"{n / 1e6:.3g} million"
    if n >= 100:
        return f"{float(f'{n:.3g}'):,.0f}"
    return f"{n:.2g}"


def run_fatigue(study: FatigueStudy, out_dir: str | Path, progress=None) -> LifeOutcome:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda a, b: None)
    if study.mean_model not in F.MEAN_MODELS:
        raise ValueError(f"unknown mean stress model '{study.mean_model}' (use {', '.join(F.MEAN_MODELS)})")
    src = _source(study)
    mat = src["material"]
    data = fatigue_data(mat)
    if data is None:
        raise ValueError(f"Wobbl has no S-N data for {mat.name}. Pick a material that has it ("
                         f"{', '.join(sorted(FATIGUE))}) or a custom metal near steel or aluminum stiffness")
    say("fatigue", f"building the S-N line for {mat.name}")
    surface = study.surface if study.surface != "auto" else (
        "as-cast" if mat.process == "cast" else "machined" if mat.category == "metal" else "polished")
    loading = study.loading if study.loading != "auto" else src.get("loading", "bending")
    size = parse_quantity(study.size, "length").value if study.size else src.get("size")
    curve = F.build_curve(mat, data, surface, size, loading, study.reliability)
    if study.size == "" and size:
        curve.factors["size kb"] = (curve.factors["size kb"][0], curve.factors["size kb"][1] +
                                    f", taken as 0.808 times the {fmt(src['thickness'])} mm wall")
    radius = parse_quantity(study.radius, "length").value if study.radius else src.get("radius")
    kt = study.kt if study.kt else src.get("kt")
    q, q_why = F.notch_sensitivity(data, mat.tensile_strength, radius) if kt else (1.0, "")
    kf = 1 + q * (kt - 1) if kt else None
    # the stress the part sees at the notch: a finite element peak already holds Kt, so it is scaled by Kf / Kt;
    # nominal stresses given directly are multiplied by Kf
    notch = (kf / kt if src["local"] else kf) if kt else 1.0
    blocks = _blocks(study, src, notch)
    target = study.target_sf or mat.recommended_sf
    n_req = count(study.cycles)
    model = study.mean_model
    D, lives = F.damage(blocks, curve, model)
    repeats = study.repeats if study.blocks else 1.0
    single = len(blocks) == 1 and not study.blocks and not src.get("spectrum")
    if single:
        b = blocks[0]
        life = lives[0]
        need = n_req if not src.get("cycles") else src["cycles"]
        sn = curve.strength(need)
        sf = F.safety_factor(b.amplitude, b.mean, sn, curve, model)
        spread = {m: F.safety_factor(b.amplitude, b.mean, sn, curve, m) for m in F.MEAN_MODELS}
    else:
        total = repeats * D
        life = repeats / total if total > 0 else math.inf
        sf = F.factor_to_failure(blocks, curve, model, repeats)
        spread = {m: F.factor_to_failure(blocks, curve, m, repeats) for m in F.MEAN_MODELS}
        need = repeats
    verdict = verdict_for(sf, target)
    checks = _checks(src, blocks, curve, data, lives, model, spread, target, kt, kf, q, q_why, radius)
    extra = []
    if data.estimated:
        extra.append(("medium", "the S-N line comes from a rule of thumb for this material, not a fatigue test"))
    level, reasons = confidence(checks, extra)
    head = _headline(single, life, need, sf, target, curve, blocks, src, repeats, D)
    if level != "high":
        head += f" Confidence is {level}."
    images = _images(out / "images", curve, blocks, lives, model, need if single else None)
    values = {"analysis": "fatigue", "source": src["kind"], "material": mat.name, "life": life,
              "required": need, "sf": sf, "target_sf": target, "mean_model": model, "damage": D * repeats,
              "blocks": [{"label": b.label, "amplitude_MPa": b.amplitude, "mean_MPa": b.mean, "cycles": b.cycles,
                          "life": n} for b, n in zip(blocks, lives)],
              "curve": {"s1000_MPa": curve.s1000, "se_MPa": curve.se, "ne": curve.ne, "knee": curve.knee,
                        "b": curve.b, "describe": curve.describe()},
              "factors": {k: {"value": v[0], "why": v[1]} for k, v in curve.factors.items()},
              "kt": kt, "kf": kf, "q": q, "notch_radius_mm": radius, "data_source": data.source,
              "data_estimated": data.estimated, "where": src.get("where", ""), "xyz": src.get("xyz"),
              "mean_model_spread": spread,
              "_figures": [["sn", "The S-N line used, with each load block at its life"],
                           ["haigh", "Mean stress diagram for the required life"]],
              "_tables": [["Load blocks", ["block", "amplitude MPa", "mean MPa", "cycles", "life"],
                           [[b.label or "load", fmt(b.amplitude), fmt(b.mean), cycles_text(b.cycles),
                             cycles_text(n)] for b, n in zip(blocks, lives)],
                           "Stresses are at the notch, after the notch sensitivity, and the life is how many such "
                           "cycles alone would start a crack."],
                          ["Marin factors", ["factor", "value", "why"],
                           [[k, f"{v[0]:.3f}", v[1]] for k, v in curve.factors.items()]]]}
    metrics = [("life", f"{cycles_text(life)} {'cycles' if single else 'repeats'}"), ("safety factor", f"{sf:.2f}"),
               ("required", f"{cycles_text(need)} {'cycles' if single else 'repeats'}"), ("target", f"{target:g}")]
    o = LifeOutcome("fatigue", src.get("part", ""), study.question or src.get("question", ""), verdict, head,
                    metrics, level, reasons, checks, _assumptions(mat, data, curve, model, src, surface, study),
                    values, images, _fixes(sf, target, single, blocks, curve, model, kt, radius, src, mat, data,
                                           need, surface), src.get("notes", []),
                    solver="Wobbl stress life", t_total=time.time() - t0)
    try:
        from .playback import fatigue_doc, write

        write(out / "playback.json", fatigue_doc(blocks, lives, repeats, single, need, values))
        o.values["playback"] = "playback.json"
    except Exception:  # noqa: BLE001  the playback file is a convenience, never a reason to fail
        pass
    o.write(out)
    (out / "study.json").write_text(json.dumps(study.to_dict(), indent=1))
    return o


# --------------------------------------------------------------------------------------------- sources
def _source(study: FatigueStudy) -> dict:
    if not study.source:
        if not study.material or not study.amplitude:
            raise ValueError("give a strength or vibration result folder, or --material with --amplitude (and "
                             "--mean) for stresses you already know")
        mat = resolve_material(study.material)
        amp = parse_quantity(study.amplitude, "stress").value
        mean = parse_quantity(study.mean, "stress").value if study.mean else 0.0
        return {"kind": "stresses given", "material": mat, "amplitude": amp, "mean": mean, "local": False,
                "part": "", "where": "at the stresses given"}
    d = Path(study.source)
    if not (d / "outcome.json").exists():
        raise ValueError(f"no Wobbl result in '{study.source}'; pass the folder a strength or vibration run printed")
    o = json.loads((d / "outcome.json").read_text())
    st = json.loads((d / "study.json").read_text()) if (d / "study.json").exists() else {}
    dom = o.get("domain")
    if dom == "strength":
        return _from_strength(o, st, d, study)
    if dom == "vibration":
        return _from_vibration(o, st, d, study)
    raise ValueError(f"fatigue reads strength and vibration results, and '{study.source}' is a {dom} result")


def _material(spec: str, name: str):
    if spec and spec != "auto":
        return resolve_material(spec)
    for m in LIBRARY.values():
        if m.name == name:
            return m
    return resolve_material(name)


def _from_strength(o: dict, st: dict, d: Path, study: FatigueStudy) -> dict:
    mat = _material(st.get("material", ""), o.get("material", ""))
    peak = float(o["governing_MPa"])
    amp, mean = F.cycle_stresses(peak, study.ratio)
    out = {"kind": "strength result", "material": mat, "amplitude": amp, "mean": mean, "local": True, "peak": peak,
           "part": str((d / o["part"]).resolve()) if not Path(o["part"]).is_absolute() else o["part"],
           "where": o.get("governing_where", ""), "xyz": o.get("governing_xyz"), "confidence": o.get("confidence"),
           "confidence_reasons": o.get("confidence_reasons", []), "verdict": o.get("verdict"),
           "question": o.get("question", ""), "measure": o.get("measure", "")}
    beam = o.get("beam") or {}
    ratio = beam.get("ratio")
    if ratio and 1.2 <= ratio <= 6.0:
        out["kt"] = float(ratio)
    if beam.get("area_mm2"):
        axial = abs(beam.get("axial_N", 0)) / beam["area_mm2"]
        out["loading"] = "axial" if axial > 0.5 * beam.get("nominal_peak_MPa", 0) else "bending"
    t = (o.get("features") or {}).get("thickness_typical_mm")
    if t:
        out["thickness"], out["size"] = float(t), 0.808 * float(t)
    out["radius"] = _notch_radius(o)
    out["peaks"] = [h for h in o.get("hotspots", []) if h.get("kind") == "feature" and not h.get("excluded")]
    return out


def _notch_radius(o: dict) -> float | None:
    feats = o.get("features") or {}
    xyz = np.asarray(o.get("governing_xyz") or [0, 0, 0], float)
    spots = [h for h in o.get("hotspots", []) if h.get("kind") == "feature" and h.get("face") is not None]
    if not spots:
        return None
    s = min(spots, key=lambda h: float(np.linalg.norm(np.asarray(h["xyz"]) - xyz)))
    for f in feats.get("fillets", []):
        if f["face"] == s["face"]:
            return float(f["radius"])
    for h in feats.get("holes", []):
        if h["face"] == s["face"]:
            return 0.5 * float(h["diameter"])
    return None


def _from_vibration(o: dict, st: dict, d: Path, study: FatigueStudy) -> dict:
    base = st.get("base") or {}
    mat = _material(base.get("material", ""), "")
    v = o.get("values", {})
    dur = parse_quantity(study.duration, "time").value if study.duration else float(v.get("duration_s") or 0)
    if dur <= 0:
        raise ValueError("say how long the part sees the vibration, for example --duration '2 min'")
    blocks, where, xyz = [], "", None
    if v.get("random"):
        r = v["random"]
        blocks += F.steinberg_blocks(r["sigma_rms_MPa"], r["rate_Hz"], dur)
        where, xyz = "at the random response peak", r["xyz"]
    if v.get("harmonic"):
        h = v["harmonic"]
        blocks.append(F.Block(h["peak_MPa"], 0.0, h["frequency_Hz"] * dur, f"sine dwell at {fmt(h['frequency_Hz'])} Hz"))
        where, xyz = where or "at the sine peak", xyz or h["xyz"]
    if not blocks:
        raise ValueError("this vibration result has no sine or random response to count; run it with --sine or --psd")
    part = o.get("part", "")
    return {"kind": "vibration result", "material": mat, "spectrum": blocks, "local": True,
            "part": str((d / part).resolve()) if part and not Path(part).is_absolute() else part, "where": where,
            "xyz": xyz, "confidence": o.get("confidence"), "confidence_reasons": o.get("confidence_reasons", []),
            "question": o.get("question", ""), "duration": dur}


def _blocks(study: FatigueStudy, src: dict, notch: float) -> list[F.Block]:
    if src.get("spectrum"):
        return [F.Block(b.amplitude * notch, b.mean * notch, b.cycles, b.label) for b in src["spectrum"]]
    amp, mean = src["amplitude"] * notch, src["mean"] * notch
    if study.blocks:
        return [F.parse_block(t, amp, mean) for t in study.blocks]
    return [F.Block(amp, mean, count(study.cycles), "the studied load" if src["local"] else "the stresses given")]


# ---------------------------------------------------------------------------------------------- checks
def _checks(src, blocks, curve, data, lives, model, spread, target, kt, kf, q, q_why, radius) -> list[Check]:
    out = []
    if src.get("confidence"):
        c = src["confidence"]
        why = (src.get("confidence_reasons") or [""])[0].split(". ")[0]
        why = why if len(why) <= 160 else why[:157].rsplit(" ", 1)[0] + "..."
        out.append(Check(src["kind"], {"high": "pass", "medium": "warn", "low": "fail"}.get(c, "info"),
                         f"the stresses come from a result Wobbl rated {c} confidence" + (f" ({why})" if why else "")))
    peak = max(b.amplitude + max(b.mean, 0.0) for b in blocks)
    ny = F.first_cycle_yield(max(b.amplitude for b in blocks), max(b.mean for b in blocks), curve.sy)
    finite = [n for n in lives if math.isfinite(n)]
    short = bool(finite) and min(finite) < 1e3
    if ny >= 1:
        out.append(Check("first cycle yield", "pass", f"the highest stress in a cycle, {fmt(peak)} MPa, stays under "
                                                      f"the {fmt(curve.sy)} MPa yield (factor {ny:.2f}), so the part "
                                                      "stays elastic and the stress life method applies"))
    else:
        out.append(Check("first cycle yield", "fail" if short else "warn",
                         f"the highest stress in a cycle, {fmt(peak)} MPa, is {1 / ny:.2f} times the {fmt(curve.sy)} "
                         "MPa yield, so the notch yields a little every cycle. " +
                         ("Under 1000 cycles this is low cycle fatigue, where stress life does not apply; use strain "
                          "life data or lower the stress" if short else
                          "The S-N line still covers lives past 1000 cycles, though strain life data would be more "
                          "accurate here")))
    if finite and min(finite) < 1e3:
        out.append(Check("S-N range", "warn", f"a block lasts only {cycles_text(min(finite))} cycles, under the 1000 "
                                              "where the stress life line starts, so its life is rough"))
    elif finite and not curve.knee and max(finite) > 10 * curve.ne:
        out.append(Check("S-N range", "info", f"a block lasts {cycles_text(max(finite))} cycles, past the "
                                              f"{cycles_text(curve.ne)} the data reach, so that life extends the line"))
    else:
        out.append(Check("S-N range", "warn" if data.estimated else "pass",
                         ("estimated line: " if data.estimated else "") + f"{curve.describe()} ({data.source})"))
    vals = [v for v in spread.values() if math.isfinite(v)]
    if vals and max(b.mean for b in blocks) > 0:
        flips = len({v >= target for v in vals}) > 1
        s = ", ".join(f"{m.capitalize()} {spread[m]:.2f}" for m in F.MEAN_MODELS)
        out.append(Check("mean stress model", "warn" if flips else "pass",
                         f"safety factor by mean stress model: {s}; the answer uses {model.capitalize()}"
                         + (". The verdict depends on that choice" if flips else ", and the verdict is the same with "
                            "each"), spread))
    if kt:
        out.append(Check("notch sensitivity", "info",
                         f"stress concentration {kt:.2f} at the peak" + (f", notch radius {fmt(radius)} mm" if radius
                                                                          else "") +
                         f": {q_why}, so the fatigue factor is Kf = {kf:.2f}", {"kt": kt, "kf": kf, "q": q}))
    out += _other_peaks(src, blocks, curve, model)
    return out


def _other_peaks(src, blocks, curve, model) -> list[Check]:
    """Every other tracked peak under the same cycling, counted with its full stress (no notch sensitivity, as its
    notch is not known), to make sure none starts a crack sooner than the one judged."""
    peaks = [h for h in src.get("peaks") or [] if h.get("values") and np.isfinite(h["values"][-1])]
    if len(peaks) < 2 or not src.get("peak"):
        return []
    judged = F.damage(blocks, curve, model)[0]
    notch = max(_notch_of(blocks, src), 1e-12)
    worse = []
    for h in peaks:
        s = h["values"][-1] / src["peak"] / notch
        if F.damage([F.Block(b.amplitude * s, b.mean * s, b.cycles) for b in blocks], curve, model)[0] > \
                judged * 1.0001:
            worse.append(h)
    if not worse:
        return [Check("other peaks", "pass", f"{len(peaks)} stress peaks were tracked away from supports and sharp "
                                             "corners, and under the same cycling none starts a crack sooner than the "
                                             "one judged")]
    h = max(worse, key=lambda x: x["values"][-1])
    return [Check("other peaks", "warn", f"the peak of {fmt(h['values'][-1])} MPa "
                                         f"({h.get('detail') or 'elsewhere on the part'}) would start a crack sooner "
                                         "when counted without notch sensitivity; its notch radius decides which one "
                                         "governs", {"xyz": h["xyz"]})]


def _notch_of(blocks, src) -> float:
    """The factor the notch sensitivity put on the judged peak's stresses (one when there was none)."""
    if not blocks or not src.get("amplitude"):
        return 1.0
    return blocks[0].amplitude / src["amplitude"] if src["amplitude"] > 0 else 1.0


def _headline(single, life, need, sf, target, curve, blocks, src, repeats, D) -> str:
    where = src.get("where", "")
    at = f" {where}" if where else ""
    if single:
        b = blocks[0]
        swing = (f"a stress swinging {fmt(b.amplitude)} MPa either side of {fmt(b.mean)} MPa" if b.mean else
                 f"a fully reversed stress of {fmt(b.amplitude)} MPa")
        if not math.isfinite(life):
            return (f"Below the endurance limit{at}: {swing} never starts a crack, safety factor {sf:.2f} on stress "
                    f"(target {target:g}).")
        lead = "Lasts" if life >= need else "Cracks start after"
        return (f"{lead} about {cycles_text(life)} cycles{at}, with {swing}. For the required "
                f"{cycles_text(need)} cycles the safety factor on stress is {sf:.2f} (target {target:g}).")
    if D <= 0:
        return (f"No block reaches the endurance limit{at}, so the duty cycle does no fatigue damage; the stresses "
                f"could rise {sf:.2f} times before it would (target {target:g}).")
    run = "test" if src.get("duration") else "duty cycle"
    lead = f"The {run} uses" if repeats == 1 else f"{fmt(repeats)} runs of the {run} use"
    return (f"{lead} {100 * D * repeats:.3g}% of the fatigue life{at} (Miner's rule), so it can run about "
            f"{cycles_text(life)} times before cracks start; the stresses could rise {sf:.2f} times before the "
            f"required {cycles_text(need)} runs would start a crack (target {target:g}).")


def _assumptions(mat, data, curve, model, src, surface, study) -> list[str]:
    a = ["Stress life method for crack initiation (Shigley's approach): an S-N line from f Sut at 1000 cycles to the "
         "corrected endurance strength. It does not cover crack growth after a crack starts, which is a separate "
         "damage tolerance question.",
         f"S-N data for {mat.name}: {fmt(data.strength)} MPa at {cycles_text(data.cycles)} cycles, smooth specimens, "
         f"fully reversed ({data.source}).",
         f"Marin factors: surface {surface} ({curve.factors['surface ka'][0]:.2f}), size "
         f"{curve.factors['size kb'][0]:.2f}, load {curve.factors['load kc'][0]:.2f}, reliability "
         f"{100 * study.reliability:g}% ({curve.factors['reliability ke'][0]:.3f}), room temperature.",
         f"Mean stress by {model.capitalize()}; a compressive mean gets no credit.",
         "Fatigue lives scatter by a factor of three to ten at the same stress, so the life is an estimate and the "
         "safety factor on stress is the steadier number."]
    if src["kind"] == "strength result" and study.blocks:
        a.append("Each block scales the studied load and cycles it between R times its peak and the peak (or at the "
                 "study's load ratio when the block gives none), and the stress field keeps its shape, as it does "
                 "for a linear part.")
    elif src["kind"] == "strength result":
        a.append(f"The peak stress cycles with the load, between {study.ratio:g} times the studied load and the load "
                 "itself, and the stress field keeps its shape, as it does for a linear part.")
    if src["kind"] == "vibration result":
        a.append("Random stress follows Steinberg's three bands (68.3% of cycles at 1 sigma, 27.1% at 2 and 4.33% at "
                 "3), at the zero crossing rate of the response, about a zero mean.")
    if mat.category != "metal":
        a.append("Plastics also heat up and soften when they cycle fast, which this does not model.")
    return a


def _fixes(sf, target, single, blocks, curve, model, kt, radius, src, mat, data, need, surface) -> list[dict]:
    if sf >= target:
        return []
    out = []
    want = target / sf
    if kt and radius:
        r2 = max(2 * radius, radius + 1.0)
        kt2 = 1 + (kt - 1) * math.sqrt(radius / r2)
        q2, _ = F.notch_sensitivity(data, mat.tensile_strength, r2)
        q1, _ = F.notch_sensitivity(data, mat.tensile_strength, radius)
        kf1, kf2 = 1 + q1 * (kt - 1), 1 + q2 * (kt2 - 1)
        gain = kf1 / kf2
        out.append({"kind": "fillet", "title": f"Open the notch radius from {fmt(radius)} to {fmt(r2)} mm",
                    "summary": f"The stress concentration at a notch falls about as one over the square root of its "
                               f"radius, from {kt:.2f} to about {kt2:.2f}, and the fatigue factor from {kf1:.2f} to "
                               f"{kf2:.2f}. That cuts the notch stress by {100 * (1 - 1 / gain):.0f}%, worth a factor "
                               f"of {gain:.2f} on the safety factor.", "params": {"radius": r2}, "auto": False})
    if mat.category == "metal" and surface not in ("polished", "ground"):
        ka_now = curve.factors["surface ka"][0]
        ka_g = F.surface_factor("ground", mat.tensile_strength, True)[0]
        if ka_g > ka_now * 1.03:
            out.append({"kind": "surface", "title": "Grind or polish the surface at the peak",
                        "summary": f"The surface factor goes from {ka_now:.2f} ({surface}) to {ka_g:.2f} ground, which "
                                   f"raises the endurance strength by {100 * (ka_g / ka_now - 1):.0f}%. Shot peening "
                                   "adds a compressive skin that helps more, though Wobbl does not count it.",
                        "auto": False})
    out.append({"kind": "thicken", "title": f"Cut the cyclic stress by {100 * (1 - 1 / want):.0f}%",
                "summary": f"A safety factor of {target:g} needs every stress {want:.2f} times lower. In bending the "
                           f"stress falls with the square of the wall, so about {math.sqrt(want):.2f} times the "
                           "thickness at the peak gets there.", "params": {"factor": want}, "auto": False})
    better = []
    for key, fd in FATIGUE.items():
        m = LIBRARY[key]
        if key == mat.key or m.category != mat.category or fd.strength <= data.strength * want:
            continue
        better.append((fd.strength / max(data.strength, 1e-9), m.name, m.process))
    if better:
        better.sort()
        ratio, name, process = better[0]
        made = f" It is a {process} alloy, so the part would be made that way instead." \
            if process and process != mat.process else ""
        out.append({"kind": "material", "title": f"Switch to {name}",
                    "summary": f"Its fatigue strength is {ratio:.2g} times that of {mat.name}, which covers the "
                               f"{want:.2f} needed if the part is otherwise the same.{made}", "auto": False})
    return out


def _images(img: Path, curve, blocks, lives, model, need) -> dict:
    from .render import haigh_plot, sn_plot

    out = {}
    try:
        pts = [(n, F.equivalent_amplitude(b.amplitude, b.mean, curve, model), b.label or "load")
               for b, n in zip(blocks, lives)]
        out["sn"] = sn_plot(img / "sn.png", curve, pts)
        if any(b.mean > 0 for b in blocks):
            sn = curve.strength(need) if need else curve.se
            out["haigh"] = haigh_plot(img / "haigh.png", curve, sn, [(b.mean, b.amplitude, b.label or "load")
                                                                      for b in blocks], model)
    except Exception as e:  # noqa: BLE001  pictures are a convenience, never a reason to fail
        out["error"] = str(e)[:200]
    return out
