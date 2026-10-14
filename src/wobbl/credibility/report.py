"""The credibility evidence in a run's report.html and summary.txt, and the figure that goes with it.

Each piece of evidence (uncertainty.json, validation.json, credibility.json, test-plan.json) is a file next to the
run. The report writers ask this module for sections, so a report rebuilt later keeps whatever evidence exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..report.html import badge, esc, figure, table
from ..units import fmt
from . import uq


def _read(d: Path, name: str) -> dict | None:
    p = Path(d) / name
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def refresh(result_dir):
    """Rebuild report.html and summary.txt of a run so they show its evidence files."""
    d = Path(result_dir)
    data = json.loads((d / "outcome.json").read_text())
    if data.get("domain") == "strength":
        from ..structural.outcome import Outcome
        from ..structural.report import write_outputs

        write_outputs(Outcome.load(d / "outcome.json"), d)
    else:
        from ..outcome_generic import GenericOutcome

        GenericOutcome.load(d / "outcome.json").write(d)


def evidence_sections(result_dir) -> list[str]:
    d = Path(result_dir)
    out = []
    u = _read(d, "uncertainty.json")
    if u:
        out.append(uncertainty_section(u, d))
    for name, build in (("validation.json", "validation_section"), ("credibility.json", "credibility_section"),
                        ("test-plan.json", "test_plan_section")):
        r = _read(d, name)
        fn = globals().get(build)
        if r and fn:
            out.append(fn(r, d))
    return out


def evidence_text(result_dir) -> str:
    """Lines for summary.txt, the file agents read first."""
    d = Path(result_dir)
    lines = []
    u = _read(d, "uncertainty.json")
    if u:
        lines += ["", "Uncertainty: " + u.get("headline", "")]
    for name, key in (("validation.json", "Tests"), ("credibility.json", "Credibility"),
                      ("test-plan.json", "Test plan")):
        r = _read(d, name)
        if r and r.get("headline"):
            lines += ["", f"{key}: {r['headline']}"]
    return "\n".join(lines)


def uncertainty_figure(path: Path, r: dict) -> str:
    """The decision output's distribution with its limit, next to the total Sobol index of each input."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..report.render import BG, FG, MARK

    h = r["histogram"]
    edges, counts = h["edges"], h["counts"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=110, facecolor=BG,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3d42")
    a1.stairs(counts, edges, fill=True, color=MARK, alpha=0.85)
    dec = r["decision"]
    unit = r["outputs"][dec]["unit"]
    a1.set_xlabel({"safety_factor": "safety factor", "t_max_C": "hottest point, C"}.get(dec, f"{dec} {unit}"),
                  color=FG, fontsize=9)
    a1.set_yticks([])
    if dec == "safety_factor":
        a1.axvline(1.0, color="#ff4d4f", ls="--", lw=1.2, label="fails")
        tgt = r.get("target")
        if tgt:
            a1.axvline(tgt, color="#3ccf7a", ls=":", lw=1.2, label="target")
        a1.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8)
    idx = r["sensitivity"]["indices"][:6][::-1]
    names = [x["input"] for x in idx]
    a2.barh(names, [max(x["total"], 0) for x in idx], color="#4a90d9", label="total")
    a2.barh(names, [max(x["first_order"], 0) for x in idx], color=MARK, height=0.4, label="first order")
    a2.set_xlim(0, 1)
    a2.set_xlabel("Sobol index", color=FG, fontsize=9)
    a2.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=7, loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def uncertainty_section(r: dict, d: Path) -> str:
    img = d / "images" / "uncertainty.png"
    try:
        uncertainty_figure(img, {**r, "target": _target(d)})
    except Exception:  # noqa: BLE001  the figure is a convenience, the numbers are in the tables
        img = None
    parts = ["<h2>Uncertainty</h2>", f"<p>{esc(r.get('headline', ''))}</p>"]
    if img:
        parts.append(figure(img, "Distribution of the answer across the input scatter, and how much of its "
                                 "variance each input accounts for"))
    rows = [[i["name"], i.get("label") or "", i.get("role") or "", _dist(i)] for i in r["inputs"]]
    parts.append("<h3>Inputs</h3>" + table(["input", "what it is", "kind", "distribution"], rows))
    rows = []
    for name, s in r["outputs"].items():
        u = s.get("unit") or ""
        rows.append([name.replace("_", " "), f"{fmt(s['mean'])} {u}", f"{fmt(s['sd'])} {u}",
                     f"{fmt(s['p05'])} to {fmt(s['p95'])} {u}", f"{fmt(s['p01'])} to {fmt(s['p99'])} {u}"])
    parts.append("<h3>Answer</h3>" + table(["output", "mean", "standard deviation", "90% of samples",
                                            "98% of samples"], rows))
    rows = []
    for k, p in r["probability"].items():
        rel = p.get("reliability_index")
        rows.append([k, p.get("against", ""), uq.odds(p), "" if rel is None else f"{rel:.2f}"])
    for a in r.get("alternatives", []):
        p = a["probability"]
        rel = p.get("reliability_index")
        rows.append([f"failure with {a['label']}", "", uq.odds(p), "" if rel is None else f"{rel:.2f}"])
    if rows:
        parts.append("<h3>Probability</h3>" + table(["event", "against", "probability", "reliability index"], rows))
    rows = [[x["input"], f"{x['first_order']:.3f}", f"{x['first_order_ci'][0]:.3f} to {x['first_order_ci'][1]:.3f}",
             f"{x['total']:.3f}", f"{x['total_ci'][0]:.3f} to {x['total_ci'][1]:.3f}"]
            for x in r["sensitivity"]["indices"]]
    parts.append("<h3>What matters most</h3>" + table(["input", "first order", "95% interval", "total",
                                                       "95% interval"], rows, {1, 2, 3, 4}))
    m = r.get("model", {})
    note = (f"{r['samples']:,} {r['sampling']} samples. The model is {m.get('method', '')}. Sobol indices use "
            f"{r['sensitivity']['base_samples']:,} base points of a scrambled Sobol sequence, with bootstrap "
            "intervals.")
    parts.append(f"<p class='muted'>{esc(note)}</p>")
    return "".join(parts)


def _reading(v: float, quantity: str) -> str:
    if quantity == "strain":
        return f"{fmt(v * 1e6)} µε"
    return f"{fmt(v)} {'mm' if quantity == 'displacement' else 'C' if quantity == 'temperature' else ''}".strip()


def validation_figure(path: Path, r: dict) -> str:
    """Each reading against the simulation, with twice the validation uncertainty as error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..report.render import BG, FG, MARK

    rows = r["readings"]
    by = {}
    for x in rows:
        by.setdefault(x["quantity"], []).append(x)
    fig, axes = plt.subplots(1, len(by), figsize=(4.2 * len(by), 3.6), dpi=110, facecolor=BG, squeeze=False)
    for ax, (q, xs) in zip(axes[0], by.items()):
        k = 1e6 if q == "strain" else 1.0
        d = np.array([x["value"] for x in xs]) * k
        s = np.array([x["simulation"] for x in xs]) * k
        e = 2 * np.array([x["u_val"] for x in xs]) * k
        ax.set_facecolor(BG)
        lo, hi = float(min(d.min(), s.min())), float(max(d.max(), s.max()))
        pad = 0.08 * (hi - lo or abs(hi) or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#3a3d42", lw=1)
        ax.errorbar(d, s, yerr=e, fmt="o", color=MARK, ecolor="#4a90d9", ms=4, capsize=2)
        unit = {"strain": "µε", "displacement": "mm", "temperature": "C"}.get(q, "")
        ax.set_xlabel(f"reading, {q} {unit}".strip(), color=FG, fontsize=9)
        ax.set_ylabel(f"simulation {unit}".strip(), color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3d42")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return str(path)


def validation_section(r: dict, d: Path) -> str:
    img = d / "images" / "validation.png"
    try:
        validation_figure(img, r)
    except Exception:  # noqa: BLE001
        img = None
    t = r["tests"]
    parts = ["<h2>Comparison with tests</h2>"]
    if t.get("synthetic"):
        parts.append("<p><strong>These readings are synthetic, made for a demonstration, and are not from a real "
                     "test.</strong></p>")
    parts.append(f"<p>{esc(r.get('headline', ''))}</p>")
    if img:
        parts.append(figure(img, "Each reading against the simulation, with bars at twice the validation "
                                 "uncertainty. Points on the line agree."))
    rows = []
    for x in r["readings"]:
        cond = (f"{fmt(x['load'])}" if x.get("load") is not None else "") + \
            (f" at {fmt(x['time'])} s" if x.get("time") is not None else "")
        rows.append([x["name"], x["quantity"], cond.strip(), f"{_reading(x['value'], x['quantity'])} ± "
                     f"{_reading(x['u'], x['quantity'])}", _reading(x["simulation"], x["quantity"]),
                     _reading(x["error"], x["quantity"]), _reading(x["u_val"], x["quantity"]),
                     f"{x['ratio']:.2f}" if x.get("ratio") is not None else "",
                     badge("pass" if x["within"] else "warn")])
    parts.append("<h3>ASME V&amp;V 20 comparison</h3>" + table(
        ["gauge", "quantity", "load", "reading", "simulation", "E = S - D", "u_val", "|E| / u_val", ""], rows))
    rows = [[a["name"], f"{fmt(a['load'])}" if a.get("load") is not None else "", str(a["readings"]),
             _reading(a["area"], a["quantity"]), f"{100 * a['relative']:.2g}%" if a.get("relative") else ""]
            for a in r["area_metric"]]
    p = r["u_pooled"]
    parts.append("<h3>Area metric</h3>" + table(["gauge", "load", "readings", "area", "of the reading"], rows) +
                 f"<p class='muted'>Pooled over all readings the area is {p['area']:.3f} against "
                 f"{p['reference_95']:.3f} for a model whose spread is right, at 95%.</p>")
    mf = r.get("model_form")
    if mf:
        rows = [[x["idealization"] + (" (assumed)" if x["idealization"] == mf["assumed"] else ""),
                 f"{fmt(x['fitted_modulus'])} MPa" if x.get("fitted_modulus") else "", f"{x['rms_in_sigma']:.2f}",
                 f"{100 * x['rms_relative_error']:.2g}%"] for x in mf["idealizations"]]
        parts.append("<h3>Which idealization the readings favour</h3>" +
                     table(["load attachment", "modulus fitted to the readings", "rms residual in reading "
                            "uncertainties", "rms error"], rows) +
                     f"<p class='muted'>The readings favour {esc(mf['favours'])}, log10 likelihood ratio "
                     f"{mf['log10_likelihood_ratio']:.1f}.</p>")
    cal = r.get("calibration")
    if cal:
        rows = []
        for c in cal["inputs"]:
            u = f" {c['unit']}" if c.get("unit") and c["unit"] != "x" else ""
            rows.append([c["input"], f"{fmt(c['prior_mean'])} ± {fmt(c['prior_sd'])}{u}",
                         f"{fmt(c['mean'])} ± {fmt(c['sd'])}{u}", f"{fmt(c['ci95'][0])} to {fmt(c['ci95'][1])}{u}",
                         f"{c['narrowed']:.2g}x"])
        a, b = cal["residual_before"], cal["residual_after"]
        parts.append("<h3>Calibration</h3>" + table(["input", "before the test", "after", "95% interval",
                                                     "narrower by"], rows) +
                     f"<p class='muted'>Residuals fall from {a['rms_in_sigma']:.2f} to {b['rms_in_sigma']:.2f} "
                     f"reading uncertainties (root mean square). Adaptive Metropolis, {cal['steps']:,} steps, "
                     f"acceptance {100 * cal['acceptance']:.0f}%, about {cal['effective_samples']:,.0f} independent "
                     "draws.</p>")
    if r.get("notes"):
        parts.append("<ul class='plain'>" + "".join(f"<li>{esc(n)}</li>" for n in r["notes"]) + "</ul>")
    return "".join(parts)


def credibility_section(r: dict, d: Path) -> str:
    parts = ["<h2>Credibility</h2>", f"<p>{esc(r['headline'])}</p>",
             f"<p class='muted'>Context of use. {esc(r['context_of_use'])}</p>"]
    parts.append(table(["question", "model influence", "decision consequence", "model risk", "proposed goal"],
                       [[r.get("question") or "", r["influence"], r["consequence"], f"{r['risk']} of 5", r["goal"]]]))
    rows = [[f["group"], f["factor"], ("html", f'<span class="badge {esc(f["level"])}">{esc(f["level"])}</span>'),
             f["goal"], badge("pass" if f["meets"] else "warn"), f["evidence"] + (f". To close it, {f['gap']}."
                                                                                  if f["gap"] else "")]
            for f in r["factors"]]
    parts.append("<h3>Factors</h3>" + table(["group", "factor", "level", "goal", "", "evidence"], rows))
    parts.append(f"<p class='muted'>{esc(r['disclaimer'])}</p>")
    return "".join(parts)


def test_plan_section(r: dict, d: Path) -> str:
    parts = ["<h2>Test plan</h2>", f"<p>{esc(r['headline'])}</p>"]
    for t in r["tests"]:
        body = f"<div>{esc(t['what'].rstrip('.'))}.</div><div class='muted'>{esc(t['why'])}</div>"
        if t.get("how_many"):
            body += f"<div class='muted'>{esc(t['how_many'])}</div>"
        rows = []
        for g in t.get("gauges", []):
            for p in g.get("predicted", []):
                val = (f"{fmt(p['microstrain'])} ± {fmt(p['band_95'])} µε" if "microstrain" in p else
                       f"{fmt(p['mm'])} ± {fmt(p['band_95'])} mm" if "mm" in p else f"{fmt(p.get('C', 0))} C")
                rows.append([g["kind"], ", ".join(fmt(x) for x in g["xyz"]),
                             ", ".join(f"{x:+.2f}" for x in g.get("direction", [])),
                             fmt(p["load"]) if "load" in p else "", val])
        if rows:
            body += table(["gauge", "x, y, z mm", "axis", "load", "predicted, 95% band"], rows)
        notes = [g["note"] for g in t.get("gauges", []) if g.get("note")]
        body += "".join(f"<div class='muted'>{esc(n)}</div>" for n in notes)
        parts.append(f"<div class='fix'><div class='t'>{t['order']}. {esc(t['kind'])}</div>{body}</div>")
    rows = [[x["test"], badge({"likely": "pass", "not yet": "warn", "cannot": "fail"}[x["verdict"]]), x["verdict"],
             x["why"]] for x in r["replace"]]
    parts.append("<h3>Which tests the simulation can stand in for</h3>" + table(["test", "", "verdict", "why"], rows))
    parts.append(f"<p class='muted'>The plan keeps {esc(r['rule'])}.</p>")
    return "".join(parts)


def credibility_page(result_dir) -> str:
    """credibility.html, a self contained page an engineer can hand to a reviewer: the question, the answer and
    its checks, the uncertainty, the comparison with tests, the factor by factor assessment and the test plan."""
    from ..report.html import page

    d = Path(result_dir)
    o = _read(d, "outcome.json") or {}
    c = _read(d, "credibility.json")
    if not c:
        return ""
    met, n = c["met"], len(c["factors"])
    sections = [f"<h2>Answer</h2><p>{esc(o.get('headline', ''))}</p>" +
                table(["status", "check", "result"], [[badge(k["status"]), k["name"], k["summary"]]
                                                      for k in o.get("checks", [])])]
    sections += evidence_sections(d)
    sections.append("<h2>Assumptions</h2><ol class='assume'>" +
                    "".join(f"<li>{esc(a)}</li>" for a in o.get("assumptions", [])) + "</ol>")
    u = _read(d, "uncertainty.json")
    pf = (u or {}).get("probability", {}).get("failure")
    metrics = [("model risk", f"{c['risk']} of 5"), ("proposed goal", c["goal"]), ("factors met", f"{met} of {n}"),
               ("failure", f"{pf['value']:.2g}" if pf and pf["failures"] else
                (f"< {pf['ci95'][1]:.2g}" if pf else "not computed"))]
    html = page("Credibility evidence", Path(o.get("part", "")).name or o.get("domain", ""), c.get("question", ""),
                "MEETS" if met == n else "SHORT", "pass" if met == n else "warn", c["headline"], metrics,
                o.get("confidence", ""), o.get("confidence_reasons", []), sections, c["disclaimer"])
    path = d / "credibility.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def _dist(i: dict) -> str:
    u = f" {i['unit']}" if i.get("unit") and i["unit"] != "x" else ""
    if i["dist"] == "uniform":
        return f"uniform from {fmt(i['low'])} to {fmt(i['high'])}{u}"
    cov = f", {100 * i['sd'] / abs(i['mean']):.3g}%" if i.get("mean") else ""
    return f"{i['dist']}, mean {fmt(i['mean'])}{u}, sd {fmt(i['sd'])}{u}{cov}"


def _target(d: Path):
    o = _read(d, "outcome.json") or {}
    return o.get("target_sf")


def status_badge(s: str):
    return badge(s)
