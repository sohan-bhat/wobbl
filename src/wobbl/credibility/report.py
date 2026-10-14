"""The credibility evidence in a run's report.html and summary.txt, and the figure that goes with it.

Each piece of evidence (uncertainty.json, validation.json, credibility.json, test-plan.json) is a file next to the
run. The report writers ask this module for sections, so a report rebuilt later keeps whatever evidence exists.
"""
from __future__ import annotations

import json
from pathlib import Path

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
            f"{r['sensitivity']['base_samples']:,} base points of a scrambled Sobol sequence, with bootstrap intervals.")
    parts.append(f"<p class='muted'>{esc(note)}</p>")
    return "".join(parts)


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
