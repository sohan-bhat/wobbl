"""Strength study report: the HTML page and the files saved next to it."""
from __future__ import annotations

import threading
from pathlib import Path

from ..report.html import badge, esc, figure, page, table
from ..units import fmt, fmt_length, fmt_stress
from .outcome import Outcome, text_summary

VERDICT_WORD = {"holds": "HOLDS", "marginal": "MARGINAL", "fails": "FAILS"}


def write_report(o: Outcome, out_dir: str | Path) -> str:
    out = Path(out_dir)
    imp = o.imperial
    img = o.images
    sections = []
    figs = []
    if img.get("stress"):
        sections.append('<h2>Stress</h2><div class="hero">' + figure(
            img["stress"], f"{o.measure} with the governing point marked "
                           f"({fmt_stress(o.governing_MPa, imp)}). Color scale tops out at the reported value.")
                        + "</div>")
    for key, cap in (("zoom", "Close up of the governing point"),
                     ("safety", "Safety factor (red is below 1, green is comfortable)"),
                     ("deformed", "Deflected shape, exaggerated so it is visible")):
        if img.get(key):
            figs.append(figure(img[key], cap))
    if figs:
        sections.append('<div class="grid">' + "".join(figs) + "</div>")

    rows = [[badge(c.status), c.name, c.summary] for c in o.checks]
    sections.append("<h2>Checks</h2>" + table(["status", "check", "result"], rows))

    if o.fixes:
        fx = []
        for f in o.fixes:
            extra = ""
            if f.get("after"):
                a = f["after"]
                extra = (f"<div class='muted mono'>re-run: peak {fmt(a.get('governing_MPa', 0))} MPa, safety factor "
                         f"{a.get('safety_factor', 0):.2f}, {a.get('verdict', '')}</div>")
            fx.append(f"<div class='fix'><div class='t'>{esc(f.get('title', ''))}</div>"
                      f"<div>{esc(f.get('summary', ''))}</div>{extra}</div>")
        if img.get("fix_before") and img.get("fix_after"):
            fx.append('<div class="grid">' + figure(img["fix_before"], "before") +
                      figure(img["fix_after"], "after the fix") + "</div>")
        sections.append("<h2>How to fix it</h2>" + "".join(fx))

    lv_rows = []
    prev = None
    for L in o.levels:
        ch = "" if prev is None else f"{100 * (L['governing'] - prev) / prev:+.1f}%"
        lv_rows.append([L["level"], f"{L['n_nodes']:,}", fmt(L["h_global"]), fmt(L["h_min"]), fmt(L["governing"]),
                        ch, fmt(L["max_disp"]), f"{L['t_solve']:.1f} s"])
        prev = L["governing"]
    conv = "<h2>Mesh convergence</h2>"
    if img.get("convergence"):
        conv += figure(img["convergence"], "Peak stress at each refinement level")
    conv += table(["level", "nodes", "global size mm", "finest mm", "governing MPa", "change", "max deflection mm",
                   "solve"], lv_rows, {1, 2, 3, 4, 5, 6, 7})
    sections.append(conv)

    hs_rows = []
    for h in o.hotspots:
        vals = " → ".join(fmt(v) for v in h["values"] if v == v)
        status = "excluded" if h["excluded"] else "counted"
        hs_rows.append([h["id"], h["detail"] or h["kind"], ", ".join(fmt(v) for v in h["xyz"]), vals, status])
    sections.append("<h2>Stress peaks</h2><p class='muted'>Every peak Wobbl found, with its value at each mesh. "
                    "Peaks at sharp corners and idealized supports are excluded from the answer because their "
                    "value depends on the idealization, not the part.</p>"
                    + table(["#", "where", "x, y, z (mm)", "MPa by level", "status"], hs_rows))

    setup = [["material", o.material], ["limit", fmt_stress(o.allowable_MPa, imp)],
             ["target safety factor", f"{o.target_sf:g}"], ["mass", f"{o.mass_kg * 1000:.4g} g"]]
    setup += [["load", x] for x in o.loads] + [["fixture", x] for x in o.fixtures]
    feats = o.features
    setup.append(["part", f"{len(feats.get('holes', []))} holes, {len(feats.get('fillets', []))} inside fillets, "
                          f"{len(feats.get('sharp_inside_corners', []))} sharp inside corners, walls about "
                          f"{fmt(feats.get('thickness_typical_mm', 0))} mm"])
    from ..credibility.report import evidence_sections

    sections += evidence_sections(out)
    sections.append("<h2>Setup</h2>" + table(["", ""], setup))
    sections.append("<h2>Assumptions</h2><ol class='assume'>" +
                    "".join(f"<li>{esc(a)}</li>" for a in o.assumptions) + "</ol>")
    if o.notes:
        sections.append("<h2>Notes</h2><ul class='plain'>" + "".join(f"<li>{esc(n)}</li>" for n in o.notes) + "</ul>")
    metrics = [("peak stress", fmt_stress(o.governing_MPa, imp)), ("safety factor", f"{o.safety_factor:.2f}"),
               ("max deflection", fmt_length(o.max_disp_mm, imp)), ("target", f"{o.target_sf:g}")]
    html_text = page("Strength report", Path(o.part).name, o.question, VERDICT_WORD.get(o.verdict, o.verdict.upper()),
                     o.verdict, o.headline, metrics, o.confidence, o.confidence_reasons, sections,
                     f"Solver {o.solver}. Total time {o.t_total:.0f} s.")
    path = out / "report.html"
    path.write_text(html_text, encoding="utf-8")
    from ..viz.embed import after_report
    after_report(out)
    return str(path)


_WRITING = threading.Lock()     # the pictures can finish and rewrite the outputs while the run writes them


def write_outputs(o: Outcome, out_dir: str | Path) -> Outcome:
    out = Path(out_dir)
    from ..credibility.report import evidence_text

    with _WRITING:
        o.report = write_report(o, out)
        (out / "summary.txt").write_text(text_summary(o) + evidence_text(out) + "\n", encoding="utf-8")
        o.save(out / "outcome.json")
    return o
