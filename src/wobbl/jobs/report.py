"""The live report of a job: state, progress and ETA, the peaks found so far and the newest lines of the log.

The watcher rewrites it every few seconds while the job runs, and the page reloads itself until the job ends,
so results show up as the solver writes them. It is one self contained file like every Wobbl report.
"""
from __future__ import annotations

import os
import time

from ..report.html import CSS, esc, table
from .store import ACTIVE, Job

WORD = {"starting": ("STARTING", "warn"), "running": ("RUNNING", "warn"), "done": ("DONE", "pass"),
        "failed": ("FAILED", "fail"), "cancelled": ("CANCELLED", "fail"), "interrupted": ("INTERRUPTED", "fail")}


def sparkline(points: list, width: int = 520, height: int = 90) -> str:
    """A small inline chart of a peak against time."""
    pts = [(float(t), float(v)) for t, v in points if v is not None]
    if len(pts) < 2:
        return ""
    t0, t1 = min(p[0] for p in pts), max(p[0] for p in pts)
    v0, v1 = min(p[1] for p in pts), max(p[1] for p in pts)
    sx = (width - 8) / ((t1 - t0) or 1.0)
    sy = (height - 8) / ((v1 - v0) or 1.0)
    path = " ".join(f"{4 + (t - t0) * sx:.1f},{height - 4 - (v - v0) * sy:.1f}" for t, v in pts)
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
            f'aria-label="peak against time"><polyline fill="none" stroke="#ffd400" stroke-width="2" '
            f'points="{path}"/></svg>')


def write_report(job: Job, st: dict) -> str:
    from ..estimate import duration

    state = st.get("state", "starting")
    word, cls = WORD.get(state, (state.upper(), "warn"))
    frac = st.get("progress")
    metrics = [("progress", f"{100 * frac:.0f}%" if frac is not None else "starting"),
               ("elapsed", duration(st.get("elapsed_s") or 0.0))]
    if st.get("eta_s") is not None and state in ACTIVE:
        metrics.append(("time left", duration(st["eta_s"])))
    if job.estimate.get("time"):
        metrics.append(("predicted", job.estimate["time"]))
    if st.get("peak_memory_gb"):
        metrics.append(("peak memory", f"{st['peak_memory_gb']:.2f} GB"))
    met = "".join(f'<div class="metric"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                  for k, v in metrics)
    bar = (f'<div style="background:#22252a;border-radius:6px;height:10px;margin-top:14px"><div style="width:'
           f'{100 * (frac or 0):.1f}%;background:var(--yellow);height:10px;border-radius:6px"></div></div>')
    sections = []
    res = st.get("results") or {}
    if res.get("peaks"):
        rows = [[p["quantity"], f"{p['max']:.5g}" + (f" (lowest {p['min']:.5g})" if "min" in p else ""),
                 f"{p['time']:g}", str(p.get("step", "")),
                 ", ".join(f"{v:.4g}" for v in p["xyz"]) if p.get("xyz") else f"node {p.get('node')}"]
                for p in res["peaks"]]
        sections.append("<h2>Results so far</h2>" + table(["quantity", "largest", "at time", "step", "where"], rows,
                                                         {1, 2}))
        for q, h in (res.get("history") or {}).items():
            if len(h) > 2:
                sections.append(f"<h2>{esc(q)} against time</h2>" + sparkline(h))
    elif res.get("headline"):
        sections.append(f"<h2>Answer</h2><p>{esc(res['headline'])}</p>" +
                        (f"<p class='muted'>Full report {esc(res.get('report', ''))}</p>" if res.get("report") else ""))
    if job.estimate.get("basis"):
        sections.append("<h2>How the prediction was made</h2><ul class='plain'>" +
                        "".join(f"<li>{esc(b)}</li>" for b in job.estimate["basis"]) + "</ul>")
    log = job.folder / "solver.log"
    if log.exists():
        tail = log.read_text(errors="replace").splitlines()[-25:]
        sections.append("<h2>Newest log lines</h2><pre>" + esc("\n".join(tail)) + "</pre>")
    refresh = '<meta http-equiv="refresh" content="10">' if state in ACTIVE else ""
    where = "this machine" if job.runner == "local" else job.runner
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Job {esc(job.id)}</title><style>{CSS}</style></head>
<body><main>
<header><div><h1><span class="brand">WOBBL</span>{esc(job.id)}</h1><p class="question">{esc(job.input)}</p></div>
<div class="meta">Job on {esc(where)}<br>{time.strftime('%Y-%m-%d %H:%M')}</div></header>
<section class="verdict">
 <div class="card"><div class="word {cls}">{word}</div><p class="headline">{esc(st.get('detail', ''))}</p>{bar}</div>
 <div class="card"><div class="metrics">{met}</div></div>
</section>
{''.join(sections)}
<footer>Job {esc(job.id)}, {job.threads} thread{'s' if job.threads != 1 else ''}, run {job.restarts + 1}.</footer>
</main></body></html>
"""
    path = job.folder / "report.html"
    tmp = path.with_name(f"report.{os.getpid()}.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)
    return str(path)
