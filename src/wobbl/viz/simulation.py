"""The page a `wobbl simulate` run opens on: the question, the setting, and each analysis played in 3D.

simulate writes simulation.json with the plan, the overall verdict, the situation (supports, loads, environment
and the objects the question names) and one entry per analysis with its result folder. Each step's scene is the
one its domain wrote, or one built from its files, and already carries the props its own supports and loads call
for. This page puts them in order under the question, says what the setting is, and links each full report.
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
from pathlib import Path

from .embed import attach, font_css, viewer_js
from .scene import FILE, load

TONE = {"pass": ("#3ccf7a", "PASSES"), "marginal": ("#ffb020", "MARGINAL"), "fail": ("#ff4d4f", "FAILS"),
        "unknown": ("#a3a7ad", "DONE")}


def title_of(thing: str) -> str:
    """A readable name for what was simulated, from a file path or the sentence the run was asked in."""
    t = str(thing or "").strip()
    if not t:
        return "Simulation"
    if " " not in t:
        p = Path(t)
        stem = p.stem
        if p.parent.name and stem.lower() in ("part", "model", "main", "bracket", "design", "rocket", "circuit"):
            stem = p.parent.name
        t = stem.replace("-", " ").replace("_", " ")
    else:
        t = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|kg|g|lb)\s+(?=[a-z])", "", t)
        t = re.sub(r"\s+(?:of|with|at)\s+(?:a\s+)?density\s+\S+\s+\S+", "", t)
        t = re.sub(r"\ba height of\b", "", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t[:1].upper() + t[1:]


_UNIT_N = {"n": 1.0, "kn": 1000.0, "lbf": 4.448}


def _load_words(text: str) -> str:
    """One load in plain words, for example 'about 4 kg pressing down on the top face'."""
    m = re.match(r"\s*([\d.]+)\s*(kn|n|lbf)\b\s*(down|up|sideways|left|right|inward|outward)?\s*(?:@|on|at)?\s*(\w+)?",
                 text, re.I)
    if not m:
        return text
    mag, unit, way, where = m.groups()
    n = float(mag) * _UNIT_N[unit.lower()]
    kg = n / 9.81
    kgs = f"{kg:.2g}" if kg < 100 else f"{kg:.0f}"
    verb = {"down": "pressing down", "up": "pulling up"}.get((way or "").lower(), f"pushing {way}" if way else "acting")
    face = f" on the {where} face" if where and where.lower() not in ("and", "within") else ""
    return f"{mag} {unit.upper() if unit.lower() != 'lbf' else 'lbf'} (about {kgs} kg) {verb}{face}"


def _support_words(text: str) -> str:
    t = re.sub(r"\s*\S*=\S*", "", text).strip()
    return {"wall": "the wall"}.get(t.lower(), t) or text


def _setting(sit: dict) -> str:
    """The situation in plain sentences. The exact form stays in the full report."""
    bits = []
    if sit.get("supports"):
        sup = [_support_words(x if isinstance(x, str) else str(x.get("text") or x.get("name") or x))
               for x in sit["supports"]]
        bits.append("Fixed at " + _join(sup) + ".")
    if sit.get("loads"):
        bits.append("Loaded with " + _join([_load_words(x if isinstance(x, str) else str(x.get("text") or x))
                                            for x in sit["loads"]]) + ".")
    env = sit.get("environment") or {}
    keep = {k: v for k, v in env.items() if isinstance(v, (str, int, float)) and not k.endswith("_name")}
    if keep:
        bits.append("Around it, " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in list(keep.items())[:4]) + ".")
    return " ".join(bits)


def _join(items) -> str:
    words = [str(x.get("text") or x.get("name") or x) if isinstance(x, dict) else str(x) for x in items]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def simulation_view(path) -> Path | None:
    """Write simulation.html next to simulation.json and return it, or None when no step has a scene."""
    path = Path(path)
    d = json.loads(path.read_text())
    out = path.parent
    steps = []
    for st in d.get("steps", []):
        rd = st.get("result_dir")
        if not rd or st.get("error"):
            continue
        rdir = Path(rd) if Path(rd).is_absolute() else (out / rd)
        # the step's folder beside the page, or the page's own folder, wins, so a moved or copied run shows its own
        here = next((p for p in (out / Path(rd).name, out) if p.name == Path(rd).name and (p / "outcome.json").exists()),
                    None)
        rdir = here or rdir
        if not rdir.exists():
            continue
        if not (rdir / FILE).exists() or (rdir / "report.html").exists():
            attach(rdir)
        if (rdir / FILE).exists():
            steps.append((st, rdir, (rdir / FILE).read_bytes()))
    if not steps:
        return None
    color, word = TONE.get(d.get("verdict", "unknown"), TONE["unknown"])
    thing = title_of(d.get("thing"))
    parts = []
    for i, (st, rdir, data) in enumerate(steps):
        head = load(data)[0]
        rep = rdir / "report.html"
        link = (f'<a href="{html.escape(os.path.relpath(rep, out))}">full report</a>' if rep.exists() else "")
        parts.append(f'<section><h2>{html.escape(str(st.get("analysis", head.get("domain", ""))).replace("-", " "))}'
                     f'</h2><p class="h">{html.escape(st.get("headline", ""))} {link}</p>'
                     f'<div data-wobbl-scene="#s{i}"></div>'
                     f'<script type="application/octet-stream" id="s{i}">{base64.b64encode(data).decode()}</script>'
                     "</section>")
    setting = _setting(d.get("situation") or {})
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(thing)} simulation</title><style>{font_css()}
body{{margin:0;background:#0e0f11;color:#f2f2ee;font:15px/1.55 Archivo,"Helvetica Neue",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 56px}}
.brand{{background:#ffd400;color:#111;font-weight:800;padding:2px 8px;margin-right:10px;font-size:13px;letter-spacing:.06em}}
h1{{font-size:24px;margin:10px 0 4px}}.q{{color:#a3a7ad;margin:0 0 12px;font-size:17px}}
.v{{display:inline-block;font-weight:800;letter-spacing:.05em;color:{color};margin-right:10px}}
.s{{color:#a3a7ad;font-size:14px;margin:6px 0 20px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:#ffd400;margin:28px 0 6px}}
.h{{margin:0 0 10px;color:#d8d8d2}}a{{color:#ffd400}}</style></head>
<body><main><div><span class="brand">WOBBL</span></div>
<h1>{html.escape(thing)}</h1><p class="q">{html.escape(d.get("question", ""))}</p>
<p><span class="v">{word}</span></p>{f'<p class="s">{html.escape(setting)}</p>' if setting else ""}
{"".join(parts)}
<script>{viewer_js()}</script></main></body></html>
"""
    target = out / "simulation.html"
    target.write_text(page, encoding="utf-8")
    return target
