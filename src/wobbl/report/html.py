"""A single self contained HTML report: verdict, numbers, heatmaps, checks, assumptions.

No external fonts, scripts or images; everything is inlined so the file can be
emailed, attached to a pull request or opened offline.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html
from pathlib import Path

from .. import __version__

CSS = """
:root{--bg:#0e0f11;--panel:#16181b;--line:#2a2d32;--fg:#f2f2ee;--muted:#a3a7ad;--yellow:#ffd400;
--pass:#3ccf7a;--warn:#ffb020;--fail:#ff4d4f;--info:#5aa9ff;--skip:#6b7078}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 20px 64px}
.mono,code,td.num{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;border-bottom:2px solid var(--yellow);
padding-bottom:14px;margin-bottom:24px}
header h1{font-size:26px;margin:0;letter-spacing:.02em}
header .meta{color:var(--muted);font-size:13px;text-align:right}
.brand{display:inline-block;background:var(--yellow);color:#111;font-weight:800;padding:2px 8px;margin-right:10px;
letter-spacing:.06em;font-size:13px;vertical-align:middle}
.question{color:var(--muted);font-size:16px;margin:4px 0 0}
.verdict{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px;margin-bottom:24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.word{font-size:40px;font-weight:800;letter-spacing:.04em;line-height:1}
.word.holds,.word.pass{color:var(--pass)}.word.marginal,.word.warn{color:var(--warn)}.word.fails,.word.fail{color:var(--fail)}
.headline{margin:12px 0 0;font-size:16px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.metric .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.metric .v{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:22px;font-weight:700}
.badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:700;
text-transform:uppercase;letter-spacing:.06em}
.badge.high,.badge.pass{background:rgba(60,207,122,.15);color:var(--pass)}
.badge.medium,.badge.warn{background:rgba(255,176,32,.15);color:var(--warn)}
.badge.low,.badge.fail{background:rgba(255,77,79,.15);color:var(--fail)}
.badge.info{background:rgba(90,169,255,.15);color:var(--info)}.badge.skip{background:#22252a;color:var(--skip)}
ul.reasons{margin:8px 0 0;padding-left:18px;color:var(--muted);font-size:13px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.1em;color:var(--yellow);margin:34px 0 12px}
figure{margin:0}
figure img{width:100%;height:auto;display:block;border-radius:8px;border:1px solid var(--line);background:#0e0f11}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
.hero{margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
td.num{text-align:right;white-space:nowrap}
.muted{color:var(--muted)}
ol.assume,ul.plain{padding-left:20px}ol.assume li,ul.plain li{margin:4px 0}
.fix{border-left:3px solid var(--yellow);padding:10px 14px;margin:10px 0;background:var(--panel);border-radius:0 8px 8px 0}
.fix .t{font-weight:700}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;font-size:12px}
@media (max-width:760px){.verdict{grid-template-columns:1fr}.word{font-size:32px}header{flex-direction:column;
align-items:flex-start}header .meta{text-align:left}}
@media print{body{background:#fff;color:#111}.card,.fix,pre{background:#f6f6f6;border-color:#ddd}}
"""


def esc(x) -> str:
    return html.escape(str(x))


def img_tag(path: str | Path, alt: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f'<img src="data:{mime};base64,{data}" alt="{esc(alt)}" loading="lazy">'


def figure(path, caption, cls="") -> str:
    t = img_tag(path, caption)
    if not t:
        return ""
    return f'<figure class="{cls}">{t}<figcaption>{esc(caption)}</figcaption></figure>'


def table(headers: list[str], rows: list[list], num_cols: set[int] | None = None) -> str:
    num_cols = num_cols or set()
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            raw = isinstance(c, tuple) and c[0] == "html"
            val = c[1] if raw else esc(c)
            cells.append(f'<td class="{"num" if i in num_cols else ""}">{val}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def badge(status: str) -> tuple[str, str]:
    return ("html", f'<span class="badge {esc(status)}">{esc(status)}</span>')


def page(title: str, part: str, question: str, verdict_word: str, verdict_cls: str, headline: str,
         metrics: list[tuple[str, str]], confidence: str, reasons: list[str], sections: list[str],
         footer_extra: str = "") -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    met = "".join(f'<div class="metric"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                  for k, v in metrics)
    rs = "".join(f"<li>{esc(r)}</li>" for r in reasons[:6])
    q = f'<p class="question">{esc(question)}</p>' if question else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body><main>
<header><div><h1><span class="brand">WOBBL</span>{esc(part)}</h1>{q}</div>
<div class="meta">{esc(title)}<br>{now}</div></header>
<section class="verdict">
 <div class="card"><div class="word {esc(verdict_cls)}">{esc(verdict_word)}</div>
  <p class="headline">{esc(headline)}</p></div>
 <div class="card"><div class="metrics">{met}</div>
  <p style="margin:14px 0 0">Confidence <span class="badge {esc(confidence)}">{esc(confidence)}</span></p>
  <ul class="reasons">{rs}</ul></div>
</section>
{''.join(sections)}
<footer>Made with Wobbl {__version__}. {footer_extra}</footer>
</main></body></html>
"""
