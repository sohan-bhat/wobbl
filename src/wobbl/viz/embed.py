"""Put the viewer and a run's scene into its report.html, or into a page of its own, and open it.

Everything goes inline (the viewer, the scene as base64 and the mono font) so the page works offline as one file.
The block sits between two markers, so attaching again replaces it instead of adding a second one.
"""
from __future__ import annotations

import base64
import html
import re
import webbrowser
from functools import lru_cache
from pathlib import Path

from .scene import FILE

HERE = Path(__file__).parent
START, END = "<!--wobbl-view-->", "<!--/wobbl-view-->"
ANCHOR = "simulation"


PARTS = ("core.js", "objects.js", "mount.js")
HEAD = """// Wobbl's viewer, which plays a run's scene.wobbl in 3D with WebGL2 and nothing else. MIT licensed.
// WobblView.mount(element, {src}) with src a URL, base64 text or the gzipped bytes, or mark an element with
// data-wobbl-scene="URL or #id of a script holding base64" and it mounts on its own.
(() => {
'use strict';
const W = globalThis.WobblView = globalThis.WobblView || {};
if (W.mount) return;
"""


def bundle() -> str:
    """The whole viewer as one script: its three parts inside one function scope."""
    return HEAD + "\n".join((HERE / "viewer" / p).read_text(encoding="utf-8") for p in PARTS) + "})();\n"


@lru_cache(maxsize=1)
def viewer_js() -> str:
    """The viewer without comment lines and indentation, which is all a page needs."""
    lines = (s.strip() for s in bundle().splitlines())
    return "\n".join(s for s in lines if s and not s.startswith("//"))


@lru_cache(maxsize=1)
def font_css() -> str:
    data = base64.b64encode((HERE / "fonts" / "ibm-plex-mono-500-latin.woff2").read_bytes()).decode()
    return ("@font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:400 600;font-display:swap;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")


def block(scene: bytes, heading: str = "Simulation", note: str = "") -> str:
    """The viewer section for a report: a heading, the 3D view and the scene it plays."""
    data = base64.b64encode(scene).decode()
    p = f'<p class="wv-note">{html.escape(note)}</p>' if note else ""
    return (f'{START}<section id="{ANCHOR}" class="wobbl-sim"><style>{font_css()}'
            ".wobbl-sim{margin:0 0 24px}.wobbl-sim h2{margin-top:0}.wv-note{color:#a3a7ad;font-size:13px;margin:6px 0 0}"
            f'</style><h2>{html.escape(heading)}</h2><div data-wobbl-scene="#wobbl-scene-data"></div>{p}'
            f'<script type="application/octet-stream" id="wobbl-scene-data">{data}</script>'
            f"<script>{viewer_js()}</script></section>{END}")


def insert(page: str, section: str) -> str:
    """The report with the viewer section in it, right after the verdict, replacing any earlier one."""
    page = re.sub(re.escape(START) + ".*?" + re.escape(END), "", page, flags=re.S)
    m = re.search(r'<section class="verdict">.*?</section>', page, flags=re.S)
    if m:
        return page[:m.end()] + "\n" + section + page[m.end():]
    if "</main>" in page:
        return page.replace("</main>", section + "</main>", 1)
    return page.replace("</body>", section + "</body>", 1)


def standalone(scene: bytes, title: str, intro: str = "") -> str:
    """A page that holds nothing but the view, for a run without a report."""
    t = html.escape(title)
    lead = f"<p>{html.escape(intro)}</p>" if intro else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t}</title><style>{font_css()}
body{{margin:0;background:#0e0f11;color:#f2f2ee;font:15px/1.55 Archivo,"Helvetica Neue",Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 48px}}h1{{font-size:22px;margin:0 0 6px}}
p{{color:#a3a7ad;margin:0 0 16px}}.brand{{background:#ffd400;color:#111;font-weight:800;padding:2px 8px;margin-right:10px;
font-size:13px;letter-spacing:.06em;vertical-align:middle}}</style></head>
<body><main><h1><span class="brand">WOBBL</span>{t}</h1>{lead}
<div data-wobbl-scene="#wobbl-scene-data"></div>
<script type="application/octet-stream" id="wobbl-scene-data">{base64.b64encode(scene).decode()}</script>
<script>{viewer_js()}</script></main></body></html>
"""


def attach(out_dir: str | Path, build: bool = True) -> Path | None:
    """Make sure a finished run has its scene, and put the viewer into its report.html. Builds the scene from the
    run's own files when the domain did not write one. Returns the page that shows the view, or None when there is
    nothing to show."""
    out = Path(out_dir)
    scene = out / FILE
    if not scene.exists() and build:
        from .adapters import build_scene
        try:
            build_scene(out)
        except Exception:  # noqa: BLE001  a view that cannot be built must never fail the run
            import os
            if os.environ.get("WOBBL_DEBUG"):
                raise
            return None
    if not scene.exists():
        return None
    data = scene.read_bytes()
    report = out / "report.html"
    if report.exists():
        report.write_text(insert(report.read_text(encoding="utf-8"), block(data)), encoding="utf-8")
        return report
    from .scene import load
    title = load(data)[0].get("title") or "Simulation"
    page = out / "view.html"
    page.write_text(standalone(data, title), encoding="utf-8")
    return page


def view_path(out_dir: str | Path) -> Path | None:
    """The page that shows a run's interactive view, if it has one."""
    out = Path(out_dir)
    for name in ("simulation.html", "report.html", "view.html"):
        p = out / name
        if p.exists() and (name != "report.html" or START in p.read_text(encoding="utf-8", errors="ignore")):
            return p
    return None


def link(page: Path) -> str:
    """A file URL to the page, at the view itself when it sits inside a report."""
    uri = Path(page).resolve().as_uri()
    return uri + f"#{ANCHOR}" if Path(page).name == "report.html" else uri


def open_view(page: str | Path) -> bool:
    """Open the page in the default browser. Returns False when no browser could be started."""
    try:
        return bool(webbrowser.open(link(Path(page))))
    except Exception:  # noqa: BLE001
        return False
