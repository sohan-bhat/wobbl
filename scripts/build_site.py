"""Build the Wobbl website into plain static files for GitHub Pages.

    python scripts/build_site.py              # writes _site/
    python scripts/build_site.py --serve      # builds, then serves http://localhost:8040/wobbl/

The Pages workflow runs the first form and publishes _site/. Use the
interpreter Wobbl is installed in (markdown-it-py comes with it through rich).

Where each page comes from:

- site/pages/*.html   the landing and 404 pages, written by hand
- docs/*.md           get started, the guides and validation (see sitegen/docs.py)
- examples/*/         one page per example, from its README, run.sh and outputs

Everything else in site/ (styles, scripts, images, the 3D part) is copied as is.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sitegen import PAGES_SRC, ROOT, SITE_SRC, cfg, url
from sitegen import docs
from sitegen.layout import asset, page

PORT = 8040


def write(out: Path, path: str, html: str) -> None:
    """Write a page at its route: '' is index.html, 'guides/' is guides/index.html."""
    dest = out / (path + "index.html" if path == "" or path.endswith("/") else path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")


def copy_static(out: Path) -> None:
    for item in SITE_SRC.iterdir():
        if item.name == PAGES_SRC.name or item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name)
        else:
            shutil.copy2(item, out / item.name)


FIELD = re.compile(r"^(\w+):[ \t]*(.*)$", re.M)
TOKEN = re.compile(r"\{\{(\w+):([^}]*)\}\}")


def hand_page(src: Path, fills: dict) -> tuple[str, str]:
    """A page from site/pages: a comment with its fields, an optional head block, then the main content."""
    text = src.read_text(encoding="utf-8")
    m = re.match(r"\s*<!--(.*?)-->\s*", text, re.S)
    meta = dict(FIELD.findall(m.group(1)))
    text = text[m.end():]
    head = ""
    h = re.match(r"<!-- head -->(.*?)<!-- /head -->\s*", text, re.S)
    if h:
        head, text = h.group(1).strip() + "\n", text[h.end():]

    def fill(t: re.Match) -> str:
        kind, arg = t.group(1), t.group(2).strip()
        if kind == "url":
            return url(arg)
        if kind == "asset":
            return asset(arg)
        return fills[kind](arg)

    head, text = TOKEN.sub(fill, head), TOKEN.sub(fill, text)
    path = meta.get("path", "")
    return path, page(path=path, title=meta["title"], description=meta["description"], main=text,
                      head=head, og_title=meta.get("og_title"), indexable=path != "404.html")


def build(out: Path) -> list[str]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    copy_static(out)
    docs.register_routes()
    routes = []
    for p in docs.PAGES:
        write(out, p.path, docs.doc_page(p))
        routes.append(p.path)
    write(out, "guides/", docs.guides_index())
    routes.append("guides/")
    for src in sorted(PAGES_SRC.glob("*.html")):
        path, html = hand_page(src, {})
        write(out, path, html)
        routes.append(path)
    return routes


class Preview(SimpleHTTPRequestHandler):
    """Serves the build under the base path, with the 404 page for anything missing, like GitHub Pages."""

    def __init__(self, *a, root: Path, **kw):
        self.root = root
        super().__init__(*a, directory=str(root), **kw)

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith(cfg.base):
            return super().translate_path("/" + clean[len(cfg.base):])
        return str(self.root / "__missing__")

    def do_GET(self):
        if self.path in ("/", "") and cfg.base != "/":
            self.send_response(302)
            self.send_header("Location", cfg.base)
            self.end_headers()
            return
        target = Path(self.translate_path(self.path))
        if target.is_dir() and not self.path.split("?")[0].endswith("/"):
            return super().do_GET()  # the stock handler redirects to the trailing slash
        if not target.exists() and not (target / "index.html").exists():
            body = (self.root / "404.html").read_bytes()
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default _site/)")
    ap.add_argument("--base", default=cfg.base, help="path the site is served under (default /wobbl/)")
    ap.add_argument("--serve", action="store_true", help="serve the build for a local preview")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    cfg.base = "/" + a.base.strip("/") + "/" if a.base.strip("/") else "/"
    routes = build(a.out)
    print(f"built {len(routes)} pages into {a.out}")
    if a.serve:
        server = ThreadingHTTPServer(("127.0.0.1", a.port), partial(Preview, root=a.out.resolve()))
        print(f"previewing at http://localhost:{a.port}{cfg.base}  (Ctrl+C stops)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
