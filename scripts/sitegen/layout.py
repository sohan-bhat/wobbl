"""The shared page shell: head tags, navigation, breadcrumb and footer."""
from __future__ import annotations

import hashlib
import struct
from html import escape
from pathlib import Path

from . import REPO, SITE_SRC, full_url, url

NAV = [
    ("Get started", "get-started/"),
    ("Guides", "guides/"),
    ("Examples", "examples/"),
    ("Validation", "validation/"),
]

_hashes: dict[str, str] = {}


def asset(name: str) -> str:
    """Link to a file in site/ with a content hash, so a changed file is never served stale."""
    if name not in _hashes:
        _hashes[name] = hashlib.sha1((SITE_SRC / name).read_bytes()).hexdigest()[:10]
    return f"{url(name)}?v={_hashes[name]}"


def image_size(path: Path) -> tuple[int, int]:
    """Width and height of a WebP or PNG file, read from its header."""
    b = path.read_bytes()[:32]
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        kind = b[12:16]
        if kind == b"VP8X":
            return 1 + int.from_bytes(b[24:27], "little"), 1 + int.from_bytes(b[27:30], "little")
        if kind == b"VP8 ":
            w, h = struct.unpack("<HH", b[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if kind == b"VP8L":
            bits = int.from_bytes(b[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", b[16:24])
    raise ValueError(f"unknown image format: {path}")


def picture(name: str, alt: str, sizes: str = "100vw", lazy: bool = True, cls: str = "") -> str:
    """An <img> for site/img/<name>.webp with its smaller copy in srcset when there is one."""
    main = SITE_SRC / "img" / f"{name}.webp"
    w, h = image_size(main)
    small = sorted((SITE_SRC / "img").glob(f"{name}-[0-9]*.webp"))
    srcset = ", ".join(f"{url('img/' + p.name)} {image_size(p)[0]}w" for p in small)
    srcset = f' srcset="{srcset}, {url("img/" + main.name)} {w}w" sizes="{sizes}"' if srcset else ""
    load = ' loading="lazy" decoding="async"' if lazy else ""
    c = f' class="{cls}"' if cls else ""
    return (f'<img{c} src="{url("img/" + main.name)}"{srcset} width="{w}" height="{h}"{load} '
            f'alt="{escape(alt)}">')


def crumbs(trail: list[tuple[str, str | None]]) -> str:
    """A breadcrumb from (label, path) pairs; the last one is the current page."""
    items = []
    for label, path in trail:
        if path is None:
            items.append(f'<li aria-current="page">{escape(label)}</li>')
        else:
            items.append(f'<li><a href="{url(path)}">{escape(label)}</a></li>')
    return f'<nav class="crumbs" aria-label="Breadcrumb"><ol>{"".join(items)}</ol></nav>'


def flow(steps: list[str]) -> str:
    """A small arrow flow such as inspect, supports, loads, solve."""
    li = "".join(f"<li>{escape(s)}</li>" for s in steps)
    return f'<ol class="flow" aria-label="Steps">{li}</ol>'


def _nav(path: str) -> str:
    links = []
    for label, target in NAV:
        if path == target:
            links.append(f'<a href="{url(target)}" aria-current="page">{label}</a>')
        elif path.startswith(target):
            links.append(f'<a href="{url(target)}" class="is-here">{label}</a>')
        else:
            links.append(f'<a href="{url(target)}">{label}</a>')
    return f"""<header class="nav">
  <div class="wrap nav-row">
    <a class="brand" href="{url()}" aria-label="Wobbl home">
      <img src="{url('assets/logo-dark.svg')}?v=1" width="136" height="22" alt="Wobbl">
    </a>
    <nav class="nav-links" aria-label="Main">
      {(chr(10) + '      ').join(links)}
    </nav>
    <a class="nav-gh" href="{REPO}">GitHub</a>
  </div>
</header>"""


FOOTER = """<footer class="footer">
  <div class="wrap footer-row">
    <div class="footer-brand">
      <img src="{logo}" width="136" height="22" alt="Wobbl" loading="lazy">
      <p>Simulate anything before you build it. Open source, by Sohan Bhat.</p>
    </div>
    <ul class="footer-links" aria-label="Project">
      <li><a href="{repo}/blob/main/CHANGELOG.md">Changelog</a></li>
      <li><a href="{repo}/issues">Issues</a></li>
      <li><a href="{repo}/blob/main/LICENSE">MIT license</a></li>
    </ul>
  </div>
</footer>"""


def page(*, path: str, title: str, description: str, main: str, head: str = "",
         og_title: str | None = None, indexable: bool = True) -> str:
    """A complete HTML document. ``path`` is the page's route under the base, '' for home."""
    t = escape(title)
    d = escape(description)
    og_t = escape(og_title or title)
    canonical = f'<link rel="canonical" href="{full_url(path)}">\n' if indexable else \
        '<meta name="robots" content="noindex">\n'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<meta name="description" content="{d}">
{canonical}<meta name="theme-color" content="#0e1011">
<meta name="color-scheme" content="dark">
<link rel="icon" href="{url('favicon.svg')}?v=1" type="image/svg+xml">
<link rel="icon" href="{url('favicon-32.png')}?v=1" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="{url('apple-touch-icon.png')}?v=1">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Wobbl">
<meta property="og:title" content="{og_t}">
<meta property="og:description" content="{d}">
<meta property="og:url" content="{full_url(path)}">
<meta property="og:image" content="{full_url('og.png')}?v=2">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="The Wobbl logo next to a stress heatmap of a cast motor mount, peak 124 MPa at the rib tip.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{og_t}">
<meta name="twitter:description" content="{d}">
<meta name="twitter:image" content="{full_url('og.png')}?v=2">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,400..800&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;display=swap">
<link rel="stylesheet" href="{asset('styles.css')}">
{head}<script type="module" src="{asset('app.js')}"></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>

{_nav(path)}

<main id="main">
{main.strip()}
</main>

{FOOTER.format(logo=url('assets/logo-dark.svg') + '?v=1', repo=REPO)}
</body>
</html>
"""
