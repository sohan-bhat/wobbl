"""Pieces of the website build. ``scripts/build_site.py`` is the entry point.

The site is plain static files. Every page shares one layout (``layout.py``);
the landing and 404 pages are written by hand in ``site/pages/``, the guides
come from ``docs/*.md`` and the example pages from ``examples/*/``, so each
topic has one source.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE_SRC = ROOT / "site"
PAGES_SRC = SITE_SRC / "pages"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"

SITE_URL = "https://sohan-bhat.github.io"
REPO = "https://github.com/sohan-bhat/wobbl"


class Config:
    """Build settings shared by every module. ``base`` is the path the site is served under."""

    base = "/wobbl/"


cfg = Config()


def url(path: str = "") -> str:
    """A link to a page or file on the site, under the base path."""
    return cfg.base + path.lstrip("/")


def full_url(path: str = "") -> str:
    """The public address of a page, for canonical and Open Graph tags."""
    return SITE_URL + url(path)


def repo_url(rel: str, folder: bool = False) -> str:
    """A file or folder in the repository on GitHub."""
    return f"{REPO}/{'tree' if folder else 'blob'}/main/{rel}"
