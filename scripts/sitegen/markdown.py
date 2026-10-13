"""Render the repository's Markdown into site HTML.

The Markdown stays readable on GitHub; this module only adapts it for the site:
links between docs become site routes, images point at site/img, code fences
become labeled terminal blocks with a copy button, tables scroll on their own,
and headings get the same ids GitHub gives them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from . import ROOT, SITE_SRC, repo_url, url
from .layout import image_size

LABELS = {"sh": "Terminal", "console": "Terminal", "python": "Python", "text": "Output", "toml": "TOML",
          "json": "JSON"}

# Repository paths that are pages on the site, filled in by docs.py and examples.py
ROUTES: dict[str, str] = {}


def slugify(text: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def _strip_comment(line: str) -> tuple[str, str]:
    """Split a shell line into the command and a trailing # comment, ignoring # inside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i].rstrip(), line[i:]
    return line, ""


def term(code: str, lang: str = "sh", label: str | None = None) -> str:
    """A terminal block: a label bar with a copy button over the code.

    Shell lines get a $ prompt, comments are dimmed, and the copy button copies
    only the commands, without prompts, comments or output, so the text pastes
    cleanly into any shell.
    """
    label = label or LABELS.get(lang, lang.upper() or "Code")
    lines, copy = [], []
    if lang in ("sh", "console"):
        cont = False
        for raw in code.rstrip("\n").split("\n"):
            if lang == "console" and not cont and not raw.startswith("$ "):
                lines.append(f'<span class="ln o">{escape(raw) or " "}</span>')
                continue
            text = raw[2:] if lang == "console" and not cont else raw
            cmd, comment = _strip_comment(text)
            if not cmd.strip():
                lines.append(f'<span class="ln"><span class="c">{escape(comment)}</span></span>' if comment
                             else '<span class="ln"> </span>')
                continue
            prompt = "" if cont else '<span class="p">$ </span>'
            tail = f' <span class="c">{escape(comment)}</span>' if comment else ""
            lines.append(f'<span class="ln">{prompt}{escape(cmd)}{tail}</span>')
            copy.append(cmd)
            cont = cmd.endswith("\\")
    else:
        lines = [f'<span class="ln">{escape(ln) or " "}</span>' for ln in code.rstrip("\n").split("\n")]
        copy = code.rstrip("\n").split("\n")
    button = ""
    if lang != "text":
        data = escape("\n".join(copy), quote=True).replace("\n", "&#10;")
        button = (f'<button class="copy" type="button" data-copy="{data}" '
                  f'aria-label="Copy: {escape(label, quote=True)}">Copy</button>')
    return (f'<div class="term" data-copy-block><div class="term-bar"><span>{escape(label)}</span>{button}</div>'
            f'<pre translate="no"><code>{"".join(lines)}</code></pre></div>\n')


def site_link(href: str, src_dir: Path) -> str:
    """Where a link in a Markdown file should point on the site."""
    if re.match(r"^[a-z]+:", href) or href.startswith("#"):
        return href
    path, _, frag = href.partition("#")
    target = (src_dir / path).resolve()
    try:
        rel = target.relative_to(ROOT).as_posix()
    except ValueError:
        return href
    rel = rel.rstrip("/")
    for key in (rel, rel + "/README.md", rel.removesuffix("/README.md")):
        if key in ROUTES:
            dest = url(ROUTES[key])
            if frag:
                dest = dest.split("#")[0] + "#" + frag
            return dest
    if rel.startswith("site/"):
        return url(rel[len("site/"):])
    return repo_url(rel, folder=target.is_dir()) + (f"#{frag}" if frag else "")


def _img(src: str, alt: str, src_dir: Path, caption: str = "") -> str:
    """An image with real dimensions when it is a site image, wrapped in a figure."""
    href = site_link(src, src_dir)
    dims = ""
    local = (src_dir / src).resolve()
    if local.is_file() and SITE_SRC in local.parents:
        w, h = image_size(local)
        dims = f' width="{w}" height="{h}"'
    img = f'<img src="{escape(href)}"{dims} loading="lazy" decoding="async" alt="{escape(alt)}">'
    cap = f"<figcaption>{escape(caption)}</figcaption>" if caption else ""
    return f"<figure>{img}{cap}</figure>\n"


def _html_attrs(html: str, src_dir: Path) -> str:
    """Rewrite src and href in raw HTML blocks the same way as Markdown links."""
    html = re.sub(r'\s+width="\d+%"', "", html)

    def one(m: re.Match) -> str:
        attr, value = m.group(1), m.group(2)
        out = f'{attr}="{escape(site_link(value, src_dir))}"'
        if attr == "src":
            local = (src_dir / value).resolve()
            if local.is_file() and SITE_SRC in local.parents:
                w, h = image_size(local)
                out += f' width="{w}" height="{h}" loading="lazy" decoding="async"'
        return out

    return re.sub(r'\b(src|href)="([^"]+)"', one, html)


@dataclass
class Doc:
    title: str
    lead: str
    body: str
    toc: list[tuple[str, str]] = field(default_factory=list)


def _md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])

    def fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        info = tok.info.strip().split(None, 1)
        lang = info[0] if info else ""
        return term(tok.content, lang, info[1] if len(info) > 1 else None)

    def table_open(self, tokens, idx, options, env):
        name = escape(tokens[idx].meta.get("label", "Table"), quote=True)
        return f'<div class="table-scroll" role="region" aria-label="{name}" tabindex="0"><table class="data">\n'

    md.add_render_rule("fence", fence)
    md.add_render_rule("table_open", table_open)
    md.add_render_rule("table_close", lambda self, t, i, o, e: "</table></div>\n")
    return md


MD = _md()


def _text(inline: Token) -> str:
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


def render(path: Path, *, demote: int = 0, used: set[str] | None = None, top_id: str | None = None) -> Doc:
    """Render one Markdown file. The H1 becomes the title and the paragraph after it the lead.

    ``demote`` shifts headings down a level for a file shown as a section of a
    longer page; its H1 then stays in the body as a heading with ``top_id``.
    """
    used = set() if used is None else used
    src_dir = path.parent
    tokens = MD.parse(path.read_text(encoding="utf-8"))
    title = lead = ""
    toc: list[tuple[str, str]] = []

    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1" and not demote:
        title = _text(tokens[1])
        tokens = tokens[3:]
        if tokens and tokens[0].type == "paragraph_open":
            lead = MD.renderer.renderInline(tokens[1].children, MD.options, {})
            tokens = tokens[3:]

    out: list[Token] = []
    last_heading = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type in ("heading_open", "heading_close"):
            level = min(int(tok.tag[1]) + demote, 6)
            tok.tag = f"h{level}"
            if tok.type == "heading_open":
                text = _text(tokens[i + 1])
                last_heading = text
                slug = top_id if (demote and level == 2 and top_id) else slugify(text)
                base, n = slug, 1
                while slug in used:
                    slug, n = f"{base}-{n}", n + 1
                used.add(slug)
                tok.attrSet("id", slug)
                if level == 2:
                    toc.append((text, slug))
        elif tok.type == "table_open":
            tok.meta["label"] = last_heading or "Table"
        elif tok.type == "html_block":
            tok.content = _html_attrs(tok.content, src_dir)
        elif tok.type == "inline":
            kids = tok.children or []
            if (tokens[i - 1].type == "paragraph_open" and len(kids) == 1 and kids[0].type == "image"):
                img = kids[0]
                html = _img(img.attrGet("src"), _text(img) or img.content, src_dir, img.attrGet("title") or "")
                out.pop()  # the paragraph_open
                block = Token("html_block", "", 0)
                block.content = html
                out.append(block)
                i += 2  # skip the paragraph_close
                continue
            for c in kids:
                if c.type == "link_open":
                    c.attrSet("href", site_link(c.attrGet("href"), src_dir))
                elif c.type == "image":
                    c.attrSet("src", site_link(c.attrGet("src"), src_dir))
                    c.attrSet("loading", "lazy")
                elif c.type == "html_inline":
                    c.content = _html_attrs(c.content, src_dir)
        out.append(tok)
        i += 1
    body = MD.renderer.render(out, MD.options, {})
    return Doc(title=title, lead=lead, body=body, toc=toc)
