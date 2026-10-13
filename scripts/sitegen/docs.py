"""Pages made from docs/*.md: get started, the guides, the command reference and validation.

The Markdown files are the one source for each topic. This module only says
which file becomes which page and adds what the site needs around it: the
breadcrumb, a short flow, an "on this page" list and links to the next guide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

from . import DOCS, url
from .layout import crumbs, flow, page, picture
from .markdown import ROUTES, render


@dataclass
class Guide:
    slug: str
    source: str
    category: str
    title: str
    sentence: str
    steps: list[str] = field(default_factory=list)
    image: str = ""
    image_alt: str = ""


GUIDES = [
    Guide("strength", "strength.md", "Strength", "Check a part’s strength",
          "Hold the part the way it is mounted, load it, and read the verdict with its checks.",
          ["inspect", "material", "supports", "loads", "verdict"],
          "cast-mount-stress", "Stress heatmap of a cast motor mount, peak 124 MPa at the rib tip."),
    Guide("fix", "fix.md", "Strength", "Fix a failing part",
          "Wobbl edits the part, solves every version again and lines up the numbers.",
          ["failing result", "fillet or gusset", "edited STEP", "solved again"],
          "bracket-gusset", "The shelf bracket with a gusset across its corner, peak 9.82 MPa."),
    Guide("heat", "heat.md", "Heat", "Heat",
          "Steady and transient temperatures with convection, radiation and held temperatures.",
          ["power in", "air and radiation out", "hottest point", "energy balance"],
          "heat-sink-temperature", "Temperature map of an aluminum heat sink, 70.3 C under the chip."),
    Guide("circuits", "circuits.md", "Circuits", "Circuits",
          "LEDs, buck converters, MOSFETs or any netlist in ngspice, checked against the textbook.",
          ["template or netlist", "ngspice", "textbook check", "fix"],
          "buck-waveform", "Output ripple of a 12 V to 5 V buck converter, 8.56 mV peak to peak."),
    Guide("agents", "agents.md", "Agents", "Use Wobbl from your agent",
          "Claude Code, Codex, Cursor and Gemini CLI call Wobbl’s tools and read the checks first.",
          ["your question", "MCP tools", "checks", "answer and heatmap"],
          "report-bracket", "The HTML report Wobbl wrote for the shelf bracket: fails, 52.3 MPa."),
    Guide("checks", "verification.md", "Verification", "How Wobbl checks an answer",
          "8 checks run on every strength answer and set its confidence, with the reasons.",
          ["convergence", "balance", "hand calculation", "singularities", "confidence"],
          "motor-mount-noribs", "A printed motor mount with two idealization peaks labeled artifact."),
]


@dataclass
class DocPage:
    path: str
    sources: list[str]
    section: str
    crumb: str
    description: str
    anchors: list[str] = field(default_factory=list)
    wide: bool = False


PAGES = [
    DocPage("get-started/", ["getting-started.md"], "Get started", "Get started",
            "Install Wobbl, add CalculiX and ngspice, connect Claude Code, Codex, Cursor or Gemini CLI, "
            "and get a first checked answer."),
    *[DocPage(f"guides/{g.slug}/", [g.source], "Guides", g.title, g.sentence) for g in GUIDES],
    DocPage("guides/reference/", ["cli.md"], "Guides", "Command line reference",
            "Every Wobbl command and option, with the load, support and face selector syntax."),
    DocPage("validation/", ["validation.md", "bakeoff.md", "limits.md"], "Validation", "Validation",
            "18 cases with known answers, a bake off against a coding agent working alone, "
            "and what Wobbl does not model yet.", anchors=["bake-off", "limits"], wide=True),
]


def register_routes() -> None:
    """Tell the link rewriter which docs are pages, so links between them stay on the site."""
    for p in PAGES:
        for i, src in enumerate(p.sources):
            ROUTES[f"docs/{src}"] = p.path + (f"#{p.anchors[i - 1]}" if i else "")
    ROUTES["docs"] = "guides/"


def _pager(path: str) -> str:
    order = [f"guides/{g.slug}/" for g in GUIDES]
    if path not in order:
        return ""
    i = order.index(path)
    links = []
    if i > 0:
        g = GUIDES[i - 1]
        links.append(f'<a class="pager-prev" href="{url(order[i - 1])}"><span>Previous</span>{escape(g.title)}</a>')
    if i + 1 < len(order):
        g = GUIDES[i + 1]
        links.append(f'<a class="pager-next" href="{url(order[i + 1])}"><span>Next</span>{escape(g.title)}</a>')
    return f'<nav class="pager" aria-label="More guides">{"".join(links)}</nav>'


def doc_page(p: DocPage) -> str:
    used: set[str] = set()
    first = render(DOCS / p.sources[0], used=used)
    body, toc = first.body, list(first.toc)
    for src, anchor in zip(p.sources[1:], p.anchors):
        part = render(DOCS / src, demote=1, used=used, top_id=anchor)
        body += part.body
        toc += part.toc
    trail = [("Home", "")]
    if p.section == "Guides":
        trail.append(("Guides", "guides/"))
    trail.append((p.crumb, None))
    guide = next((g for g in GUIDES if p.path == f"guides/{g.slug}/"), None)
    steps = flow(guide.steps) if guide else ""
    aside = ""
    if len(toc) > 2 and not p.wide:
        items = "".join(f'<li><a href="#{slug}">{escape(text)}</a></li>' for text, slug in toc)
        aside = f'<aside class="toc" aria-labelledby="toc-title"><p id="toc-title">On this page</p><ol>{items}</ol></aside>'
    main = f"""<div class="wrap page{' page-wide' if p.wide else ''}">
  {crumbs(trail)}
  <header class="page-head">
    <h1>{escape(first.title)}</h1>
    {f'<p class="page-lead">{first.lead}</p>' if first.lead else ''}
    {steps}
  </header>
  <div class="doc-grid">
    <article class="prose">
{body}
    </article>
    {aside}
  </div>
  {_pager(p.path)}
</div>"""
    title = f"{first.title} | Wobbl"
    return page(path=p.path, title=title, description=p.description, main=main)


def guides_index() -> str:
    blocks = []
    for g in GUIDES:
        blocks.append(f"""  <li class="guide">
    <div class="guide-text">
      <p class="guide-cat">{escape(g.category)}</p>
      <h2><a href="{url(f'guides/{g.slug}/')}">{escape(g.title)}</a></h2>
      <p>{escape(g.sentence)}</p>
      {flow(g.steps)}
    </div>
    {picture(g.image, g.image_alt, sizes="(min-width: 900px) 360px, 100vw", cls="guide-img") if g.image else ''}
  </li>""")
    main = f"""<div class="wrap page">
  {crumbs([("Home", ""), ("Guides", None)])}
  <header class="page-head">
    <h1>Guides</h1>
    <p class="page-lead">Each guide follows a real example: the question, the command and what comes back.</p>
  </header>
  <ol class="guides">
{chr(10).join(blocks)}
  </ol>
  <p class="guides-ref">Every command and option: <a href="{url('guides/reference/')}">command line reference</a>.</p>
</div>"""
    return page(path="guides/", title="Guides | Wobbl",
                description="Guides to checking strength, fixing parts, heat, circuits, using Wobbl from a "
                            "coding agent and how Wobbl checks its answers.", main=main)
