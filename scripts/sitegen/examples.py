"""Example pages, read straight from examples/*/: the README, run.sh and each run's outputs.

Nothing here is written by hand. The question and command come from run.sh,
the verdict, numbers and checks from outcome.json, the plain text from
summary.txt, and the notes from the example's README, so the pages always
match the files a user gets when they run the example.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from . import EXAMPLES, ROOT, repo_url, url
from .layout import crumbs, page, picture
from .markdown import ROUTES, render, term

ORDER = ["cast-motor-mount", "shelf-bracket", "motor-mount", "enclosure", "drone-arm", "heat-sink",
         "battery-pack", "circuits"]

# Site images (site/img, made by scripts/build_site_images.py) for each run's output images
RUN_IMAGES = {
    "cast-motor-mount/output": {"stress": "cast-mount-stress"},
    "shelf-bracket/output": {"stress": "bracket-as-drawn"},
    "shelf-bracket/output/fix-fillet": {"stress": "bracket-fillet"},
    "shelf-bracket/output/fix-thicken": {"stress": "bracket-thicken"},
    "shelf-bracket/output/fix-gusset": {"stress": "bracket-gusset"},
    "motor-mount/output": {"stress": "motor-mount-ribs"},
    "motor-mount/output-noribs": {"stress": "motor-mount-noribs"},
    "enclosure/output": {"stress": "enclosure-stress"},
    "drone-arm/output": {"stress": "drone-arm-stress"},
    "heat-sink/output": {"temperature": "heat-sink-temperature"},
    "battery-pack/output": {"temperature": "battery-temperature", "history": "battery-history"},
    "circuits/buck": {"waveform": "buck-waveform"},
}

DOMAIN = {"strength": "Strength", "thermal": "Heat", "circuit": "Circuits"}
COLOR = {"holds": "holds", "ok": "holds", "marginal": "marginal", "fails": "fails", "hot": "fails"}


@dataclass
class Run:
    rel: str                 # e.g. "shelf-bracket/output"
    question: str
    command: str
    outcome: dict
    summary: str
    report: str | None       # site path of the copied report, if it is in the repository
    fixes: list[dict] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.outcome["verdict"]

    def images(self) -> list[tuple[str, str]]:
        """(site image name, alt text) for each image this run has on the site."""
        names = RUN_IMAGES.get(self.rel, {})
        kinds = {"stress": "Stress heatmap", "temperature": "Temperature map", "history": "Temperature over time",
                 "waveform": "Simulated waveform"}
        return [(names[k], f"{kinds.get(k, k)}: {self.outcome['headline']}") for k in names]

    def metrics(self) -> list[tuple[str, str]]:
        o = self.outcome
        if o["domain"] == "strength":
            rows = [("Peak stress", f"{o['governing_MPa']:.3g} MPa"), ("Safety factor", f"{o['safety_factor']:.2f}"),
                    ("Target", f"{o['target_sf']:g}")]
        else:
            rows = [(k[:1].upper() + k[1:], v) for k, v in o.get("metrics", [])[:3]]
        return [(k, v.replace(" ", "\u00a0")) for k, v in rows] + [("Confidence", o["confidence"])]


@dataclass
class Example:
    name: str
    title: str
    subtitle: str
    runs: list[Run]

    @property
    def domain(self) -> str:
        return DOMAIN.get(self.runs[0].outcome["domain"], "")


def _tracked() -> set[str] | None:
    try:
        out = subprocess.run(["git", "ls-files", "examples"], cwd=ROOT, capture_output=True, text=True, check=True)
        return set(out.stdout.split())
    except (OSError, subprocess.CalledProcessError):
        return None


def display(cmd: str) -> str:
    """The command as a reader would type it: no --question or -o, wrapped at about 72 columns."""
    cmd = re.sub(r'\s--question\s+"[^"]*"', "", cmd)
    cmd = re.sub(r"\s-o\s+\S+", "", cmd)
    groups: list[str] = []
    for w in shlex.split(cmd, posix=False):
        if groups and not w.startswith("-"):
            groups[-1] += " " + w  # the command's own words, or an option's value
        else:
            groups.append(w)
    lines, cur = [], ""
    for g in groups:
        if cur and len(cur) + len(g) + 1 > 72:
            lines.append(cur + " \\")
            cur = "  " + g
        else:
            cur = f"{cur} {g}" if cur else g
    return "\n".join(lines + [cur])


def parse_run_sh(path: Path) -> list[tuple[list[str], str]]:
    """The wobbl commands in a run.sh, with shell variables filled in: (argv, display text)."""
    text = re.sub(r"\\\n\s*", " ", path.read_text(encoding="utf-8"))
    env: dict[str, str] = {}
    out = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.+)$", line)
        if m:
            env[m.group(1)] = shlex.split(m.group(2))[0]
            continue
        if not line.startswith("wobbl "):
            continue
        for k, v in env.items():
            line = re.sub(r"\$\{?" + k + r"\}?", lambda _: v, line)
        out.append((shlex.split(line), display(line)))
    return out


def _opt(argv: list[str], *names: str) -> str | None:
    for i, a in enumerate(argv[:-1]):
        if a in names:
            return argv[i + 1]
    return None


def load(name: str, tracked: set[str] | None) -> Example:
    folder = EXAMPLES / name
    readme = (folder / "README.md").read_text(encoding="utf-8")
    h1 = readme.splitlines()[0].lstrip("# ").strip()
    title, _, sub = h1.partition(":")
    runs: list[Run] = []
    for argv, shown in parse_run_sh(folder / "run.sh"):
        if argv[1] == "fix":
            target = next((r for r in runs if r.rel == f"{name}/{argv[2]}"), None)
            if target:
                target.fixes = [f for f in target.outcome.get("fixes", []) if f.get("after")]
                target.command += "\n" + shown
            continue
        outdir = _opt(argv, "-o", "--out") or "output"
        rel = f"{name}/{outdir}"
        outcome = json.loads((EXAMPLES / rel / "outcome.json").read_text(encoding="utf-8"))
        report_rel = f"examples/{rel}/report.html"
        has_report = (EXAMPLES / rel / "report.html").exists() and (tracked is None or report_rel in tracked)
        runs.append(Run(rel=rel, question=_opt(argv, "--question") or outcome.get("question", ""), command=shown,
                        outcome=outcome, summary=(EXAMPLES / rel / "summary.txt").read_text(encoding="utf-8"),
                        report=f"examples/{rel}/report.html" if has_report else None))
    sub = sub.strip()
    return Example(name, title.strip(), sub[:1].upper() + sub[1:], runs)


def load_all() -> list[Example]:
    tracked = _tracked()
    examples = [load(n, tracked) for n in ORDER if (EXAMPLES / n / "run.sh").exists()]
    for ex in examples:
        ROUTES[f"examples/{ex.name}"] = f"examples/{ex.name}/"
        for r in ex.runs:
            for kind, img in RUN_IMAGES.get(r.rel, {}).items():
                src = r.outcome.get("images", {}).get(kind)
                if src:
                    ROUTES[f"examples/{r.rel}/{src}"] = f"img/{img}.webp"
    ROUTES["examples"] = "examples/"
    return examples


def verdict_html(run: Run, headline: bool = True) -> str:
    v = run.verdict
    text = f' {escape(run.outcome["headline"])}' if headline else ""
    return (f'<p class="run-verdict"><span class="vword" data-verdict="{COLOR.get(v, "marginal")}">{escape(v.upper())}'
            f'</span>{text}</p>')


def readout(run: Run) -> str:
    rows = "".join(f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k, v in run.metrics())
    return f'<dl class="readout">{rows}</dl>'


def checks(run: Run) -> str:
    items = "".join(f'<li data-status="{escape(c["status"])}"><span class="st">{escape(c["status"])}</span>'
                    f'{escape(c["name"])}</li>' for c in run.outcome.get("checks", []) if c["status"] != "skip")
    return f'<ul class="run-checks">{items}</ul>'


def result_visual(run: Run, sizes: str, lazy: bool = True) -> str:
    imgs = run.images()
    if imgs:
        return picture(imgs[0][0], imgs[0][1], sizes=sizes, lazy=lazy, cls="run-img")
    k, v = run.metrics()[0]
    return f'<div class="run-metric" data-verdict="{COLOR.get(run.verdict, "marginal")}"><p>{escape(v)}</p><p>{escape(k)}</p></div>'


def run_card(run: Run, *, compact: bool = False, link: tuple[str, str] | None = None) -> str:
    """The question, the command, the checks and the result of one run, side by side."""
    more = ""
    if not compact:
        more = (f'<details class="run-summary"><summary>summary.txt</summary>{term(run.summary, "text", "summary.txt")}'
                f"</details>")
        if run.report:
            more += f'<p class="run-files"><a href="{url(run.report)}">Open the full report</a></p>'
    if link:
        more += f'<p class="run-files"><a href="{link[0]}">{escape(link[1])}</a></p>'
    fixed = ""
    if compact and run.fixes:
        rows = "".join(f'<li><span class="fx-name">{escape(fix_label(f))}</span><span class="fx-n">'
                       f'{f["after"]["governing_MPa"]:.3g}\u00a0MPa</span><span class="vword" data-verdict="'
                       f'{COLOR.get(f["after"]["verdict"], "marginal")}">{escape(f["after"]["verdict"].upper())}</span></li>'
                       for f in run.fixes)
        fixed = f'<div class="run-step run-fixed"><p class="run-label">After wobbl fix --apply</p><ol>{rows}</ol></div>'
    sugg = [f["title"] for f in run.outcome.get("fixes", [])][:3] if not run.fixes else []
    if sugg:
        items = "".join(f"<li>{escape(t)}</li>" for t in sugg)
        label = "Suggested fix" if len(sugg) == 1 else "Suggested fixes"
        more = f'<div class="run-sugg"><p class="run-label">{label}</p><ul>{items}</ul></div>' + more
    extra = ""
    if not compact:
        extra = "".join(f'<figure class="run-extra">{picture(n, a, sizes="(min-width: 1000px) 40vw, 100vw")}</figure>'
                        for n, a in run.images()[1:])
    return f"""<div class="run">
  <div class="run-steps">
    <div class="run-step"><p class="run-label">You ask</p><p class="run-ask">{escape(run.question)}</p></div>
    <div class="run-step"><p class="run-label">Wobbl runs</p>{term(run.command, "sh")}</div>
    {f'<div class="run-step"><p class="run-label">Checks</p>{checks(run)}</div>' if run.outcome.get("checks") else ""}
    {fixed}
  </div>
  <div class="run-result">
    {result_visual(run, "(min-width: 1000px) 46vw, 100vw")}
    {verdict_html(run, headline=not compact)}
    {readout(run)}
    {extra}{more}
  </div>
</div>"""


def fix_label(f: dict) -> str:
    """A short name for an applied fix, from the parameters it was built with."""
    p = f["after"].get("params", {})
    if f["kind"] == "fillet" and "fillet" in p:
        return f"{p['fillet']:g} mm fillet"
    if f["kind"] == "thicken" and "thickness" in p:
        return f"Walls at {p['thickness']:g} mm"
    if f["kind"] == "gusset":
        return "Gusset across the corner"
    return f["title"]


def fix_strip(run: Run) -> str:
    o = run.outcome
    cells = [("As drawn", o["governing_MPa"], o["safety_factor"], o["verdict"], f"{run.rel}")]
    for f in run.fixes:
        a = f["after"]
        cells.append((fix_label(f), a["governing_MPa"], a["safety_factor"], a["verdict"],
                      f"{run.rel}/{a['report'].split('/')[0]}"))
    items = []
    for name, mpa, sf, verdict, rel in cells:
        img = RUN_IMAGES.get(rel, {}).get("stress")
        pic = picture(img, f"{name}: {mpa:.3g} MPa, {verdict}", sizes="(min-width: 1000px) 25vw, 50vw") if img else ""
        items.append(f'<li class="fix" data-verdict="{COLOR.get(verdict, "marginal")}">{pic}<h3>{escape(name)}</h3>'
                     f'<p class="fix-n"><span>{mpa:.3g} MPa</span><span>SF {sf:.2f}</span></p>'
                     f'<p class="fix-v">{escape(verdict.capitalize())}</p></li>')
    return (f'<section class="ex-fixes" aria-labelledby="fixes-title"><h2 class="ex-h" id="fixes-title">After '
            f'<code>wobbl fix --apply</code></h2><ol class="fixes">{"".join(items)}</ol></section>')


def example_page(ex: Example, prev: Example | None, nxt: Example | None) -> str:
    sections = []
    for r in ex.runs:
        sections.append(run_card(r))
        if r.fixes:
            sections.append(fix_strip(r))
    notes = render(EXAMPLES / ex.name / "README.md", keep_lead=True, drop_code={"./run.sh"})
    pager = []
    if prev:
        pager.append(f'<a class="pager-prev" href="{url(f"examples/{prev.name}/")}"><span>Previous</span>{escape(prev.title)}</a>')
    if nxt:
        pager.append(f'<a class="pager-next" href="{url(f"examples/{nxt.name}/")}"><span>Next</span>{escape(nxt.title)}</a>')
    main = f"""<div class="wrap page">
  {crumbs([("Home", ""), ("Examples", "examples/"), (ex.title, None)])}
  <header class="page-head">
    <h1>{escape(ex.title)}</h1>
    <p class="page-lead">{escape(ex.subtitle)}</p>
  </header>
  <div class="ex-runs">
{chr(10).join(sections)}
  </div>
  <div class="doc-grid">
    <article class="prose">
      <h2 id="about">About this example</h2>
{notes.body}
      <h2 id="run-it">Run it yourself</h2>
{term(f"git clone https://github.com/sohan-bhat/wobbl{chr(10)}cd wobbl/examples/{ex.name} && ./run.sh", "sh")}
      <p>The part, the script and every output are in <a href="{repo_url(f"examples/{ex.name}", folder=True)}">examples/{ex.name}</a> on GitHub.</p>
    </article>
  </div>
  <nav class="pager" aria-label="More examples">{"".join(pager)}</nav>
</div>"""
    first = ex.runs[0]
    desc = f"{first.question} {first.outcome['headline']}"
    return page(path=f"examples/{ex.name}/", title=f"{ex.title} example | Wobbl", description=desc, main=main)


def gallery(examples: list[Example]) -> str:
    cards = []
    for ex in examples:
        r = ex.runs[0]
        k, v = r.metrics()[0]
        cards.append(f"""  <li><a class="ex-card" href="{url(f'examples/{ex.name}/')}">
    {result_visual(r, "(min-width: 1100px) 300px, (min-width: 720px) 45vw, 100vw")}
    <span class="ex-cat">{escape(ex.domain)}</span>
    <span class="ex-title">{escape(ex.title)}</span>
    <span class="ex-sub">{escape(ex.subtitle)}</span>
    <span class="ex-v"><span class="vword" data-verdict="{COLOR.get(r.verdict, 'marginal')}">{escape(r.verdict.upper())}</span> {escape(v)}</span>
  </a></li>""")
    main = f"""<div class="wrap page">
  {crumbs([("Home", ""), ("Examples", None)])}
  <header class="page-head">
    <h1>Examples</h1>
    <p class="page-lead">Real parts and circuits, each run with Wobbl. Every number here comes from the run's own output.</p>
  </header>
  <ul class="ex-grid">
{chr(10).join(cards)}
  </ul>
</div>"""
    return page(path="examples/", title="Examples | Wobbl",
                description="Real Wobbl runs on a cast motor mount, a printed bracket, a drone arm, an enclosure, a heat "
                            "sink, a battery pack and three circuits, with their reports.", main=main)


def build(out: Path, examples: list[Example]) -> list[str]:
    routes = ["examples/"]
    (out / "examples").mkdir(parents=True, exist_ok=True)
    (out / "examples" / "index.html").write_text(gallery(examples), encoding="utf-8")
    for i, ex in enumerate(examples):
        prev = examples[i - 1] if i else None
        nxt = examples[i + 1] if i + 1 < len(examples) else None
        dest = out / "examples" / ex.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "index.html").write_text(example_page(ex, prev, nxt), encoding="utf-8")
        routes.append(f"examples/{ex.name}/")
        for r in ex.runs:
            if r.report:
                target = out / r.report
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / r.report, target)
    return routes
