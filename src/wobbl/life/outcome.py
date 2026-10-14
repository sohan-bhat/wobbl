"""The answer object of buckling, vibration and fatigue results: the shared generic outcome with life's own report.

The verdicts are the strength ones (holds, marginal, fails), each domain gets its own report title, and the work
folder is removed after a run unless the caller keeps it, since the solver files of a modal run are large and
nothing reads them afterwards.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..outcome_generic import GenericOutcome

TITLES = {"buckling": "Buckling report", "vibration": "Vibration report", "fatigue": "Fatigue report"}
VERDICT_CLASS = {"holds": "pass", "marginal": "warn", "fails": "fail"}


def verdict_for(factor: float, target: float) -> str:
    if factor >= target:
        return "holds"
    if factor >= 1.0:
        return "marginal"
    return "fails"


class LifeOutcome(GenericOutcome):
    def write(self, out_dir: str | Path, sections_extra: list[str] | None = None) -> "LifeOutcome":
        from ..report.html import badge, esc, figure, page, table

        out = Path(out_dir)
        sections = []
        for key, cap in self.values.get("_figures", []):
            if self.images.get(key):
                sections.append(figure(self.images[key], cap))
        sections.append("<h2>Checks</h2>" + table(["status", "check", "result"],
                                                  [[badge(c.status), c.name, c.summary] for c in self.checks]))
        if self.fixes:
            sections.append("<h2>How to fix it</h2>" + "".join(
                f"<div class='fix'><div class='t'>{esc(f['title'])}</div><div>{esc(f['summary'])}</div></div>"
                for f in self.fixes))
        # tables travel as data (title, headers, rows, note) so a result read back from outcome.json rebuilds them
        sections += [html_table(*t) for t in self.values.get("_tables", [])] + (sections_extra or [])
        sections.append("<h2>Assumptions</h2><ol class='assume'>" +
                        "".join(f"<li>{esc(a)}</li>" for a in self.assumptions) + "</ol>")
        if self.notes:
            sections.append("<h2>Notes</h2><ul class='plain'>" + "".join(f"<li>{esc(n)}</li>" for n in self.notes)
                            + "</ul>")
        html = page(TITLES.get(self.domain, "Report"), Path(self.part).name if self.part else self.domain,
                    self.question, self.verdict.upper(), VERDICT_CLASS.get(self.verdict, "warn"), self.headline,
                    self.metrics, self.confidence, self.confidence_reasons, sections,
                    f"Solver {self.solver}. Total time {self.t_total:.0f} s.")
        self.report = str(out / "report.html")
        self.workdir = str(out)
        Path(self.report).write_text(html, encoding="utf-8")
        (out / "summary.txt").write_text(self.text() + "\n", encoding="utf-8")
        self.save(out / "outcome.json")
        return self


def html_table(title: str, headers: list[str], rows: list[list], note: str = "") -> str:
    from ..report.html import esc, table

    return f"<h2>{esc(title)}</h2>" + (f"<p class='muted'>{esc(note)}</p>" if note else "") + table(headers, rows)


def clean_work(work: Path, keep: bool):
    """Remove the solver scratch of a finished run unless it is wanted for a closer look."""
    if not keep and work.exists():
        shutil.rmtree(work, ignore_errors=True)
