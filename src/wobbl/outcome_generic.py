"""Answer object shared by the heat, circuit and motion domains (strength has its own richer one)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .verify.checks import Check, _jsonable


@dataclass
class GenericOutcome:
    domain: str                       # thermal, circuit, motion
    part: str
    question: str
    verdict: str                      # ok, warm, hot / ok, fails / ok, marginal, fails
    headline: str
    metrics: list[tuple[str, str]]
    confidence: str
    confidence_reasons: list[str]
    checks: list[Check]
    assumptions: list[str]
    values: dict = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    fixes: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    report: str = ""
    workdir: str = ""
    t_total: float = 0.0
    solver: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return _jsonable(d)

    def save(self, path: str | Path):
        from .verify.checks import relative_paths
        Path(path).write_text(json.dumps(relative_paths(self.to_dict(), Path(path).parent), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "GenericOutcome":
        from .verify.checks import absolute_paths
        d = absolute_paths(json.loads(Path(path).read_text()), Path(path).parent)
        d["checks"] = [Check(**c) for c in d["checks"]]
        d["metrics"] = [tuple(m) for m in d["metrics"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def text(self) -> str:
        lines = [f"{self.verdict.upper()}. {self.headline}", "",
                 f"Confidence: {self.confidence}. " + "; ".join(self.confidence_reasons[:4])]
        lines += [f"{k}: {v}" for k, v in self.metrics]
        lines += ["", "Checks:"] + [f"  [{c.status}] {c.name}: {c.summary}" for c in self.checks]
        lines += ["", "Assumptions:"] + [f"  - {a}" for a in self.assumptions]
        if self.fixes:
            lines += ["", "Suggested fixes:"] + [f"  - {f['title']}: {f['summary']}" for f in self.fixes]
        from .verify.checks import display_path
        if self.images:
            lines += ["", "Images: " + ", ".join(f"{k}: {display_path(v, self.workdir)}"
                                                for k, v in self.images.items())]
        if self.report:
            lines.append(f"Report: {display_path(self.report, self.workdir)}")
        return "\n".join(lines)

    def write(self, out_dir: str | Path, sections_extra: list[str] | None = None) -> "GenericOutcome":
        from .credibility.report import evidence_sections, evidence_text
        from .report.html import badge, esc, figure, page, table

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
        sections += sections_extra or []
        sections += evidence_sections(out)
        sections.append("<h2>Assumptions</h2><ol class='assume'>" +
                        "".join(f"<li>{esc(a)}</li>" for a in self.assumptions) + "</ol>")
        word = self.verdict.upper()
        cls = {"ok": "pass", "warm": "warn", "hot": "fail", "fails": "fail"}.get(self.verdict, "warn")
        title = {"thermal": "Heat report", "circuit": "Circuit report", "motion": "Motion report"}.get(self.domain,
                                                                                                    "Report")
        html = page(title, Path(self.part).name if self.part else self.domain, self.question, word, cls,
                    self.headline, self.metrics, self.confidence, self.confidence_reasons, sections,
                    f"Solver {self.solver}. Total time {self.t_total:.0f} s.")
        self.report = str(out / "report.html")
        self.workdir = str(out)
        Path(self.report).write_text(html, encoding="utf-8")
        (out / "summary.txt").write_text(self.text() + evidence_text(out) + "\n", encoding="utf-8")
        self.workdir = str(out)
        self.save(out / "outcome.json")
        return self
