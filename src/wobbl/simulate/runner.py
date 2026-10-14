"""Run the analyses a plan picked, one after another, and keep what each assumed and how to override it.

Every analysis writes request.json next to its outcome.json. It holds the domain, the analysis and the spec it ran
from, with every value the run used, guessed ones included, so the run can be repeated with nothing guessed:
registry.get(domain, analysis).run(spec, out_dir, quick=False), or registry.rerun(result_dir, out_dir, changes).
"""
from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .registry import Analysis


def _quote(v: str) -> str:
    return shlex.quote(str(v))


@dataclass
class Job:
    """What an analysis entry gets: the thing, the question, the options and a place to write."""
    analysis: Analysis
    thing: str
    kind: str
    question: str
    options: dict[str, list[str]]
    out: Path
    quick: bool = False
    earlier: dict[str, Any] = field(default_factory=dict)    # outcomes of the analyses run before, by name
    log: Callable[[str, str], None] = lambda stage, msg: None
    assumptions: list[tuple[str, str]] = field(default_factory=list)   # (what was assumed, how to override it)
    used: dict[str, list[str]] = field(default_factory=dict)          # every value the run used, for request.json
    criteria: list[dict] = field(default_factory=list)
    situation: dict = field(default_factory=dict)    # supports, loads, environment and objects, for the 3D view

    @property
    def path(self) -> Path | None:
        """The thing as a file or folder, when it is one."""
        p = Path(self.thing).expanduser() if self.thing else None
        return p.resolve() if p is not None and p.exists() else None

    def value(self, name: str, default: str | None = None, why: str = "") -> str | None:
        """The option as given, or the default with the guess recorded. why reads like 'drop height not given'."""
        if self.options.get(name):
            v = self.options[name][-1]
        elif default is None:
            return None
        else:
            v = default
            self.assume(f"{why or name + ' not given'}: assumed {default}", f"--{name} {_quote(default)}")
        self.used[name] = [v]
        return v

    def values(self, name: str, default: list[str] | None = None, why: str = "") -> list[str]:
        """Every value of a repeatable option, or the defaults with the guess recorded."""
        vs = list(self.options.get(name) or [])
        if not vs and default:
            vs = list(default)
            self.assume(f"{why or name + ' not given'}: assumed {', '.join(vs)}",
                        " ".join(f"--{name} {_quote(v)}" for v in vs))
        if vs:
            self.used[name] = vs
        return vs

    def flag(self, name: str) -> bool:
        on = bool(self.options.get(name)) and str(self.options[name][-1]).lower() not in ("0", "false", "no", "off")
        if on:
            self.used[name] = ["true"]
        return on

    def number(self, name: str, default: float | None = None) -> float | None:
        v = self.value(name)
        return float(v) if v not in (None, "") else default

    def assume(self, text: str, override: str = "") -> None:
        """Record a guess and the exact option that would replace it."""
        self.assumptions.append((text, override))
        self.log("assume", f"{text}" + (f" (override with {override})" if override else ""))

    def use(self, name: str, value: str | list[str]) -> None:
        """Record a value the run settled on some other way, such as from the question's words."""
        self.used[name] = list(value) if isinstance(value, (list, tuple)) else [str(value)]

    def criterion(self, name: str, value: str, limit: float, sense: str = "max") -> None:
        """A pass or fail line for uncertainty work: values[value] must stay below (max) or above (min) limit."""
        self.criteria.append({"name": name, "value": value, "limit": float(limit), "sense": sense})

    def situate(self, **parts) -> None:
        """Describe the scenario for the view: supports=[...], loads=[...], environment={...}, objects=[...] and
        anything else worth drawing. Lists add up and dicts merge across calls."""
        for k, v in parts.items():
            if v in (None, "", [], {}):
                continue
            if isinstance(v, dict):
                self.situation.setdefault(k, {}).update(v)
            elif isinstance(v, (list, tuple)):
                have = self.situation.setdefault(k, [])
                have += [x for x in v if x not in have]
            else:
                self.situation[k] = v

    def spec(self) -> dict:
        opts = {k: list(v) for k, v in self.options.items()}
        opts.update(self.used)
        return {"thing": str(self.path or self.thing), "kind": self.kind, "question": self.question,
                "options": opts}


@dataclass
class Step:
    analysis: Analysis
    job: Job
    outcome: Any = None
    error: str = ""
    hint: str = ""
    seconds: float = 0.0

    def get(self, key: str, default=None):
        o = self.outcome
        if o is None:
            return default
        return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)

    def summary(self) -> dict:
        d = {"analysis": self.analysis.name, "domain": self.analysis.domain}
        if self.error:
            return d | {"error": self.error, **({"hint": self.hint} if self.hint else {})}
        return d | {"verdict": self.get("verdict", ""), "headline": self.get("headline", ""),
                    "confidence": self.get("confidence", ""), "result_dir": str(self.get("workdir", "") or
                                                                                 self.job.out),
                    "report": self.get("report", ""),
                    "assumptions": [{"assumed": t, "override": o} for t, o in self.job.assumptions]}


GOOD = {"ok", "holds", "pass", "passes", "fine", "safe", "stable", "lifts", "clears"}
FAIL = {"fails", "fail", "hot", "collapses", "slips", "separates", "tips", "stalls", "unsafe", "runaway", "binds",
        "breaks", "cracks"}


def grade(verdict: str) -> str:
    """pass, marginal or fail, whatever words a domain uses for its verdict."""
    v = (verdict or "").lower()
    return "pass" if v in GOOD else "fail" if v in FAIL else "marginal" if v else "unknown"


def _outcome_path(o: Any, out: Path) -> Path:
    wd = (o.get("workdir") if isinstance(o, dict) else getattr(o, "workdir", "")) or out
    return Path(wd) / "outcome.json"


def _fold(job: Job, o: Any) -> None:
    """Put the guesses at the top of the outcome's assumptions and write its files again."""
    if not job.assumptions or isinstance(o, dict) or not hasattr(o, "assumptions"):
        return
    texts = [f"Wobbl guessed: {t}." for t, _ in job.assumptions]
    if all(t in o.assumptions for t in texts):
        return
    o.assumptions = texts + [a for a in o.assumptions if a not in texts]
    wd = Path(getattr(o, "workdir", "") or job.out)
    if getattr(o, "domain", "") == "strength":
        from ..structural.report import write_outputs
        write_outputs(o, wd)
    elif hasattr(o, "write"):
        o.write(wd)


def _finish(job: Job, o: Any) -> dict:
    """Write request.json and the criteria next to outcome.json, and return outcome.json's content."""
    from .. import __version__
    path = _outcome_path(o, job.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    req = {"domain": job.analysis.domain, "analysis": job.analysis.name, "spec": job.spec(), "wobbl": __version__}
    (path.parent / "request.json").write_text(json.dumps(req, indent=1))
    if path.exists():
        d = json.loads(path.read_text())
        if job.criteria and not d.get("criteria"):
            d["criteria"] = job.criteria
            path.write_text(json.dumps(d, indent=1))
        return d
    return o if isinstance(o, dict) else o.to_dict()


def run_job(job: Job) -> Any:
    o = job.analysis.function()(job)
    _fold(job, o)
    _finish(job, o)
    return o


def run_spec(analysis: Analysis, spec: dict, out: Path, quick: bool = False) -> dict:
    """The rerun contract: the same spec gives the same study, with nothing guessed that the spec holds."""
    opts = {k: (list(v) if isinstance(v, (list, tuple)) else [str(v)]) for k, v in spec.get("options", {}).items()}
    job = Job(analysis, str(spec.get("thing", "")), str(spec.get("kind", "")), str(spec.get("question", "")), opts,
              Path(out), quick)
    o = analysis.function()(job)
    _fold(job, o)
    return _finish(job, o)


def run_plan(plan, options: dict[str, list[str]], out: Path, log=None, quick: bool = False) -> list[Step]:
    """Run each analysis of a plan. With more than one, each writes into its own folder under out, and a later
    one can read an earlier one's outcome through job.earlier, as fatigue reads the stress result. Every run ends
    with simulation.json in out, which the 3D view reads."""
    log = log or (lambda stage, msg: None)
    many = len(plan.analyses) > 1
    earlier: dict[str, Any] = {}
    steps = []
    for a in plan.analyses:
        job = Job(a, plan.thing, plan.kind, plan.question, options, out / a.name if many else out, quick, earlier,
                  log)
        job.situate(objects=objects_named(plan.question))
        step = Step(a, job)
        t0 = time.time()
        try:
            if many:
                log("physics", f"{a.name}: {a.summary[:70]}")
            step.outcome = run_job(job)
            earlier[a.name] = step.outcome
        except Exception as e:  # noqa: BLE001  one failed analysis must not lose the others
            if not many:
                raise
            step.error, step.hint = str(e).strip("'\""), getattr(e, "hint", "")
        step.seconds = time.time() - t0
        steps.append(step)
    write_summary(plan, steps, out)
    return steps


SCENE_WORDS = ("wall", "shelf", "floor", "ground", "table", "desk", "bench", "ceiling", "beam", "breadboard", "pad",
               "concrete", "tile", "carpet", "grass", "water", "sea", "air", "box", "enclosure", "tripod", "car",
               "bike", "drone", "hand", "pocket", "bag", "rail", "launch rail", "pipe", "room", "board", "case")


def objects_named(question: str) -> list[str]:
    """The surroundings a question names, like the wall and the shelf, so the view can draw them."""
    import re
    q = (question or "").lower()
    return [w for w in SCENE_WORDS if re.search(rf"\b{re.escape(w)}s?\b", q)]


def overall(steps: list[Step]) -> str:
    grades = [grade(s.get("verdict", "")) if not s.error else "fail" for s in steps]
    for g in ("fail", "marginal", "unknown"):
        if g in grades:
            return g
    return "pass"


def situation(steps: list[Step]) -> dict:
    out: dict = {}
    for s in steps:
        for k, v in s.job.situation.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            elif isinstance(v, list):
                have = out.setdefault(k, [])
                have += [x for x in v if x not in have]
            else:
                out[k] = v
    return out


def write_summary(plan, steps: list[Step], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    d = {"thing": plan.thing, "question": plan.question, "plan": plan.to_dict(), "verdict": overall(steps),
         "situation": situation(steps), "steps": [s.summary() for s in steps]}
    p = out / "simulation.json"
    p.write_text(json.dumps(d, indent=1, default=str))
    return p


def view(out: Path) -> Path | None:
    """The interactive 3D view of a finished simulate run, from wobbl.viz when it is installed. It reads the
    simulation.json in out and returns the path of the page it wrote."""
    try:
        from ..viz import simulation_view
    except ImportError:
        return None
    try:
        p = simulation_view(Path(out) / "simulation.json")
    except Exception:  # noqa: BLE001  a view that fails must not lose the result
        return None
    return Path(p) if p else None
