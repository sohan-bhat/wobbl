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


class PlanOnly(Exception):
    """Raised by job.ready() in a --plan preview, where the analysis stops before it solves."""


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
    part: str = ""             # the part of the question this analysis answers, when it asks several things
    dry: bool = False          # a --plan preview: ready() stops the entry before it solves

    @property
    def focus(self) -> str:
        """The words this analysis should read its own loads and values from. A question that asks for a drop
        and a separate static load gives each analysis only its own clause, with the shared context."""
        return self.part or self.question

    def ready(self) -> None:
        """Call right before the solve, once every input is read and every guess recorded. Under --plan the run
        stops here, so the plan can show what the run would assume without running it."""
        if self.dry:
            raise PlanOnly(self.analysis.key)

    @property
    def path(self) -> Path | None:
        """The thing as a file or folder, when it is one."""
        p = Path(self.thing).expanduser() if self.thing else None
        return p.resolve() if p is not None and p.exists() else None

    @property
    def text(self) -> str:
        """Every word the user gave: the thing when it is a description or a name, and the question."""
        return " ".join(x for x in ("" if self.path else self.thing, self.question) if x and x != "-")

    def value(self, name: str, default: str | None = None, why: str = "") -> str | None:
        """The option as given, or the default with the guess recorded. why reads like 'drop height not given'."""
        if self.options.get(name):
            v = self.options[name][-1]
        elif default is None:
            return None
        else:
            v = default
            why = why or f"{name} not given"
            self.assume(f"{why} ({default})" if " so " in why else f"{why}, so Wobbl assumed {default}",
                        f"--{name} {_quote(default)}")
        self.used[name] = [v]
        return v

    def values(self, name: str, default: list[str] | None = None, why: str = "") -> list[str]:
        """Every value of a repeatable option, or the defaults with the guess recorded."""
        vs = list(self.options.get(name) or [])
        if not vs and default:
            vs = list(default)
            self.assume(f"{why or name + ' not given'}, so Wobbl assumed {', '.join(vs)}",
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

    def readings(self, given: dict | None = None) -> list[tuple[str, list[str]]]:
        """Values the run took from the question's words rather than from an option or a guess, such as a drop
        height of 1 m, so a plan can show them with their override too."""
        import re
        guessed = {k for _, o in self.assumptions for k in re.findall(r"--([\w-]+)", o)}
        given = self.options if given is None else given
        return [(k, v) for k, v in self.used.items() if k not in guessed and k not in given]

    def spec(self) -> dict:
        opts = {k: list(v) for k, v in self.options.items()}
        opts.update(self.used)
        return {"thing": str(self.path or self.thing), "kind": self.kind, "question": self.question,
                **({"part": self.part} if self.part else {}), "options": opts}


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
              Path(out), quick, part=str(spec.get("part", "")))
    o = analysis.function()(job)
    _fold(job, o)
    return _finish(job, o)


def settle_options(plan, opts: dict[str, list[str]]) -> list[str]:
    """Fit the given options to the plan in place, and return the names no analysis of it reads. A plan that
    reads a 'set' option, as circuits do for part values like R1.power=1W, takes the other names as key=value
    there, so --set R1.power=1W and --R1.power 1W both reach it."""
    takes = {o.name for a in plan.analyses for o in a.all_options()}
    if "*" in takes:
        return []
    bad = [k for k in opts if k not in takes]
    if bad and "set" in takes:
        for k in bad:
            opts.setdefault("set", []).extend(f"{k}={v}" for v in opts.pop(k))
        return []
    return bad


def _jobs(plan, options: dict[str, list[str]], out: Path, log, quick: bool, dry: bool):
    """Each analysis's job in turn. A value a part study settles, such as the material or the supports, carries to
    the analyses after it, so a guess is made once for the whole run."""
    from .registry import domain
    many = len(plan.analyses) > 1
    earlier: dict[str, Any] = {}
    shared = {o.name for o in domain(plan.domain).options}
    carry = {k: list(v) for k, v in options.items()}
    focus = getattr(plan, "focus", {}) or {}
    for a in plan.analyses:
        job = Job(a, plan.thing, plan.kind, plan.question, dict(carry), out / a.name if many else out, quick,
                  earlier, log, part=focus.get(a.name, ""), dry=dry)
        job.situate(objects=objects_named(plan.question))
        step = Step(a, job)
        yield step
        if step.outcome is not None:
            earlier[a.name] = step.outcome
        carry.update({k: v for k, v in job.used.items() if k in shared and k not in carry})


def run_plan(plan, options: dict[str, list[str]], out: Path, log=None, quick: bool = False) -> list[Step]:
    """Run each analysis of a plan. With more than one, each writes into its own folder under out, and a later
    one can read an earlier one's outcome through job.earlier, as fatigue reads the stress result. Every run ends
    with simulation.json in out, which the 3D view reads."""
    log = log or (lambda stage, msg: None)
    many = len(plan.analyses) > 1
    steps = []
    for step in _jobs(plan, options, out, log, quick, False):
        t0 = time.time()
        try:
            if many:
                log("physics", f"{step.analysis.name}: {step.analysis.summary[:70]}")
            step.outcome = run_job(step.job)
        except Exception as e:  # noqa: BLE001  one failed analysis must not lose the others
            if not many:
                raise
            step.error, step.hint = str(e).strip("'\""), getattr(e, "hint", "")
        step.seconds = time.time() - t0
        steps.append(step)
    write_summary(plan, steps, out)
    return steps


def preview(plan, options: dict[str, list[str]]) -> list[Step]:
    """What each analysis would assume, read without solving. An analysis that calls job.ready() before its
    solve stops there; one registered without dry=True is not started, and its guesses show when it runs."""
    import tempfile
    steps = []
    with tempfile.TemporaryDirectory(prefix="wobbl-plan-") as tmp:
        for step in _jobs(plan, options, Path(tmp), lambda stage, msg: None, True, True):
            steps.append(step)
            if not step.analysis.dry:
                step.hint = "its guesses show when it runs"
                continue
            try:
                step.analysis.function()(step.job)
                step.hint = "it ran in full, since it never called job.ready()"
            except PlanOnly:
                pass
            except Exception as e:  # noqa: BLE001  a plan should show every problem, not stop at the first
                step.error, step.hint = str(e).strip("'\""), getattr(e, "hint", "")
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
