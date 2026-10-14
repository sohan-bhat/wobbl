"""Any Wobbl run, of any domain, as one thing the cost model, the jobs and the remote machines understand.

A workload is a Wobbl command line. The CLI itself parses it without running it, which gives its command, its
parameters and the files it reads. Domains plug in through one registry, keyed by the command a user types:

    register("run", domain="strength", solves=1.0, refine=run_submodel)
    register("flow pipe", domain="flow", cost=my_cost_function)

``cost`` predicts a run outright. ``solves`` says that the command meshes a CAD part and how many static solves
of that mesh its procedure costs, and the shared CalculiX model prices it from the part's first mesh. ``refine``
takes a finished result and refines it locally. A command with none of these is still predicted, from the
measured runs of the same command on this machine, and before any from the run of its bundled example on the
reference machine.

Every finished run feeds that history. The two outcome writers every domain uses call ``record_outcome``, which
keeps the command, its domain, its wall time, its peak memory and a signature of its inputs, so running the same
study again is predicted from its own last run.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

CAD = {".step", ".stp", ".iges", ".igs", ".brep", ".stl", ".py"}
NOT_SIZE = {"out", "json", "open_report", "no_render", "render", "quiet", "verbose"}

# seconds of each bundled example on the reference machine, for commands with nothing measured here yet
PRIORS = {"runaway": 10.3, "mechanism": 26.0, "motion": 4.0, "motor": 24.2, "fatigue": 7.8, "flame": 28.1}


@dataclass
class Hooks:
    domain: str = ""
    cost: Callable | None = None       # (Workload) -> estimate.Cost
    solves: float | None = None        # static solves per mesh, for a command that meshes a CAD part
    dof: int = 3                       # 3 for displacements, 1 for temperatures
    refine: Callable | None = None     # (result_dir, at, radius, size, out, progress) -> dict


REGISTRY: dict[str, Hooks] = {}


def register(command: str, **hooks) -> Hooks:
    """Plug a domain's command into the cost model, the jobs and the refinements. Later calls update earlier."""
    h = REGISTRY.setdefault(command, Hooks())
    for k, v in hooks.items():
        if not hasattr(h, k):
            raise TypeError(f"unknown hook {k}")
        setattr(h, k, v)
    return h


def refiner(domain: str) -> Callable | None:
    """The refinement registered for a domain's results, if any."""
    for h in REGISTRY.values():
        if h.domain == domain and h.refine is not None:
            return h.refine
    return None


def refine(result_dir: str | Path, at=None, radius: float | None = None, size: float | None = None,
           out: str | Path | None = None, progress: Callable | None = None) -> dict:
    """Refine a finished result of any domain locally, with the refinement its domain registered."""
    d = Path(result_dir)
    try:
        domain = json.loads((d / "outcome.json").read_text()).get("domain", "")
    except (OSError, ValueError):
        raise WorkloadError(f"no Wobbl result in {d}", "Pass the folder a Wobbl run printed") from None
    fn = refiner(domain)
    if fn is None:
        meshed = sorted({h.domain for h in REGISTRY.values() if h.refine is not None})
        raise WorkloadError(f"{domain or 'this'} results have no local refinement, which needs a meshed field with "
                            "a local peak", "Results that have one: " + ", ".join(meshed))
    return fn(d, at, radius, size, out, progress)


@dataclass
class Workload:
    argv: list[str]                   # the command line after 'wobbl'
    command: str                      # 'run', 'flow pipe'
    params: dict = field(default_factory=dict)
    inputs: list[Path] = field(default_factory=list)
    out: str | None = None            # the result folder it was told to use, if any
    tokens: list[str] = field(default_factory=list)   # each input as it was typed, in the order of inputs

    @property
    def hooks(self) -> Hooks:
        return REGISTRY.get(self.command) or Hooks()

    @property
    def domain(self) -> str:
        return self.hooks.domain or self.command

    @property
    def part(self) -> Path | None:
        p = self.params.get("part")
        return Path(p) if isinstance(p, str) and Path(p).suffix.lower() in CAD and Path(p).exists() else None

    def signature(self) -> str:
        """Same command, same size settings and the same input files give the same signature."""
        typed = set(self.tokens)

        def norm(v):   # an input counts by its file below, not by how its path was typed
            if isinstance(v, (list, tuple)):
                return [norm(x) for x in v]
            return Path(str(v)).name if str(v) in typed else v
        items = {k: norm(v) for k, v in sorted(self.params.items()) if k not in NOT_SIZE}
        files = []
        for p in self.inputs:
            try:
                st = p.stat()
                files.append([p.name, st.st_size, int(st.st_mtime)])
            except OSError:
                continue
        blob = json.dumps([self.command, items, files], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class WorkloadError(ValueError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def is_command(token: str) -> bool:
    from ..cli.main import cli

    return token in cli.commands


def parse(argv: list[str]) -> Workload:
    """Read a Wobbl command line with the CLI's own parser, running nothing."""
    import click

    from ..cli.main import cli

    rest, path, cmd = list(argv), [], cli
    while isinstance(cmd, click.Group) and rest and rest[0] in cmd.commands:
        path.append(rest[0])
        cmd = cmd.commands[rest.pop(0)]
    if not path or isinstance(cmd, click.Group):
        raise WorkloadError(f"'{' '.join(argv[:2])}' is not a Wobbl command",
                            "Give a command as you would type it after wobbl, for example run part.step -m pla")
    try:
        ctx = cmd.make_context(" ".join(path), rest, resilient_parsing=True)
    except click.ClickException as e:
        raise WorkloadError(f"wobbl {' '.join(path)} cannot read these arguments ({e.format_message()})") from None
    params = dict(ctx.params)
    inputs, tokens, seen = [], [], set()

    def add(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                add(x)
            return
        v = str(v) if isinstance(v, Path) else v
        if isinstance(v, str) and v not in (".", "..") and len(v) < 4096:
            p = Path(v).expanduser()
            try:
                if p.exists() and p.resolve() not in seen:
                    seen.add(p.resolve())
                    inputs.append(p)
                    tokens.append(v)
            except OSError:
                pass
    from click.core import ParameterSource

    for k, v in params.items():
        if k not in NOT_SIZE and ctx.get_parameter_source(k) == ParameterSource.COMMANDLINE:
            add(v)
    out = params.get("out")
    return Workload(list(argv), " ".join(path), params, inputs, str(out) if out else None, tokens)


def rewrite(argv: list[str], mapping: dict[str, str], out: str | None = None) -> list[str]:
    """A command line with each token in ``mapping`` replaced, as a value or as --option=value, and its result
    folder set to ``out`` when given."""
    res, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if out is not None and a in ("-o", "--out") and i + 1 < len(argv):
            skip = True
            continue
        if out is not None and a.startswith("--out="):
            continue
        if "=" in a and a.startswith("--"):
            k, v = a.split("=", 1)
            res.append(f"{k}={mapping.get(v, v)}")
        else:
            res.append(mapping.get(a, a))
    return res + (["-o", out] if out is not None else [])


def absolute(w: Workload, out: str) -> list[str]:
    """The command line with every input and the result folder as absolute paths, so it runs the same from any
    folder, and later from another one when it is resumed."""
    return rewrite(w.argv, {t: str(p.resolve()) for t, p in zip(w.tokens, w.inputs)},
                   str(Path(w.out).expanduser().resolve()) if w.out else out)


# ------------------------------------------------------------------------------------------------- history
def record_outcome(outcome: dict, out_dir: str | Path | None = None):
    """Keep a finished run in this machine's history. Only runs from the Wobbl CLI or its jobs are kept, so test
    suites and scripts that write outcomes never touch it."""
    if not from_cli():
        return
    try:
        _record(outcome, out_dir)
    except Exception:  # noqa: BLE001  (history must never break a finished run)
        pass


def from_cli() -> bool:
    """Whether this process is the Wobbl CLI, run as wobbl or as python -m wobbl."""
    if not sys.argv or not sys.argv[0]:
        return False
    a = Path(sys.argv[0])
    if a.name in ("wobbl", "wobbl.exe", "wobbl-script.py"):
        return True
    try:
        return a.resolve() == (Path(__file__).resolve().parents[1] / "__main__.py")
    except OSError:
        return False


def _peak_rss() -> float:
    try:
        import resource

        k = 1.0 if sys.platform == "darwin" else 1024.0
        return k * max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except Exception:  # noqa: BLE001
        return 0.0


def _record(outcome: dict, out_dir):
    from .. import estimate as E

    argv = sys.argv[1:]
    try:
        w = parse(argv)
        command, sig = w.command, w.signature()
    except Exception:  # noqa: BLE001  (an MCP or scripted run has no command line, so its domain stands in)
        command, sig = outcome.get("domain", ""), ""
    wall = float(outcome.get("t_total") or 0.0)
    if wall <= 0:
        return
    disk = 0.0
    if out_dir:
        disk = float(sum(p.stat().st_size for p in Path(out_dir).rglob("*") if p.is_file()))
    cal = E.load_calibration()
    if any(r.get("size", {}).get("signature") == sig and r.get("measured_time") == wall
           for r in cal.get("runs", [])[-20:]):
        return                                   # the same result saved again, after a fix or a new report
    pred = cal.get("predictions", {}).get(sig, {}) if sig else {}
    E.record(f"wobbl {command}", pred.get("time", 0.0), wall, pred.get("memory", 0.0), _peak_rss(),
             {"domain": outcome.get("domain", ""), "signature": sig, "disk": disk}, "run")


def remember_prediction(w: Workload, seconds: float, peak_bytes: float):
    """Keep what an estimate said, so the run that follows can be scored against it."""
    from .. import estimate as E

    cal = E.load_calibration()
    preds = cal.setdefault("predictions", {})
    preds[w.signature()] = {"time": seconds, "memory": peak_bytes, "when": time.time()}
    cal["predictions"] = dict(sorted(preds.items(), key=lambda kv: kv[1]["when"])[-100:])
    E.save_calibration(cal)


def history(w: Workload) -> tuple[list[dict], bool]:
    """Measured runs of this command here, newest last, and whether they are of these very inputs."""
    from .. import estimate as E

    runs = [r for r in E.load_calibration().get("runs", []) if r.get("solver") == f"wobbl {w.command}"]
    same = [r for r in runs if r.get("size", {}).get("signature") == w.signature()]
    return (same, True) if same else (runs, False)


# ---------------------------------------------------------------------------------------------------- cost
def cost(w: Workload):
    """What a workload costs on one core of the reference machine: from its domain's hook, from the part it
    meshes, from its own history here, or from its bundled example."""
    from .. import estimate as E

    h = w.hooks
    runs, exact = history(w)
    solver = f"wobbl {w.command}"
    if exact:
        same = runs[-5:]
        t = float(np.median([r["measured_time"] for r in same]))
        mem = max(r.get("measured_memory", 0.0) for r in same)
        disk = max(r.get("size", {}).get("disk", 0.0) for r in same)
        return E.Cost(solver, w.domain, t, 0.0, mem, disk, {"runs": len(runs)},
                      [f"You ran this very {w.domain} study here before, and it took {E.duration(t)}."], known=True,
                      measured=True)
    if h.cost is not None:
        c = h.cost(w)
        c.solver = solver
        return c
    if h.solves is not None and w.part is not None:
        return part_cost(w, h)
    if runs:
        times = [r["measured_time"] for r in runs[-12:]]
        t = float(math.exp(np.median(np.log(times))))
        mem = max(r.get("measured_memory", 0.0) for r in runs[-12:])
        disk = max(r.get("size", {}).get("disk", 0.0) for r in runs[-12:])
        lo, hi = min(times), max(times)
        return E.Cost(solver, w.domain, t, 0.0, mem, disk, {"runs": len(runs)},
                      [f"From {len(runs)} earlier wobbl {w.command} run{'s' if len(runs) != 1 else ''} here, which "
                       f"took from {E.duration(lo)} to {E.duration(hi)}."], measured=True)
    if w.command.split()[0] in PRIORS:
        t = PRIORS[w.command.split()[0]]
        return E.Cost(solver, w.domain, t, 0.0, 5e8, 5e6, {},
                      [f"From the bundled {w.command.split()[0]} example, which took {E.duration(t)} on "
                       f"{E.REFERENCE}. The first run here measures it."])
    return E.Cost(solver, w.domain, 0.0, 0.0, 0.0, 0.0, {},
                  [f"Nothing has measured wobbl {w.command} on this machine yet, so its time is not known. Its first "
                   "run, here or as a job, measures it."], known=False)


def part_cost(w: Workload, h: Hooks):
    """A command that meshes a CAD part, priced from the part's first mesh and its procedure's solves."""
    import tempfile

    from .. import estimate as E
    from ..structural.run import describe_part

    p = w.params
    with tempfile.TemporaryDirectory() as tmp:
        prep, feats = describe_part(str(w.part), tmp, float(p.get("scale") or 1.0))
        domain = "strength" if h.dof == 3 else "heat"
        mesh = E.first_mesh(prep, feats, domain, str(p.get("quality") or "normal"), p.get("mesh_size"),
                            bool(p.get("duration")))
    c = E.study_cost(mesh, domain, str(p.get("quality") or "normal"), bool(p.get("duration")),
                     float(p.get("budget") or 300.0))
    c.core_seconds *= h.solves
    c.solver, c.procedure = f"wobbl {w.command}", w.domain
    if h.solves != 1.0:
        c.basis.insert(1, f"A {w.domain} study costs about {h.solves:g} static solves of each mesh, which its first "
                          "measured run here corrects.")
    return c


def _builtin():
    from ..structural.submodel import run_submodel

    register("run", domain="strength", solves=1.0, refine=run_submodel)
    register("thermal", domain="thermal", solves=1.0, dof=1)
    register("buckling", domain="buckling", solves=2.5)
    register("vibration", domain="vibration", solves=2.0)
    register("plastic", domain="plastic", solves=25.0)


_builtin()
