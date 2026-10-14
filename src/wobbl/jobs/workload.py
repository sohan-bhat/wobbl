"""Any Wobbl run, of any domain, as one thing the cost model, the jobs and the remote machines understand.

A workload is a Wobbl command line. The CLI itself parses it without running it, which gives its command, its
parameters and the files it reads. Domains plug in through one registry, keyed by the command a user types:

    register("run", domain="strength", solves=1.0)
    register("flow pipe", domain="flow", cost=my_cost_function)

``cost`` predicts a run outright. ``solves`` says that the command meshes a CAD part and how many static solves
of that mesh its procedure costs, and the shared CalculiX model prices it from the part's first mesh. A command with none of these is still predicted, from the
measured runs of the same command on this machine, and before any from the run of its bundled example on the
reference machine.

Every finished run feeds that history. The two outcome writers every domain uses call ``record_outcome``, which
keeps the command, its domain, its wall time, its peak memory and a signature of its inputs, so running the same
study again is predicted from its own last run.
"""
from __future__ import annotations

import copy
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


REGISTRY: dict[str, Hooks] = {}


def register(command: str, **hooks) -> Hooks:
    """Plug a domain's command into the cost model and the jobs. Later calls update earlier."""
    h = REGISTRY.setdefault(command, Hooks())
    for k, v in hooks.items():
        if not hasattr(h, k):
            raise TypeError(f"unknown hook {k}")
        setattr(h, k, v)
    return h


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
    # only a result folder that was typed counts, a default one belongs to wherever the run is started from
    out = params.get("out") if "out" in params and ctx.get_parameter_source("out") == ParameterSource.COMMANDLINE \
        else None
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


PLAN_COMMANDS = ("simulate", "jobs run-plan")
SIZE_OPTIONS = ("quality", "mesh-size", "scale", "budget", "duration", "modes", "shell", "body")
_pending: list[tuple[dict, Path | None, float, bool]] = []


def _record(outcome: dict, out_dir):
    """Record a run now when its command line says what it was. A simulate run says so in the request.json the
    runner writes just after the outcome, and a run from the MCP server has no command line at all, so those are
    kept until the next save or the end of the process."""
    try:
        w = parse(sys.argv[1:])
    except Exception:  # noqa: BLE001
        w = None
    if w is None or w.command in PLAN_COMMANDS or "out" not in w.params:
        if not _pending:
            import atexit
            atexit.register(_flush)
        _flush()
        _pending.append((outcome, Path(out_dir) if out_dir else None, _peak_rss() if w is not None else 0.0,
                         w is not None))
        return
    _store(f"wobbl {w.command}", w.signature(), outcome, out_dir, _peak_rss())


def _flush():
    """Record the runs kept back, each under its analysis when its request.json is there, else under the command
    of its outcome's domain."""
    while _pending:
        outcome, out_dir, peak, _ = _pending.pop(0)
        try:
            req = json.loads((out_dir / "request.json").read_text()) if out_dir else None
        except (OSError, ValueError):
            req = None
        if req:
            from ..simulate import get

            a = get(req["domain"], req["analysis"])
            spec = req.get("spec", {})
            w = analysis_workload(a, str(spec.get("thing", "")), spec.get("options", {}))
            solver, sig = f"wobbl {w.command}", w.signature()
        else:
            dom = outcome.get("domain", "")
            command = next((c for c, h in REGISTRY.items() if h.domain == dom), f"domain {dom}")
            solver, sig = f"wobbl {command}", ""
        try:
            _store(solver, sig, outcome, out_dir, peak)
        except Exception:  # noqa: BLE001
            pass


def _store(solver: str, sig: str, outcome: dict, out_dir, peak: float):
    from .. import estimate as E

    wall = float(outcome.get("t_total") or 0.0)
    if wall <= 0:
        return
    disk = 0.0
    if out_dir:
        disk = float(sum(p.stat().st_size for p in Path(out_dir).rglob("*") if p.is_file()))
    cal = E.load_calibration()
    if any(r.get("solver") == solver and r.get("size", {}).get("signature") == sig
           and r.get("size", {}).get("wall", r.get("measured_time")) == wall for r in cal.get("runs", [])[-20:]):
        return                                   # the same result saved again, after a fix or a new report
    pred = cal.get("predictions", {}).get(sig, {}) if sig else {}
    b = E.busy(included=True)          # a run slowed by other work counts at what it would take on an idle machine
    E.record(solver, pred.get("time", 0.0), wall / b, pred.get("memory", 0.0), peak,
             {"domain": outcome.get("domain", ""), "signature": sig, "disk": disk, "wall": wall,
              "load": round(E.load(), 1)}, "run")


def remember_prediction(w: Workload, seconds: float, peak_bytes: float):
    """Keep what an estimate said, as wall seconds on an idle machine at the threads the run will get, so the run
    that follows can be scored against it."""
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


_PART_COSTS: dict = {}


def part_cost(w: Workload, h: Hooks):
    """A command that meshes a CAD part, priced from the part's first mesh and its procedure's solves."""
    import tempfile

    from .. import estimate as E
    from ..structural.run import describe_part

    p = w.params
    domain = "strength" if h.dof == 3 else "heat"
    st = w.part.stat()
    key = (str(w.part.resolve()), st.st_size, int(st.st_mtime), domain, p.get("quality"), p.get("mesh_size"),
           p.get("scale"), bool(p.get("duration")), p.get("budget"))
    if key not in _PART_COSTS:     # several analyses of one plan mesh the same part the same way
        with tempfile.TemporaryDirectory() as tmp:
            prep, feats = describe_part(str(w.part), tmp, float(p.get("scale") or 1.0))
            mesh = E.first_mesh(prep, feats, domain, str(p.get("quality") or "normal"), p.get("mesh_size"),
                                bool(p.get("duration")))
        _PART_COSTS[key] = E.study_cost(mesh, domain, str(p.get("quality") or "normal"), bool(p.get("duration")),
                                        float(p.get("budget") or 300.0))
    c = copy.deepcopy(_PART_COSTS[key])
    c.core_seconds *= h.solves
    c.solver, c.procedure = f"wobbl {w.command}", w.domain
    if h.solves != 1.0:
        c.basis.insert(1, f"A {w.domain} study costs about {h.solves:g} static solves of each mesh, which its first "
                          "measured run here corrects.")
    return c


# ----------------------------------------------------------------------------------------------- simulate plans
def analysis_command(a) -> str:
    """The command an analysis of the simulate registry is priced and remembered under: the command it replaced,
    so 'wobbl run' and 'part.stress' share one history, or else its own key."""
    legacy = getattr(a, "legacy", "") or ""
    return legacy[len("wobbl "):] if legacy.startswith("wobbl ") else f"simulate {a.key}"


def analysis_workload(a, thing: str, options: dict) -> Workload:
    """One analysis of a plan as a workload, from the options that set its size."""
    params = {}
    for k in SIZE_OPTIONS:
        v = options.get(k)
        if v not in (None, [], ""):
            params[k.replace("-", "_")] = v[-1] if isinstance(v, (list, tuple)) else v
    path = Path(thing).expanduser() if thing else None
    has = path is not None and len(thing) < 1024 and path.exists()
    if has:
        params["part"] = str(path)
    command = analysis_command(a)
    if command not in REGISTRY and a.key in REGISTRY:
        command = a.key
    return Workload([], command, params, [path] if has else [], None, [str(path)] if has else [])


def plan_cost(plan, options: dict, here: bool = True):
    """What a whole simulate plan costs: each analysis priced as its own workload, then added up. This machine's
    calibration of each analysis is applied here, since a plan mixes several."""
    from .. import estimate as E

    parts = []
    for a in plan.analyses:
        w = analysis_workload(a, plan.thing, options)
        c = cost(w)
        if here and c.known and not c.measured:     # so the run that follows is scored against it, analysis by analysis
            t = min(E.default_threads(), E.physical_cores())
            remember_prediction(w, c.core_seconds / E.amdahl(t, c.parallel), c.peak_bytes)
        parts.append((a, c))
    known = [(a, c) for a, c in parts if c.known]
    missing = [a.key for a, c in parts if not c.known]
    solver = f"simulate {plan.domain}"
    if not known:
        return E.Cost(solver, plan.domain, 0.0, 0.0, 0.0, 0.0, {"analyses": [a.key for a, _ in parts]},
                      [f"Nothing has measured {', '.join(missing)} on this machine yet, so its time is not known. Its "
                       "first run, here or as a job, measures it."], known=False)
    secs = par = mem = disk = 0.0
    basis = []
    for a, c in known:
        f = E.factor(c.solver)[0] if here and not c.measured else 1.0
        secs += c.core_seconds * f
        par += c.core_seconds * f * c.parallel
        mem = max(mem, c.peak_bytes)
        disk += c.disk_bytes
        basis.append(f"{a.key}: {c.basis[0]}" if c.basis else a.key)
    if missing:
        basis.append(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not counted, since nothing has "
                     "measured them on this machine yet.")
    size = {"analyses": [a.key for a, _ in parts]}
    for _, c in known:
        size.update({k: v for k, v in c.size.items() if k in ("nodes", "levels", "dof")})
    return E.Cost(solver, plan.domain, secs, par / max(secs, 1e-9), mem, disk, size, basis, measured=True)


def plan_argv(plan, options: dict, out: str | None = None, quick: bool = False) -> list[str]:
    """The plumbing command line that runs a plan exactly as planned, here or on another machine."""
    argv = ["jobs", "run-plan", plan.thing or "-", "--question", plan.question or "", "--domain", plan.domain,
            "--kind", plan.kind]
    for a in plan.analyses:
        argv += ["--physics", a.key]
    for k, vs in options.items():
        for v in (vs if isinstance(vs, (list, tuple)) else [vs]):
            argv += ["--opt", k, str(v)]
    if quick:
        argv.append("--quick")
    return argv + (["-o", out] if out else [])


def plan_of(w: Workload):
    """The plan a 'jobs run-plan' workload runs, and its options."""
    from ..simulate import route

    p = w.params
    plan = route(p.get("thing") if p.get("thing") != "-" else "", p.get("question") or "", list(p.get("physics") or []),
                 p.get("domain") or None, p.get("kind") or None)
    options: dict[str, list[str]] = {}
    for k, v in p.get("opts") or ():
        options.setdefault(k, []).append(v)
    return plan, options


def _run_plan_cost(w: Workload):
    plan, options = plan_of(w)
    return plan_cost(plan, options)


def _builtin():
    register("run", domain="strength", solves=1.0)
    register("thermal", domain="thermal", solves=1.0, dof=1)
    register("buckling", domain="buckling", solves=2.5)
    register("vibration", domain="vibration", solves=2.0)
    register("plastic", domain="plastic", solves=25.0)
    register("jobs run-plan", domain="simulate", cost=_run_plan_cost)


_builtin()
