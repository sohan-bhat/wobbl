"""One interface from a finished run, in any domain, to the same run with other inputs.

Uncertainty, the comparison with tests, the credibility assessment and the test plan all need the same three things
from a run, whatever it simulates.

1. What it was run from. 'wobbl simulate' writes request.json next to every outcome.json, {"domain": "part",
   "analysis": "stress", "spec": {"thing", "kind", "question", "options": {name: [values]}}}, with every value the
   run used. Its numbers are the inputs that can scatter, each in the unit it was written in: a plain number, a
   quantity such as "9 cm", a load such as "220 N down @ top" (its number scatters, the rest stays) or a parameter
   such as "thickness=6". A repeatable option with several values names them load1, load2 and so on.
2. How to run it again. wobbl.simulate.get(domain, analysis).run(spec, out, quick) runs a spec into a folder and
   returns its outcome. A domain the registry does not have yet registers a runner here (register).
3. What decides it. outcome.json lists its criteria, [{"name", "value", "limit", "sense"}], each a value (a key of
   its values, or of the outcome itself) that must stay at most (max) or at least (min) its limit. A run that lists
   none has them inferred from value and limit pairs such as t_max_C and limit_C.

Runs from before request.json existed are read through legacy loaders where their folder kept enough (a strength or
heat study's study.json, a circuit template's parameters), and runs of a domain the registry does not have yet keep
a request.json of their own, {"domain", "kind", "spec"}, written by save_request.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REQUEST = "request.json"
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class DomainError(ValueError):
    pass


@dataclass
class Request:
    domain: str
    spec: dict
    kind: str = ""                      # the analysis, or the entry point within a domain such as pipe for flow
    roles: dict = field(default_factory=dict)   # optional input name -> role, from the domain
    registry: bool = False              # written by 'wobbl simulate', so its registry runs it again


PATH_KEYS = {"part", "file", "path", "netlist", "model", "design", "spec_file"}


def save_request(out_dir, domain: str, spec: dict, kind: str = "", roles: dict | None = None) -> Path:
    """Write request.json, so the run can be solved again with other inputs. A file the spec names (a part, a
    netlist, a design file) is written relative to the result folder, as study.json does, so the folder can move
    with its inputs."""
    out = Path(out_dir)
    d = {"domain": domain, "kind": kind, "spec": _paths(_plain(spec), lambda p: _rel(p, out))}
    if roles:
        d["roles"] = roles
    p = out / REQUEST
    p.write_text(json.dumps(d, indent=1, default=str))
    return p


def _rel(p: str, out: Path) -> str:
    import os

    q = Path(p)
    if not q.is_absolute() or not q.exists():
        return p
    try:
        return os.path.relpath(q, out.resolve())
    except ValueError:      # another drive on Windows
        return p


def _paths(x, fn):
    if isinstance(x, dict):
        return {k: (fn(v) if k in PATH_KEYS and isinstance(v, str) and v else _paths(v, fn)) for k, v in x.items()}
    if isinstance(x, list):
        return [_paths(v, fn) for v in x]
    return x


def _plain(x):
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    elif hasattr(x, "__dataclass_fields__"):
        from dataclasses import asdict
        x = asdict(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# --- reading what a run was made from ------------------------------------------------
LEGACY: dict[str, Callable[[Path, dict], Request | None]] = {}


def legacy(domain: str):
    def deco(fn):
        LEGACY[domain] = fn
        return fn
    return deco


def load_request(result_dir, outcome: dict | None = None) -> Request:
    d = Path(result_dir)
    p = d / REQUEST
    if p.exists():
        r = json.loads(p.read_text())
        if "analysis" in r:
            return Request(r["domain"], r.get("spec", {}), r["analysis"], r.get("roles", {}), registry=True)

        def absolute(v: str) -> str:
            q = Path(v)
            return str((d / q).resolve()) if not q.is_absolute() and (d / q).exists() else v
        return Request(r["domain"], _paths(r.get("spec", {}), absolute), r.get("kind", ""), r.get("roles", {}))
    o = outcome or json.loads((d / "outcome.json").read_text())
    from . import runners   # noqa: F401  registers each domain's legacy reader too

    fn = LEGACY.get(o.get("domain", ""))
    req = fn(d, o) if fn else None
    if req is None:
        raise DomainError(f"this {o.get('domain', '')} result has no request.json, so Wobbl cannot solve it again "
                          "with other inputs; run it again with this version of Wobbl")
    return req


@legacy("circuit")
def _circuit_legacy(d: Path, o: dict) -> Request | None:
    params = (o.get("values") or {}).get("params")
    if params is None or o.get("part", "").endswith((".cir", ".sp", ".net")):
        return None
    return Request("circuit", {"template": o.get("part", ""), "params": dict(params)}, "template")


# --- running again ------------------------------------------------------------------
Runner = Callable[[Request, Path], dict]
_RUNNERS: dict[str, Runner] = {}
_ROUTER: list[Callable[[Request, Path], dict | None]] = []


def register(domain: str):
    """Register how a domain runs a request into a folder and returns its outcome dict."""
    def deco(fn: Runner) -> Runner:
        _RUNNERS[domain] = fn
        return fn
    return deco


def set_router(fn: Callable[[Request, Path], dict | None]):
    """A router that can run any domain (the simulate registry). It is tried first and returns None for a domain
    it does not know."""
    _ROUTER[:] = [fn]


def _analysis(req: Request):
    """The registry's analysis for a request, or None when the registry does not have it."""
    try:
        from ..simulate import get
    except ImportError:
        return None
    try:
        return get(req.domain, req.kind) if req.kind else get(req.domain)
    except (KeyError, LookupError, ValueError):
        return None


def _router_run(req: Request, out: Path) -> dict | None:
    if _ROUTER:
        res = _ROUTER[0](req, out)
        if res is not None:
            return res
    if not req.registry:
        return None
    a = _analysis(req)
    if a is None:
        raise DomainError(f"this run came from the {req.domain} {req.kind} analysis, which this Wobbl does not have")
    return _as_outcome(a.run(req.spec, out, quick=True), out)


def _as_outcome(res, out: Path) -> dict:
    if isinstance(res, dict) and "values" in res:
        return res
    if hasattr(res, "to_dict"):
        return res.to_dict()
    p = Path(out) / "outcome.json"
    if p.exists():
        return json.loads(p.read_text())
    raise DomainError("the rerun wrote no outcome")


def rerun(req: Request, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = _router_run(req, out)
    if res is not None:
        return res
    from . import runners   # noqa: F401  the runners each domain registers

    fn = _RUNNERS.get(req.domain)
    if fn is None:
        raise DomainError(f"Wobbl does not know how to run a {req.domain} study again yet")
    return _as_outcome(fn(req, out), out)


def can_rerun(req: Request) -> bool:
    if req.registry:
        return _analysis(req) is not None
    from . import runners   # noqa: F401
    return req.domain in _RUNNERS or bool(_ROUTER)


# --- the numbers a spec is made of ---------------------------------------------------
SKIP_KEYS = {"question", "name", "part", "material", "template", "kind", "quality", "seed", "version", "solver",
             "label", "source", "file", "path", "out", "output", "budget", "timeout", "threads", "id", "units",
             "base", "cell", "prop", "shape", "fluid", "fuel", "oxidizer", "led", "eng", "design", "keep", "cfd",
             "thing", "body", "scale", "modes", "wobbl"}


@dataclass
class Leaf:
    path: tuple
    value: float
    unit: str
    text: bool            # written as a string, so it is written back the same way
    label: str = ""       # the short name it goes by
    prefix: str = ""      # what comes before the number, as "thickness=" in "thickness=6"
    suffix: str = ""      # what comes after the number and its unit, as " down @ top" in "220 N down @ top"

    def write(self, v: float):
        if not self.text:
            return float(v)
        return f"{self.prefix}{float(v):.10g}{' ' + self.unit if self.unit else ''}{self.suffix}"


_UNIT = r"[A-Za-z\u00b5\u00b0%][\w/^\u00b7*.%\u00b5\u00b0-]*"
_STR = re.compile(rf"^\s*(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?(?P<rest>\s+\S.*)?\s*$")
_PARAM = re.compile(rf"^\s*(?P<key>[A-Za-z_][\w-]*)\s*=\s*(?P<num>{_NUM})\s*(?P<unit>{_UNIT})?\s*$")


def _parse(x: str) -> tuple[float, str, str, str, str] | None:
    """(value, unit, prefix, suffix, own name) of a string that holds one number to vary, or None."""
    m = _PARAM.match(x)
    if m:
        return float(m["num"]), m["unit"] or "", f"{m['key']}=", "", m["key"]
    m = _STR.match(x)
    if m and (m["unit"] or not m["rest"]):
        return float(m["num"]), m["unit"] or "", "", m["rest"] or "", ""
    return None


def leaves(spec: dict, skip: set[str] = SKIP_KEYS) -> dict[str, Leaf]:
    """Every number of a spec that can scatter, by a short name: its key, its key and place for a repeated option
    (load1, load2), the name a parameter gives itself (thickness in "thickness=6"), or, when two share a name, the
    shortest tail of its path that tells them apart (rocket.length and nose.length)."""
    found: list[Leaf] = []

    def add(x, path, label):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            if np.isfinite(x):
                found.append(Leaf(path, float(x), "", False, label))
        elif isinstance(x, str):
            got = _parse(x)
            if got:
                v, unit, pre, suf, own = got
                found.append(Leaf(path, v, unit, True, own or label, pre, suf))

    def walk(x, path, label):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() not in skip and not str(k).startswith("_"):
                    walk(v, path + (k,), str(k))
        elif isinstance(x, list):
            if x and all(isinstance(v, dict) for v in x):
                for j, v in enumerate(x):
                    walk(v, path + (j,), label)
            elif 0 < len(x) <= 8 and all(isinstance(v, (int, float, str)) for v in x):
                for j, v in enumerate(x):
                    add(v, path + (j,), label if len(x) == 1 else f"{label}{j + 1}")
        else:
            add(x, path, label)
    walk(spec, (), "")

    def name(lf: Leaf, k: int) -> str:
        comps = [str(c) for c in lf.path[:-1] if not isinstance(c, int) and str(c) not in ("options", "spec")]
        if not isinstance(lf.path[-1], int) and comps and comps[-1] == lf.path[-1]:
            comps = comps[:-1]
        parts = comps[max(0, len(comps) - (k - 1)):] if k > 1 else []
        return ".".join(parts + [lf.label]).lower()
    depth = {id(lf): 1 for lf in found}
    for _ in range(8):
        names = [name(lf, depth[id(lf)]) for lf in found]
        dup = {n for n in names if names.count(n) > 1}
        if not dup:
            break
        for lf, n in zip(found, names):
            if n in dup:
                depth[id(lf)] += 1
    out = {}
    for lf in found:
        n = name(lf, depth[id(lf)])
        while n in out:           # a clash no path can settle keeps its place in the list
            n = f"{n}.{len(out)}"
        out[n] = lf
    return out


def with_values(spec: dict, values: dict[str, float], found: dict[str, Leaf]) -> dict:
    """A copy of the spec with these leaves set, each written in the form and unit it had."""
    s = copy.deepcopy(spec)
    for name, v in values.items():
        lf = found[name]
        node = s
        for p in lf.path[:-1]:
            node = node[p]
        node[lf.path[-1]] = lf.write(v)
    return s


ROLE_WORDS = [
    ("strength", r"yield|uts|strength|allow|limit|rating|imax|max_|_max|tj_max|endurance|toughness|kic"),
    ("stiffness", r"^e$|modulus|young|stiff"),
    ("load", r"load|force|power|thrust|pressure|flow|current|voltage|supply|vin|vbus|torque|speed|velocity|rpm|"
             r"mass|weight|payload|moment|accel|impulse|heat|watts|^p$|^f$|^v$"),
    ("geometry", r"diameter|length|thick|width|height|radius|span|size|depth|gap|chord|pitch|area|^d$|^l$|^t$|^r$|"
                 r"draft|beam|wall|bore|throat|grain"),
    ("boundary", r"film|^h$|ambient|wind|temperature|temp|rough|altitude|angle|cd|drag|friction|damping|rail|"
                 r"humidity|density_air|air"),
]


def role_of(name: str, roles: dict | None = None) -> str:
    if roles and name in roles:
        return roles[name]
    n = name.split(".")[-1].lower()
    for role, pat in ROLE_WORDS:
        if re.search(pat, n):
            return role
    return "material"


# --- what decides the answer ---------------------------------------------------------
UNIT_TOKENS = {"C": "C", "K": "K", "F": "F", "mm": "mm", "m": "m", "km": "km", "s": "s", "ms": "ms", "min": "min",
               "h": "h", "N": "N", "kN": "kN", "Pa": "Pa", "kPa": "kPa", "MPa": "MPa", "bar": "bar", "W": "W",
               "kW": "kW", "A": "A", "mA": "mA", "V": "V", "mV": "mV", "Hz": "Hz", "dB": "dB", "J": "J", "kg": "kg",
               "g": "g", "Ns": "N s", "mps": "m/s", "m_s": "m/s", "deg": "deg", "pct": "%", "percent": "%", "rpm": "rpm",
               "Nm": "N m", "ohm": "ohm", "T": "T", "cal": "cal", "ft": "ft", "cycles": "cycles"}


def unit_of(key: str) -> str:
    parts = key.split("_")
    for n in (2, 1):
        if len(parts) > n and "_".join(parts[-n:]) in UNIT_TOKENS:
            return UNIT_TOKENS["_".join(parts[-n:])]
    return ""


def scalar_values(o: dict, top: bool = False) -> dict[str, float]:
    """The numbers in an outcome's values, and with top its own numeric fields too (a strength outcome keeps
    governing_MPa and safety_factor at the top), with values winning a clash."""
    out = {}
    if top:
        for k, x in o.items():
            if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and k not in ("t_total",):
                out[k] = float(x)
    v = o.get("values") or {}
    for k, x in v.items():
        if k.startswith("_") or isinstance(x, bool):
            continue
        if isinstance(x, (int, float)) and np.isfinite(x):
            out[k] = float(x)
    return out


def criteria(o: dict) -> list[dict]:
    """The run's pass or fail criteria: each {"name", "value", "limit" (a number) or "limit_key", "sense"}, with
    "max" for a value that must stay at or under its limit and "min" for one that must stay at or over it."""
    vals = scalar_values(o, top=True)
    given = o.get("criteria") or (o.get("values") or {}).get("_criteria") or []
    out = []
    for c in given:
        c = dict(c)
        if c.get("value") in vals and (c.get("limit_key") in vals or isinstance(c.get("limit"), (int, float))):
            c.setdefault("name", c["value"])
            c.setdefault("sense", "max")
            out.append(c)
    if out:
        return out
    return _inferred(vals)


_LIMIT_FORMS = [("{k}_limit", "max"), ("{k}_max", "max"), ("max_{k}", "max"), ("{k}_allowable", "max"),
                ("allowable_{k}", "max"), ("{k}_min", "min"), ("min_{k}", "min"), ("{k}_target", "max")]


_LIMITISH = re.compile(r"^(limit|max|min|allow\w*|target|rated|rating)(_|$)|(_limit|_allowable|_target|_rating)$")
_PEAKISH = re.compile(r"(^|_)(max|peak|top|hot|worst)(_|$)")


def _inferred(vals: dict[str, float]) -> list[dict]:
    """Criteria read from the names of a run's values, for a domain that lists none: a value and its limit by name
    (x and x_limit, x_max, max_x, x_allowable, x_min), a peak value and the limit in its unit (t_max_C and
    limit_C), and a safety factor or margin, which must stay over 1 and 0."""
    out, used = [], set()
    for k in vals:
        if _LIMITISH.search(k):
            continue
        base = k.rsplit("_", 1)[0] if "_" in k and k.rsplit("_", 1)[1] in UNIT_TOKENS else k
        hit = next(((form.format(k=b), sense) for b in dict.fromkeys([k, base]) for form, sense in _LIMIT_FORMS
                    if form.format(k=b) in vals and form.format(k=b) != k), None)
        if hit is None and "_" in k and k.rsplit("_", 1)[1] in UNIT_TOKENS and _PEAKISH.search(k):
            u = k.rsplit("_", 1)[1]
            hit = next(((f"{pre}_{u}", sense) for pre, sense in (("limit", "max"), ("max", "max"), ("min", "min"))
                        if f"{pre}_{u}" in vals), None)
        if hit:
            out.append({"name": k, "value": k, "limit_key": hit[0], "sense": hit[1], "inferred": True})
            used.add(k)
    for k, sense, lim in (("safety_factor", "min", 1.0), ("margin", "min", 0.0), ("sf", "min", 1.0)):
        if k in vals and k not in used:
            out.append({"name": k, "value": k, "limit": lim, "sense": sense, "inferred": True})
    return out


def limit_of(c: dict, vals: dict[str, float]) -> float:
    return float(vals[c["limit_key"]]) if c.get("limit_key") else float(c["limit"])


def margin(c: dict, vals: dict[str, float]) -> float:
    """How far a criterion is from failing, as a fraction of its limit: positive passes, negative fails."""
    v, lim = vals[c["value"]], limit_of(c, vals)
    scale = max(abs(lim), 1e-12)
    return (lim - v) / scale if c["sense"] == "max" else (v - lim) / scale


def governing(crit: list[dict], vals: dict[str, float]) -> dict | None:
    """The criterion closest to failing in the run itself."""
    ok = [c for c in crit if c["value"] in vals and (c.get("limit_key") in vals or "limit" in c)]
    return min(ok, key=lambda c: margin(c, vals)) if ok else None


# --- numerical error from a run's own checks -----------------------------------------
_CONV = re.compile(r"converg|time step|tolerance|refin|step size|mesh|integrated again|halved|tighter", re.I)


def numerical_change(o: dict) -> float:
    """The relative change the run's convergence checks measured, the numerical uncertainty it carries."""
    best = 0.0
    for c in o.get("checks", []):
        if not _CONV.search(c.get("name", "")):
            continue
        det = c.get("detail") or {}
        for key in ("last_change", "relative_change", "change"):
            x = det.get(key)
            if isinstance(x, (int, float)) and np.isfinite(x):
                best = max(best, abs(float(x)))
                break
        else:
            m = re.search(rf"(?:last change|changed? by|moves?|apart|differ\w*)[^%\d]*?({_NUM})\s*%",
                          c.get("summary", ""))
            if not m:
                m = re.search(rf"\(({_NUM})\s*%\s*apart\)", c.get("summary", ""))
            if m:
                best = max(best, abs(float(m.group(1))) / 100.0)
    return best


# --- a pool of reruns ----------------------------------------------------------------
class Reruns:
    """Runs a request again at other values of its leaves, each in its own folder that is removed afterwards."""

    def __init__(self, req: Request, found: dict[str, Leaf], keep_series: bool = False):
        self.req, self.found, self.keep = req, found, keep_series
        self.work = Path(tempfile.mkdtemp(prefix="wobbl-uq-"))
        self.count = 0
        self.seconds: list[float] = []

    def __call__(self, values: dict[str, float]) -> tuple[dict, dict[str, dict]]:
        """The outcome at these values, and any time series the run wrote (CSV files with a time column)."""
        self.count += 1
        out = self.work / f"r{self.count}"
        spec = with_values(self.req.spec, values, self.found)
        t = time.perf_counter()
        o = rerun(Request(self.req.domain, spec, self.req.kind, self.req.roles, self.req.registry), out)
        self.seconds.append(time.perf_counter() - t)
        series = read_series(out) if self.keep else {}
        shutil.rmtree(out, ignore_errors=True)
        return o, series

    def cleanup(self):
        shutil.rmtree(self.work, ignore_errors=True)


_TIME_COLS = ("time_s", "t_s", "time", "t")


def read_series(folder: Path) -> dict[str, dict]:
    """Every CSV in a result folder with a time column, as {column: array} keyed by file stem."""
    import csv

    out = {}
    for p in sorted(Path(folder).glob("*.csv")):
        try:
            rows = list(csv.reader(p.read_text().splitlines()))
        except (OSError, UnicodeDecodeError):
            continue
        rows = [r for r in rows if r and not r[0].startswith("#")]
        if len(rows) < 3:
            continue
        head = [h.strip() for h in rows[0]]
        tcol = next((j for j, h in enumerate(head) if h.lower() in _TIME_COLS), None)
        if tcol is None:
            continue
        cols = {}
        for j, h in enumerate(head):
            try:
                cols[h] = np.array([float(r[j]) for r in rows[1:] if len(r) > j])
            except ValueError:
                continue
        if head[tcol] in cols:
            cols["time"] = cols[head[tcol]]
            out[p.stem] = cols
    return out


def series_value(series: dict[str, dict], quantity: str, t: float) -> float | None:
    """A column of a time series at time t, by its name or 'file:column'."""
    stem, _, col = quantity.partition(":") if ":" in quantity else ("", "", quantity)
    for name, cols in series.items():
        if stem and name != stem:
            continue
        if col in cols and col != "time":
            return float(np.interp(t, cols["time"], cols[col]))
    return None
