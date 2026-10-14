"""The model behind uncertainty, tests and credibility for a run of any domain, through request.json and a rerun.

Any numeric leaf of the request can scatter, in the unit it was written in. Each design point is the same request
run again with those leaves changed, and the answer is a polynomial chaos fit of the reruns with its leave one out
error. The run is first solved again exactly as it was, which checks that the rerun reproduces the answer and says
how long one rerun takes. A rerun under half a second earns a larger design, since it costs little.

What fails is read from the run's criteria (domains.criteria), and the numerical uncertainty from the change its
own convergence checks measured. A domain with a faster exact path, like strength, plugs in its own model instead,
behind the same interface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..units import fmt
from . import domains as D
from . import uq
from .models import Model, ModelError, _numeric_factor
from .readings import ReadingError
from .uq import Input, Known

CHEAP_S = 0.5            # a rerun this quick gets a design three times the usual size
MAX_SOLVES = 150


def _signed(name: str, unit: str) -> bool:
    """An input that may go below zero: a temperature on a scale with an arbitrary zero, an angle, a wind or a
    height above some datum."""
    import re

    return unit.strip().lower() in ("c", "f", "°c", "°f", "degc", "degf", "deg", "°") or bool(
        re.search(r"temp|ambient|angle|wind|altitude|offset|rise", name.split(".")[-1].lower()))


class GenericModel(Model):
    def __init__(self, result_dir, outcome):
        super().__init__(result_dir, outcome)
        try:
            self.req = D.load_request(self.dir, outcome)
        except D.DomainError as e:
            raise ModelError(str(e)) from None
        self.domain = outcome.get("domain") or self.req.domain
        self.found = D.leaves(self.req.spec)
        for name, lf in self.found.items():
            words = " ".join(str(p) for p in lf.path).replace("_", " ")
            self.known[name] = Known(lf.value, lf.unit, D.role_of(name, self.req.roles), f"the {words}",
                                     positive=lf.value > 0 and not _signed(name, lf.unit))
            tail = str(lf.path[-1]).lower()
            if "." in name and tail not in self.known:
                self.aliases.setdefault(tail, name)
        self.vals = D.scalar_values(outcome, top=True)
        self.crit = D.criteria(outcome)
        gov = D.governing(self.crit, self.vals)
        names = [c["value"] for c in self.crit]
        own = list(D.scalar_values(outcome))
        if gov:
            self.decision = gov["value"]
        elif own:
            self.decision = own[0]
        else:
            raise ModelError(f"this {self.domain} result has no numeric values to carry scatter to")
        if names:
            self.outputs = list(dict.fromkeys([self.decision] + names))
        else:
            self.outputs = list(dict.fromkeys([self.decision] + own[:5]))
        # each criterion's limit as each rerun states it, so a limit that is itself an input scatters too
        self.limit_keys = [f"limit:{c['name']}" for c in self.crit]
        self.units = {k: D.unit_of(k) for k in self.outputs}
        self.reproduced: float | None = None
        self._limits = None

    def numerical(self) -> Input | None:
        return _numeric_factor(D.numerical_change(self.o), "numerical error, the change the run's own convergence "
                                                           "checks measured")

    def describe_inputs(self) -> str:
        return ", ".join(sorted(self.known))

    # --- running again -----------------------------------------------------------------
    def design(self, solved: list[Input], keys: list[str], extract, progress=None, solves=None, series=False):
        """Rerun the request at a Latin hypercube of the solved inputs and fit polynomial chaos to extract(outcome,
        series) for each design point. Returns the surrogate (None with nothing to solve) and the nominal answer."""
        if not D.can_rerun(self.req):
            raise ModelError(f"Wobbl does not know how to run a {self.req.domain} study again yet")
        runs = D.Reruns(self.req, self.found, keep_series=series)
        try:
            if progress:
                progress("solve", f"the {self.domain} study again as it was, to check it reproduces")
            o0, s0 = runs({})
            nominal = np.asarray(extract(o0, s0), float)
            v0 = D.scalar_values(o0, top=True)
            if self.decision in v0 and self.decision in self.vals:
                ref = self.vals[self.decision]
                self.reproduced = abs(v0[self.decision] - ref) / max(abs(ref), 1e-12)
            if not solved:
                return None, nominal
            n = solves or uq.design_size(len(solved))
            if not solves and runs.seconds[0] < CHEAP_S:
                n = min(max(n, 3 * uq.design_size(len(solved))), MAX_SOLVES)
            X = uq.sample(solved, n, "lhs", seed=7)
            Y = []
            for k, row in enumerate(X):
                if progress:
                    progress("solve", f"{self.domain} rerun {k + 1} of {n}")
                o, s = runs({i.name: float(v) for i, v in zip(solved, row)})
                Y.append(extract(o, s))
            return uq.fit_pce(solved, X, np.array(Y, float), keys), nominal
        finally:
            runs.cleanup()

    def _values(self, keys):
        given = {f"limit:{c['name']}": D.limit_of(c, self.vals) for c in self.crit}

        def extract(o, _s):
            v = D.scalar_values(o, top=True)
            lims = {f"limit:{c['name']}": D.limit_of(c, v) for c in D.criteria(o)
                    if c.get("limit_key") in v or "limit" in c}
            row = []
            for k in keys:
                if k in given:
                    row.append(lims.get(k, given[k]))
                elif k in v:
                    row.append(v[k])
                else:
                    raise ModelError(f"a rerun of this {self.domain} study gave no {k}")
            return row
        return extract

    def evaluator(self, inputs, progress=None, solves=None):
        solved = [i for i in inputs if i.role != "numerical"]
        keys = self.outputs + [k for k in self.limit_keys if k not in self.outputs]
        sur, nominal = self.design(solved, keys, self._values(keys), progress, solves)
        names = [i.name for i in inputs]
        idx = [names.index(i.name) for i in solved]
        nout = len(self.outputs)

        def full(X):
            X = np.atleast_2d(X)
            return sur(X[:, idx]) if sur is not None else np.tile(nominal, (len(X), 1))

        def f(X):
            Y = full(X)[:, :nout].copy()
            return Y * self._col(np.atleast_2d(X), inputs, "mesh", 1.0)[:, None]

        def limits(X):
            Y = full(X)
            return {k: Y[:, keys.index(k)] for k in self.limit_keys}
        self._limits = limits
        what = f"reruns of the {self.domain} study"
        if sur is None:
            info = {"kind": "exact", "method": "no input needs a new run, so the run's own answer is used as it is"}
        else:
            worst = max(sur.loo) if len(sur.loo) else 0.0
            info = {"kind": "polynomial chaos", "surrogate": sur.summary(),
                    "method": f"polynomial chaos of degree {sur.degree} fitted to {sur.n_train} {what} across "
                              f"{', '.join(i.name for i in solved)}, leave one out error {100 * worst:.2g}% of the "
                              "variance"}
        if self.reproduced is not None:
            info["reproduced"] = self.reproduced
            info["method"] += (f". Run again as it was, the study gave its {self.decision} back "
                               + ("exactly" if self.reproduced < 1e-9 else f"within {100 * self.reproduced:.2g}%"))
        return f, info

    def fails(self, X, Y, inputs):
        if not self.crit:
            return {}
        lims = self._limits(X) if self._limits else {}
        bad = np.zeros(len(Y), bool)
        words = []
        for c in self.crit:
            if c["value"] not in self.outputs:
                continue
            v = Y[:, self.outputs.index(c["value"])]
            lim = lims.get(f"limit:{c['name']}", np.full(len(Y), D.limit_of(c, self.vals)))
            bad |= (v > lim) if c["sense"] == "max" else (v < lim)
            u = D.unit_of(c["value"])
            lim0 = D.limit_of(c, self.vals)
            words.append(c.get("label") or f"{c['name']} {'at most' if c['sense'] == 'max' else 'at least'} "
                                            f"{fmt(lim0)}{' ' + u if u else ''}")
        lead = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        return {"failure": (bad, lead)}

    def predictor(self, readings, inputs, load_unc, progress=None, solves=None):
        return GenericPredictor(self, readings, inputs, load_unc, progress, solves)

    def context(self) -> str:
        q = self.o.get("question") or ""
        what = f"A {self.domain} model of {Path(str(self.o.get('part', ''))).name or 'the design'}"
        crit = ", ".join(f"{c['name']} against {fmt(D.limit_of(c, self.vals))}" for c in self.crit[:3])
        return what + (f", used to judge {crit}" if crit else "") + (f", to answer '{q}'" if q else "") + "."


class GenericPredictor:
    """Predictions at the readings from reruns: a value of the outcome by name, or a column of a time series the
    run wrote at a reading's time. A reading's load column sets the request's one load leaf for that reading."""

    def __init__(self, model: GenericModel, readings, inputs, load_unc, progress=None, solves=None):
        self.model, self.readings, self.load_unc = model, readings, load_unc
        self.notes: list[str] = []
        self.inputs = [i for i in inputs if i.name != "mesh"]
        loads = sorted({r.load for r in readings if r.load is not None})
        self.load_leaf = None
        if loads:
            cands = [n for n, k in model.known.items() if k.role == "load"]
            if len(cands) != 1:
                raise ReadingError("readings give a load, but this study has "
                                   + (f"{len(cands)} load inputs ({', '.join(cands)})" if cands else "no load input")
                                   + " to set it on; leave the load column out or name the load in request.json roles")
            self.load_leaf = cands[0]
            self.notes.append(f"each reading's load sets {self.load_leaf}, solved again at every load")
        keys = [r.quantity for r in readings]
        groups = {}
        for k, r in enumerate(readings):
            groups.setdefault(r.load, []).append(k)
        self.inputs = [i for i in self.inputs if i.name != self.load_leaf]
        solved, found = self.inputs, model.found

        def predict(row: dict, o: dict, series: dict, ks: list[int], out: list):
            vals = D.scalar_values(o, top=True)
            for k in ks:
                r = readings[k]
                q = keys[k]
                v = D.series_value(series, q, r.time) if r.time is not None else None
                if v is None:
                    v = vals.get(q)
                if v is None:
                    have = sorted(vals) + sorted({c for s in series.values() for c in s if c != "time"})
                    raise ReadingError(f"row {r.row}: this {model.domain} study has no value {q} (it has "
                                       f"{', '.join(have[:20])})")
                out[k] = float(v)

        runs = D.Reruns(model.req, found, keep_series=any(r.time is not None for r in readings))

        def solve(row: dict):
            out = [0.0] * len(readings)
            for load, ks in groups.items():
                r2 = dict(row)
                if load is not None:
                    r2[self.load_leaf] = load
                o, s = runs(r2)
                predict(r2, o, s, ks, out)
            return out
        try:
            if solved:
                n = solves or uq.design_size(len(solved))
                X = uq.sample(solved, n, "lhs", seed=7)
                Y = []
                for k, row in enumerate(X):
                    if progress:
                        progress("solve", f"{model.domain} rerun {k + 1} of {n}")
                    Y.append(solve({i.name: float(v) for i, v in zip(solved, row)}))
                self.sur = uq.fit_pce(solved, X, np.array(Y), keys)
                self.nominal = None
            else:
                self.sur, self.nominal = None, np.array(solve({}))
        finally:
            runs.cleanup()

    def __call__(self, X):
        X = np.atleast_2d(X)
        return self.sur(X) if self.sur is not None else np.tile(self.nominal, (len(X), 1))
