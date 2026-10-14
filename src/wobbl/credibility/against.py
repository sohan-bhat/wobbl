"""wobbl validate-against: compare a finished run with physical test readings and calibrate its uncertain inputs.

The comparison follows ASME V&V 20. For each reading the comparison error is E = S - D, the simulation minus the
data, and the validation uncertainty u_val = sqrt(u_num^2 + u_input^2 + u_D^2) combines the numerical uncertainty
the run's own convergence checks measured, the uncertainty the model's inputs carry into that reading and the
reading's own uncertainty. When |E| is within 2 u_val the model agrees as closely as the comparison can resolve.
When it is larger, the model error is about E, give or take 2 u_val.

The area metric of Ferson, Oberkampf and Ginzburg adds a second view, the area between the model's distribution
at a reading and the readings themselves, in the reading's units, and a pooled area over every reading on one
scale. Then the uncertain inputs are calibrated against the readings (calibrate.py), which says which values of,
say, Young's modulus or a film coefficient make the model agree, and how sure that is.

Wobbl reports this evidence. It does not decide that a model is valid for a purpose, which is the reviewer's call.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..units import fmt
from . import uq
from .models import CircuitModel, ModelError, StrengthModel, ThermalModel, design_surrogate, for_result
from .readings import Reading, ReadingError, area_metric, default_uncertainty, read_tests, u_pooled

MC = 20000


class Predictor:
    """Predictions of a model at the readings, as a cheap function of the uncertain inputs that matter there."""

    def __init__(self, model, readings: list[Reading], inputs: list[uq.Input], load_unc: float, progress=None):
        self.model, self.readings, self.inputs, self.load_unc = model, readings, inputs, load_unc
        self.notes: list[str] = []

    def __call__(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StrengthPredictor(Predictor):
    """Strain and displacement at gauges from the converged fields, scaled exactly to each reading's load and to
    Young's modulus, the only input that changes them once the load is measured."""

    def __init__(self, model: StrengthModel, readings, inputs, load_unc, progress=None):
        super().__init__(model, readings, inputs, load_unc, progress)
        from .probes import GAUGE_MM, gauge, load_fields, surface

        f = load_fields(model.dir)
        if f is None or "S" not in f:
            raise ModelError("this run kept no fields (work/fields.npz), so gauge readings cannot be predicted; run "
                             "the study again without deleting its work folder")
        E0, nu = model.mat.E, model.mat.nu

        def read(f):
            surf, out = surface(f), []
            for r in readings:
                if r.quantity == "strain":
                    g = gauge(f, r.xyz, E0, nu, r.direction, r.gauge_length or GAUGE_MM, surf)
                    if r.direction is None:
                        r.direction = g["direction"]
                    out.append(g["strain"])
                elif r.quantity == "displacement":
                    i = int(np.argmin(np.linalg.norm(f["X"] - np.asarray(r.xyz), axis=1)))
                    u = f["U"][i]
                    if r.direction is None:
                        r.direction = (u / max(np.linalg.norm(u), 1e-300)).tolist()
                    out.append(float(u @ np.asarray(r.direction)))
                else:
                    raise ReadingError(f"row {r.row}: a strength run predicts strain and displacement, not "
                                       f"{r.quantity}")
            return np.array(out)
        self.unit = read(f)
        # the other load attachment idealization the run solved, when its field was kept
        alt = load_fields(Path(model.dir) / "work" / "load_path")
        self.alt_unit = read(alt) if alt is not None and "S" in alt else None
        self.alt = model.alternatives()[0] if self.alt_unit is not None and model.alternatives() else None
        p0 = model.load_nominal
        self.ratio = np.array([(r.load if r.load is not None else p0) / p0 for r in readings])
        self.e = [j for j, i in enumerate(inputs) if i.name == "E"]
        self.E0 = E0
        dropped = [i.name for i in inputs if i.name != "E"]
        if dropped:
            self.notes.append(f"{', '.join(dropped)} does not change a gauge reading once the test load is measured "
                              "(or would need new solves at the gauges), so it is left out of the comparison")
        self.inputs = [inputs[j] for j in self.e]

    def __call__(self, X):
        X = np.atleast_2d(X)
        e = self.E0 / X[:, 0] if self.inputs else np.ones(len(X))
        return e[:, None] * (self.unit * self.ratio)[None, :]


class ThermalPredictor(Predictor):
    """Temperatures at the sensors, from new heat solves across the uncertain inputs fitted with polynomial chaos."""

    def __init__(self, model: ThermalModel, readings, inputs, load_unc, progress=None, solves=None):
        super().__init__(model, readings, inputs, load_unc, progress)
        bad = [r for r in readings if r.quantity != "temperature"]
        if bad:
            raise ReadingError(f"row {bad[0].row}: a heat run predicts temperature, not {bad[0].quantity}")
        self.inputs = [i for i in inputs if i.name not in ("limit", "mesh")]
        pts = np.array([r.xyz for r in readings])
        times = [r.time for r in readings]

        def extract(mesh, res):
            idx = np.argmin(np.linalg.norm(mesh.X[None, :, :] - pts[:, None, :], axis=2), axis=1)
            out = []
            for k, i in enumerate(idx):
                if times[k] is None or not len(res.history_t):
                    out.append(float(res.T[i]))
                else:
                    t = np.concatenate([[0.0], res.history_t])
                    h = np.concatenate([[model.study.initial if model.study.initial is not None
                                         else model.study.ambient], res.history_T[:, i]])
                    out.append(float(np.interp(times[k], t, h)))
            return out
        solver = model.solver(extract)
        if self.inputs:
            self.sur = design_surrogate(solver, self.inputs, [r.name for r in readings], progress, solves, "heat solve")
            self.nominal = None
        else:
            self.sur, self.nominal = None, np.array(solver({}))
            solver.cleanup()

    def __call__(self, X):
        X = np.atleast_2d(X)
        return self.sur(X) if self.sur is not None else np.tile(self.nominal, (len(X), 1))


class CircuitPredictor(Predictor):
    """Template values (i_led_A, ripple_V, tj_C) from new ngspice runs across the uncertain inputs."""

    def __init__(self, model: CircuitModel, readings, inputs, load_unc, progress=None, solves=None):
        super().__init__(model, readings, inputs, load_unc, progress)
        from ..circuits.run import run_circuit

        lim = {"imax", "max_ripple", "tj_max"}
        self.inputs = [i for i in inputs if i.name not in lim and i.name != "mesh"]
        keys = [r.quantity for r in readings]
        work = Path(model.dir) / "work" / "validate"
        count = [0]

        wave = [k for k, r in zip(keys, readings) if "(" in k and r.time is not None]

        def solve(row):
            from ..circuits.run import simulate
            from ..circuits.templates import TEMPLATES

            count[0] += 1
            p = {**model.params, **row}
            o = run_circuit(model.template, p, work / f"d{count[0]}")
            data = {}
            if wave:   # waveform readings: the simulated signal at each reading's time
                tpl = TEMPLATES[model.template](p)
                data = simulate(tpl, work / f"w{count[0]}")
                if "time" not in data:
                    raise ReadingError(f"the {model.template} template is a steady operating point, so it has no "
                                       "waveform to read at a time")
            out = []
            for k, r in zip(keys, readings):
                if k in wave:
                    if k not in data:
                        raise ReadingError(f"the {model.template} template has no signal {k} (it has "
                                           f"{', '.join(x for x in data if x != 'time')})")
                    out.append(float(np.interp(r.time, data["time"], data[k])))
                elif k in o.values:
                    out.append(float(o.values[k]))
                else:
                    raise ReadingError(f"the {model.template} template has no value {k} (it has "
                                       f"{', '.join(x for x in o.values if not x.startswith('_') and x != 'params')}"
                                       ", or a signal such as v(out) with a time)")
            return out
        solve.cleanup = lambda: __import__("shutil").rmtree(work, ignore_errors=True)
        if self.inputs:
            self.sur = design_surrogate(solve, self.inputs, keys, progress, solves, "ngspice run")
            self.nominal = None
        else:
            self.sur, self.nominal = None, np.array(solve({}))
            solve.cleanup()

    def __call__(self, X):
        X = np.atleast_2d(X)
        return self.sur(X) if self.sur is not None else np.tile(self.nominal, (len(X), 1))


def _inputs_for(model, vary: list[str], d: Path) -> list[uq.Input]:
    """The run's stated scatter (from uncertainty.json) with --vary on top, less the inputs a test fixes."""
    from .uncertainty import load

    given = {i.name: i for i in model.parse(list(vary))}
    u = load(d)
    if u:
        for raw in u["inputs"]:
            inp = uq.Input.from_dict(raw)
            if inp.name not in given and inp.role not in ("numerical",):
                given[inp.name] = inp
    # a test measures its own load, and the strength or temperature limit does not move a reading
    return [i for i in given.values() if i.role not in ("load", "strength", "numerical")]


def run_validation(result_dir, tests: str, vary: list[str] | None = None, load_uncertainty: float = 0.01,
                   solves: int | None = None, calibrate_inputs: bool = True, progress=None) -> dict:
    d = Path(result_dir)
    model = for_result(d)
    readings, meta = read_tests(tests)
    inputs = _inputs_for(model, list(vary or []), d)
    if progress:
        progress("predict", f"{len(readings)} readings from {meta['file']}")
    if isinstance(model, StrengthModel):
        pred = StrengthPredictor(model, readings, inputs, load_uncertainty, progress)
    elif isinstance(model, ThermalModel):
        pred = ThermalPredictor(model, readings, inputs, 0.0, progress, solves)
    else:
        pred = CircuitPredictor(model, readings, inputs, 0.0, progress, solves)
    num = model.numerical()
    rel_num = num.sd if num else 0.0
    data = np.array([r.value for r in readings])
    u_d = np.array([r.u if r.u is not None else default_uncertainty(r) for r in readings])
    notes = list(pred.notes)
    if any(r.u is None for r in readings):
        notes.append("readings with no stated uncertainty were given a typical one: 5 microstrain or 2% for a "
                     "bonded strain gauge, 0.005 mm or 1% for a dial gauge, 0.5 C for a thermocouple")
    ins = pred.inputs
    mean = np.array([[i.mean for i in ins]]) if ins else np.zeros((1, 0))
    S = pred(mean)[0]
    X = uq.sample(ins, MC, "lhs", seed=11) if ins else np.zeros((MC, 0))
    rng = np.random.default_rng(12)
    noise = 1.0 + rel_num * rng.standard_normal((MC, 1)) + pred.load_unc * rng.standard_normal((MC, 1))
    samples = pred(X) * noise
    u_in = np.sqrt(np.var(pred(X), axis=0) + (pred.load_unc * S) ** 2)
    u_num = rel_num * np.abs(S)
    u_val = np.sqrt(u_num ** 2 + u_in ** 2 + u_d ** 2)
    E = S - data
    rows, us = [], []
    for k, r in enumerate(readings):
        within = abs(E[k]) <= 2 * u_val[k]
        us.append(float(np.mean(samples[:, k] <= data[k])))
        rows.append({**r.to_dict(), "u": float(u_d[k]), "simulation": float(S[k]), "error": float(E[k]),
                     "relative_error": float(E[k] / data[k]) if data[k] else None, "u_num": float(u_num[k]),
                     "u_input": float(u_in[k]), "u_val": float(u_val[k]),
                     "ratio": float(abs(E[k]) / u_val[k]) if u_val[k] > 0 else None, "within": bool(within),
                     "verdict": "agrees within the validation uncertainty" if within else
                     f"model error of about {_q(E[k], r.quantity)}, give or take {_q(2 * u_val[k], r.quantity)}"})
    groups = {}
    for k, r in enumerate(readings):
        groups.setdefault((r.name, r.load, r.time), []).append(k)
    areas = []
    for (name, load, time), ks in groups.items():
        a = area_metric(samples[:, ks[0]], data[ks])
        areas.append({"name": name, "load": load, "time": time, "readings": len(ks), "area": a,
                      "relative": a / abs(np.mean(data[ks])) if np.mean(data[ks]) else None,
                      "quantity": readings[ks[0]].quantity})
    pooled = u_pooled(np.array(us))
    res = {"kind": "validation", "domain": model.domain, "wobbl": __version__,
           "created": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "tests": meta, "readings": rows,
           "area_metric": areas, "u_pooled": pooled,
           "inputs": [i.to_dict() for i in ins], "numerical_relative": rel_num, "load_uncertainty": pred.load_unc,
           "summary": {"readings": len(rows), "within": int(sum(r["within"] for r in rows)),
                       "mean_relative_error": _mean([r["relative_error"] for r in rows]),
                       "rms_relative_error": _rms([r["relative_error"] for r in rows]),
                       "largest_ratio": max((r["ratio"] or 0.0) for r in rows)},
           "notes": notes}
    sigma = np.sqrt(u_d ** 2 + u_num ** 2 + (pred.load_unc * S) ** 2)
    if calibrate_inputs and ins:
        from .calibrate import calibrate

        if progress:
            progress("calibrate", ", ".join(i.name for i in ins))
        res["calibration"] = calibrate(pred, ins, data, sigma)
    if getattr(pred, "alt_unit", None) is not None and pred.alt:
        res["model_form"] = model_form(pred, data, sigma)
    res["headline"] = headline(res)
    from .report import refresh
    from .uncertainty import _clean

    (d / "validation.json").write_text(json.dumps(_clean(res), indent=1))
    refresh(d)
    return res


def model_form(pred: StrengthPredictor, data: np.ndarray, sigma: np.ndarray) -> dict:
    """Which load attachment idealization the readings favour. The modulus is fitted to each one on its own (one
    scale factor, since every reading goes as 1 / E), so neither wins by a stiffness it does not have, and the
    ratio of their likelihoods says how strongly the readings prefer one."""
    rows = []
    for label, unit in ((pred.alt["assumed"], pred.unit), (pred.alt["label"].replace("loads entering through ", ""),
                                                           pred.alt_unit)):
        s = unit * pred.ratio
        k = float(np.sum(s * data / sigma ** 2) / max(np.sum(s ** 2 / sigma ** 2), 1e-300))
        r = (k * s - data) / sigma
        rel = (k * s - data) / np.where(data != 0, np.abs(data), 1.0)
        rows.append({"idealization": label, "fitted_modulus": pred.E0 / k if k > 0 else None,
                     "chi2": float(r @ r), "rms_in_sigma": float(np.sqrt(np.mean(r ** 2))),
                     "rms_relative_error": float(np.sqrt(np.mean(rel ** 2)))})
    log10 = (rows[1]["chi2"] - rows[0]["chi2"]) / (2 * np.log(10))
    return {"idealizations": rows, "favours": rows[0]["idealization"] if log10 >= 0 else rows[1]["idealization"],
            "log10_likelihood_ratio": float(abs(log10)), "assumed": rows[0]["idealization"]}


def ratio_words(mf: dict) -> str:
    x = mf["log10_likelihood_ratio"]
    if x < 1:
        return "barely"
    return f"by a likelihood ratio of about {round(10 ** x):,}" if x < 3 else "by a likelihood ratio of more than 1,000"


def _q(v: float, quantity: str) -> str:
    if quantity == "strain":
        return f"{fmt(v * 1e6)} microstrain"
    return f"{fmt(v)} {'mm' if quantity == 'displacement' else 'C' if quantity == 'temperature' else ''}".strip()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _rms(xs):
    xs = [x for x in xs if x is not None]
    return float(np.sqrt(np.mean(np.square(xs)))) if xs else None


def headline(r: dict) -> str:
    s, t = r["summary"], r["tests"]
    out = f"{s['readings']} readings from {t['file']}"
    out += ", which are synthetic and do not come from a real test. " if t.get("synthetic") else ". "
    out += f"{s['within']} of {s['readings']} agree with the simulation within twice their validation uncertainty"
    if s["mean_relative_error"] is not None:
        out += f", and the comparison error averages {100 * s['mean_relative_error']:+.1f}% of the reading"
    out += "."
    p = r["u_pooled"]
    if p["readings"] >= 3:
        fit = "within" if p["area"] <= p["reference_95"] else "outside"
        out += (f" The pooled area metric is {p['area']:.3f}, {fit} the {p['reference_95']:.3f} a model with the "
                "right spread would stay under 95% of the time.")
    mf = r.get("model_form")
    if mf:
        other = next(x["idealization"] for x in mf["idealizations"] if x["idealization"] != mf["favours"])
        out += f" The readings favour {mf['favours']} over {other} {ratio_words(mf)}, with the modulus fitted to each."
    for c in (r.get("calibration") or {}).get("inputs", []):
        u = f" {c['unit']}" if c["unit"] and c["unit"] != "x" else ""
        out += (f" Calibration puts {c['input']} at {fmt(c['mean'])}{u} (95% interval {fmt(c['ci95'][0])} to "
                f"{fmt(c['ci95'][1])}{u}, {c['narrowed']:.2g}x narrower than before the test).")
    return out


__all__ = ["run_validation", "ReadingError", "ModelError", "CircuitPredictor"]
