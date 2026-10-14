"""How each kind of result turns uncertain inputs into its answer, as cheaply as the physics allows.

Strength. A Wobbl strength study is linear elastic, with one material held by supports that do not move and
loaded by forces, so its answer scales exactly. Stress grows in proportion to the load and does not depend on
Young's modulus, deflection grows with the load and falls with the modulus, and the safety factor is the strength
over the stress. Load, stiffness and strength scatter therefore need no new solve and carry no surrogate error.
A parameter of a parametric .py part changes the shape, so it gets new studies at a small design of values and a
polynomial chaos fit of the stress and deflection ratios, which still scale exactly with the rest.

Heat and circuits. Each design point is a new solve, on the converged mesh for heat and with ngspice for a
circuit template, and the answer is a polynomial chaos fit of those solves with its leave one out error.

Every model also carries the numerical uncertainty its own checks measured: the last change over the mesh levels
for strength and heat, and the half time step difference for a circuit.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..units import fmt
from . import uq
from .uq import Input, Known

STRENGTH_ALIASES = {"e": "E", "modulus": "E", "young": "E", "youngs": "E", "stiffness": "E", "force": "load",
                    "loads": "load", "uts": "strength", "allowable": "strength", "limit": "strength"}


class ModelError(ValueError):
    pass


def load_outcome(result_dir) -> dict:
    p = Path(result_dir) / "outcome.json"
    if not p.exists():
        raise ModelError(f"no Wobbl result in '{result_dir}' (no outcome.json)")
    return json.loads(p.read_text())


def for_result(result_dir) -> "Model":
    d = load_outcome(result_dir)
    cls = {"strength": StrengthModel, "thermal": ThermalModel, "circuit": CircuitModel}.get(d.get("domain"))
    if cls is None:
        raise ModelError(f"cannot carry uncertainty through a '{d.get('domain')}' result")
    return cls(Path(result_dir), d)


def _check(d: dict, name: str) -> dict:
    return next((c for c in d.get("checks", []) if c["name"] == name), {})


class Model:
    domain = ""
    decision = ""                    # the output judged against the limit
    outputs: list[str] = []
    units: dict[str, str] = {}

    def __init__(self, result_dir: Path, outcome: dict):
        self.dir, self.o = Path(result_dir), outcome
        self.known: dict[str, Known] = {}
        self.aliases: dict[str, str] = {}

    def parse(self, texts: list[str]) -> list[Input]:
        inputs = [uq.parse_vary(t, self.known, self.aliases) for t in texts]
        names = [i.name for i in inputs]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise uq.VaryError(f"{', '.join(sorted(dup))} is varied twice")
        return inputs

    def numerical(self) -> Input | None:
        """The discretization error as one more input: a factor on the answer with the spread the checks found."""
        return None

    def evaluator(self, inputs: list[Input], progress=None, solves: int | None = None):
        """A cheap function from an (n, d) array of input values to (n, len(outputs)), and what it is."""
        raise NotImplementedError

    def fails(self, X: np.ndarray, Y: np.ndarray, inputs: list[Input]) -> dict:
        """Boolean failure arrays by name, each with the limit it was judged against."""
        raise NotImplementedError

    def alternatives(self) -> list[dict]:
        """Other idealizations Wobbl solved, which bound a modeling choice no distribution describes."""
        return []

    def _col(self, X, inputs, name, default):
        for j, inp in enumerate(inputs):
            if inp.name == name:
                return X[:, j]
        return np.full(len(X), float(default))


def _numeric_factor(rel: float, label: str) -> Input | None:
    if not rel or not np.isfinite(rel) or rel <= 0:
        return None
    rel = max(float(rel), 1e-4)
    return Input("mesh", "normal", 1.0, rel, unit="x", nominal=1.0, role="numerical", label=label,
                 text=f"mesh {100 * rel:.2g}%")


# --- strength -----------------------------------------------------------------------
class StrengthModel(Model):
    domain = "strength"
    decision = "safety_factor"
    outputs = ["safety_factor", "governing_MPa", "max_deflection_mm"]
    units = {"safety_factor": "", "governing_MPa": "MPa", "max_deflection_mm": "mm"}

    def __init__(self, result_dir, outcome):
        super().__init__(result_dir, outcome)
        o = outcome
        self.study = json.loads((self.dir / "study.json").read_text()) if (self.dir / "study.json").exists() else {}
        loads = self.study.get("loads", [])
        self.gov0, self.disp0, self.allow0 = o["governing_MPa"], o["max_disp_mm"], o["allowable_MPa"]
        self.target = o["target_sf"]
        from ..materials import resolve_material

        try:
            self.mat = resolve_material(self.study.get("material") or o["material"])
        except Exception:  # noqa: BLE001  an old result whose material no longer resolves still scales
            self.mat = None
        self.E0 = self.mat.E if self.mat else float("nan")
        ductile = self.mat.ductile if self.mat else "Mises" in o.get("measure", "")
        self.limit_name = "yield" if ductile else "tensile"
        self.aliases = dict(STRENGTH_ALIASES, **{"yield": self.limit_name, "tensile": self.limit_name,
                                                 "strength": self.limit_name})
        if len(loads) == 1 and loads[0].get("kind") != "gravity":
            ld = loads[0]
            unit = {"pressure": "MPa", "moment": "N mm"}.get(ld["kind"], "N")
            self.load_nominal = float(ld["magnitude"])
            self.known["load"] = Known(self.load_nominal, unit, "load", f"the {fmt(self.load_nominal)} {unit} load")
        else:
            self.load_nominal = 1.0
            what = "self weight" if loads and all(x.get("kind") == "gravity" for x in loads) else "every load"
            self.known["load"] = Known(1.0, "x", "load", f"a factor on {what}")
        if self.mat:
            self.known["E"] = Known(self.E0, "MPa", "stiffness", "Young's modulus")
        self.known[self.limit_name] = Known(self.allow0, "MPa", "strength",
                                            "yield strength" if ductile else "tensile strength")
        self.geometry = {}
        from ..geometry.parametric import is_parametric, load_params

        part = self.study.get("part", "")
        if part and is_parametric(part):
            path = (self.dir / part).resolve() if not Path(part).is_absolute() else Path(part)
            if path.exists():
                params = {**load_params(path), **{k: float(v) for k, v in self.study.get("params", {}).items()}}
                for k, v in params.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in self.known:
                        self.geometry[k] = path
                        self.known[k] = Known(float(v), "mm", "geometry", f"the {k} of the part")

    def numerical(self) -> Input | None:
        rel = _check(self.o, "mesh convergence").get("detail", {}).get("last_change")
        return _numeric_factor(rel, "discretization error, the last change over the mesh levels")

    def evaluator(self, inputs, progress=None, solves=None):
        geo = [i for i in inputs if i.role == "geometry"]
        ratio, info = None, {"kind": "exact", "method": "exact scaling of the converged solution: stress follows the "
                             "load and not the modulus, deflection follows the load over the modulus"}
        if geo:
            ratio, sur = self._geometry_surrogate(geo, progress, solves)
            info = {"kind": "polynomial chaos", "method": f"polynomial chaos of degree {sur.degree} fitted to "
                    f"{sur.n_train} new studies across the {', '.join(i.name for i in geo)} range, leave one out "
                    f"error {100 * max(sur.loo):.2g}% of the variance, times the exact scaling for load, stiffness "
                    "and strength", "surrogate": sur.summary()}
        names = [i.name for i in inputs]

        def f(X):
            X = np.atleast_2d(X)
            a = self._col(X, inputs, "load", self.load_nominal) / self.load_nominal
            e = self.E0 / self._col(X, inputs, "E", self.E0) if "E" in names else 1.0
            y = self._col(X, inputs, self.limit_name, self.allow0)
            m = self._col(X, inputs, "mesh", 1.0)
            g, dsp = (np.ones(len(X)), np.ones(len(X))) if ratio is None else ratio(X[:, [names.index(i.name)
                                                                                           for i in geo]])
            gov = self.gov0 * a * m * g
            return np.column_stack([y / gov, gov, self.disp0 * a * e * dsp])
        return f, info

    def _geometry_surrogate(self, geo: list[Input], progress, solves):
        """New studies at a Latin hypercube of geometry values, fitted as ratios to the converged answer. Each one
        refines its own mesh until it converges, as the first study did, and selectors that name face ids are
        mapped onto the new shape the way 'wobbl fix' does."""
        import copy

        from ..geometry.remap import remap_selector
        from ..structural.pipeline import analyze_strength
        from ..structural.run import describe_part
        from ..structural.spec import StructuralStudy

        n = solves or uq.design_size(len(geo))
        X = uq.sample(geo, n, "lhs", seed=7)
        st = StructuralStudy.from_dict(self.study)
        st.part = str(self.geometry[geo[0].name])
        work = Path(tempfile.mkdtemp(prefix="wobbl-uq-"))
        try:
            old = describe_part(st.part, work / "base", st.scale, st.body, st.params)[1]
            Y = []
            for k, row in enumerate(X):
                change = {i.name: f"{float(v):.6g}" for i, v in zip(geo, row)}
                if progress:
                    progress("solve", f"geometry study {k + 1} of {n}: " +
                             ", ".join(f"{a}={b}" for a, b in change.items()))
                st2 = copy.deepcopy(st)
                st2.params = {**st.params, **change}
                pre = describe_part(st2.part, work / f"g{k}" / "work", st2.scale, st2.body, st2.params)
                for item in [*st2.loads, *st2.fixtures]:
                    if item.faces:
                        item.faces = remap_selector(item.faces, old.faces, pre[1].faces, old.bbox)
                o2 = analyze_strength(st2, work / f"g{k}", render=False, fixes=False, pre=pre)
                Y.append([o2.governing_MPa / self.gov0, o2.max_disp_mm / max(self.disp0, 1e-30)])
                shutil.rmtree(work / f"g{k}", ignore_errors=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        sur = uq.fit_pce(geo, X, np.array(Y), ["stress ratio", "deflection ratio"])
        return (lambda Xg: (sur(Xg)[:, 0], sur(Xg)[:, 1])), sur

    def fails(self, X, Y, inputs):
        sf = Y[:, 0]
        return {"failure": (sf < 1.0, f"the {self.known[self.limit_name].label}"),
                "below target": (sf < self.target, f"the target safety factor {self.target:g}")}

    def alternatives(self) -> list[dict]:
        lp = _check(self.o, "load introduction").get("detail", {})
        if not lp or "alternative_peak" not in lp:
            return []
        return [{"label": f"loads entering through a {lp['alternative']} attachment",
                 "assumed": f"a {lp['primary']} attachment", "governing_MPa": float(lp["alternative_peak"]),
                 "settled": bool(lp.get("converged") or not lp.get("refined"))}]


# --- heat ---------------------------------------------------------------------------
class ThermalModel(Model):
    domain = "thermal"
    decision = "t_max_C"
    outputs = ["t_max_C"]
    units = {"t_max_C": "C"}

    def __init__(self, result_dir, outcome):
        super().__init__(result_dir, outcome)
        p = self.dir / "study.json"
        if not p.exists():
            raise ModelError("this heat result has no study.json, so Wobbl cannot solve it again; run it again with "
                             "this version of Wobbl")
        from ..materials import resolve_material
        from ..thermal.study import Film, ThermalStudy

        d = json.loads(p.read_text())
        d["films"] = [Film(**f) for f in d.get("films", [])]
        d["fixed"] = [tuple(x) for x in d.get("fixed", [])]
        part = d["part"]
        d["part"] = str((self.dir / part).resolve()) if not Path(part).is_absolute() else part
        self.study = ThermalStudy(**{k: v for k, v in d.items() if k in ThermalStudy.__dataclass_fields__})
        self.mat = resolve_material(self.study.material)
        v = outcome["values"]
        self.t0, self.limit0 = float(v["t_max_C"]), float(v["limit_C"])
        st = self.study
        self.aliases = {"k": "conductivity", "h": "film", "convection": "film", "heat": "power", "air": "ambient",
                        "emissivity": "emissivity", "max": "limit"}
        if st.power:
            self.known["power"] = Known(st.power / 1000.0, "W", "load", "the heat made in the part")
        if len(st.films) == 1:
            self.known["film"] = Known(st.films[0].h * 1000.0, "W/m2K", "boundary", "the film coefficient")
        elif st.films:
            self.known["film"] = Known(1.0, "x", "boundary", "a factor on every film coefficient")
            for k, f in enumerate(st.films):
                self.known[f"film{k + 1}"] = Known(f.h * 1000.0, "W/m2K", "boundary", f"the film at {f.faces}")
        self.known["ambient"] = Known(st.ambient, "C", "boundary", "the air temperature", positive=False)
        self.known["conductivity"] = Known(self.mat.conductivity, "W/(m K)", "material", "the conductivity")
        if st.emissivity:
            self.known["emissivity"] = Known(st.emissivity, "", "boundary", "the emissivity")
        for k, (temp, sel) in enumerate(st.fixed):
            self.known[f"held{k + 1}" if len(st.fixed) > 1 else "held"] = Known(temp, "C", "boundary",
                                                                               f"the held temperature at {sel}", False)
        self.known["limit"] = Known(self.limit0, "C", "strength", "the temperature limit", positive=False)

    def numerical(self) -> Input | None:
        rel = _check(self.o, "mesh convergence").get("detail", {}).get("last_change")
        return _numeric_factor(rel, "discretization error, the last change in the temperature rise over the meshes")

    def study_at(self, row: dict):
        """The study with one design point's values put in."""
        import copy

        st = copy.deepcopy(self.study)
        mat = self.mat
        if "power" in row:
            st.power = row["power"] * 1000.0
        shift = row.get("ambient", st.ambient) - st.ambient
        for k, f in enumerate(st.films):
            if "film" in row:
                f.h = row["film"] / 1000.0 if len(st.films) == 1 else f.h * row["film"]
            if f"film{k + 1}" in row:
                f.h = row[f"film{k + 1}"] / 1000.0
            f.t_inf += shift
        st.ambient += shift
        if st.initial is not None:
            st.initial += shift
        if "emissivity" in row:
            st.emissivity = row["emissivity"]
        for k in range(len(st.fixed)):
            key = f"held{k + 1}" if len(st.fixed) > 1 else "held"
            if key in row:
                st.fixed[k] = (row[key], st.fixed[k][1])
        if "conductivity" in row:
            mat = mat.with_overrides(conductivity=row["conductivity"])
        return st, mat

    def solver(self, extract=None):
        """A function that solves one design point on the converged mesh and returns extract(mesh, result)."""
        from ..mesh.mesher import Sizing, build_mesh
        from ..solvers import base as solvers
        from ..structural.run import describe_part
        from ..thermal.study import build_heat

        work = Path(tempfile.mkdtemp(prefix="wobbl-uq-"))
        prep, feats = describe_part(self.study.part, work)
        mesh = build_mesh(prep, Sizing(h=float(self.o["values"]["levels"][-1]["h"])))
        adapter = solvers.for_domain("thermal")
        extract = extract or (lambda m, res: [float(res.T.max())])

        def solve(row: dict):
            st, mat = self.study_at(row)
            model, _ = build_heat(mesh, st, feats.faces, feats.bbox, mat)
            res = adapter.solve_heat(model, work / "design")
            return extract(mesh, res)
        solve.cleanup = lambda: shutil.rmtree(work, ignore_errors=True)
        return solve

    def evaluator(self, inputs, progress=None, solves=None):
        solved = [i for i in inputs if i.name not in ("limit", "mesh")]
        sur = design_surrogate(self.solver(), solved, ["t_max_C"], progress, solves, "heat solve")
        names = [i.name for i in inputs]
        idx = [names.index(i.name) for i in solved]
        amb = self.study.ambient

        def f(X):
            X = np.atleast_2d(X)
            t = sur(X[:, idx])[:, 0] if solved else np.full(len(X), self.t0)
            a = self._col(X, inputs, "ambient", amb)
            return (a + (t - a) * self._col(X, inputs, "mesh", 1.0))[:, None]
        return f, _surrogate_info(sur, solved, "heat solves on the converged mesh")

    def fails(self, X, Y, inputs):
        lim = self._col(X, inputs, "limit", self.limit0)
        return {"failure": (Y[:, 0] > lim, f"the {fmt(self.limit0)} C limit"),
                "below target": (Y[:, 0] > lim - 5.0, f"5 C of margin under the {fmt(self.limit0)} C limit")}


# --- circuits -----------------------------------------------------------------------
# what each template is judged on: the value, the parameter holding its limit, the default limit, the words
CIRCUIT_LIMITS = {"led": ("i_led_A", "imax", None, "the LED current rating"),
                  "buck": ("ripple_V", "max_ripple", None, "the ripple limit"),
                  "mosfet": ("tj_C", "tj_max", 150.0, "the junction limit")}
CIRCUIT_INPUTS = {
    "led": {"supply": ("V", "load"), "resistor": ("ohm", "material"), "vf": ("V", "material"),
            "imax": ("A", "strength")},
    "rc": {"r": ("ohm", "material"), "c": ("F", "material"), "v": ("V", "load")},
    "buck": {"vin": ("V", "load"), "current": ("A", "load"), "frequency": ("Hz", "material"),
             "inductor": ("H", "material"), "capacitor": ("F", "material"), "esr": ("ohm", "material"),
             "max_ripple": ("V", "strength")},
    "mosfet": {"current": ("A", "load"), "rds_on": ("ohm", "material"), "vbus": ("V", "load"),
               "theta_ja": ("C/W", "boundary"), "ambient": ("C", "boundary"), "switching_time": ("s", "material"),
               "tj_max": ("C", "strength")},
}


class CircuitModel(Model):
    domain = "circuit"

    def __init__(self, result_dir, outcome):
        super().__init__(result_dir, outcome)
        self.template = outcome.get("part", "")
        if self.template not in CIRCUIT_INPUTS:
            raise ModelError("uncertainty runs on the led, rc, buck and mosfet templates; a netlist has no named "
                             "values to vary")
        v = outcome["values"]
        self.params = dict(v.get("params", {}))
        lim = CIRCUIT_LIMITS.get(self.template)
        self.decision = lim[0] if lim else "v_at_tau"
        self.outputs = [self.decision]
        self.units = {"i_led_A": "A", "ripple_V": "V", "tj_C": "C", "v_at_tau": "V"}
        self.value0 = float(v[self.decision])
        defaults = {"max_ripple": 0.01 * float(self.params.get("vout", 0) or 0), "tj_max": 150.0,
                    "ambient": 25.0, "switching_time": 40e-9, "theta_ja": 62.0}
        for name, (unit, role) in CIRCUIT_INPUTS[self.template].items():
            val = self.params.get(name, defaults.get(name))
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if name == "max_ripple" and val <= 0:   # run_circuit's own default, 1% of the output
                val = defaults["max_ripple"]
            if val > 0 or name == "ambient":
                self.known[name] = Known(val, unit, role, name.replace("_", " "), positive=name != "ambient")

    def numerical(self) -> Input | None:
        m = re.search(r"\(([\d.e+-]+)% apart\)", _check(self.o, "time step convergence").get("summary", ""))
        return _numeric_factor(float(m.group(1)) / 100.0 if m else 0.0,
                               "time step error, the change when the step is halved")

    def solver(self):
        from ..circuits.run import run_circuit

        work = Path(tempfile.mkdtemp(prefix="wobbl-uq-"))
        count = [0]

        def solve(row: dict):
            count[0] += 1
            out = work / f"d{count[0]}"
            o = run_circuit(self.template, {**self.params, **row}, out)
            shutil.rmtree(out, ignore_errors=True)
            return [float(o.values[self.decision])]
        solve.cleanup = lambda: shutil.rmtree(work, ignore_errors=True)
        return solve

    def evaluator(self, inputs, progress=None, solves=None):
        lim = CIRCUIT_LIMITS.get(self.template)
        solved = [i for i in inputs if i.name != "mesh" and not (lim and i.name == lim[1])]
        sur = design_surrogate(self.solver(), solved, [self.decision], progress, solves, "ngspice run")
        names = [i.name for i in inputs]
        idx = [names.index(i.name) for i in solved]

        def f(X):
            X = np.atleast_2d(X)
            v = sur(X[:, idx])[:, 0] if solved else np.full(len(X), self.value0)
            return (v * self._col(X, inputs, "mesh", 1.0))[:, None]
        return f, _surrogate_info(sur, solved, "ngspice runs")

    def fails(self, X, Y, inputs):
        lim = CIRCUIT_LIMITS.get(self.template)
        if not lim or lim[1] not in self.known:
            return {}
        limit = self._col(X, inputs, lim[1], self.known[lim[1]].nominal)
        return {"failure": (Y[:, 0] > limit, lim[3]), "below target": (Y[:, 0] > 0.8 * limit, f"80% of {lim[3]}")}


# --- design runs --------------------------------------------------------------------
def design_surrogate(solve, inputs: list[Input], names: list[str], progress=None, solves: int | None = None,
                     what: str = "solve"):
    """Run the solver at a Latin hypercube of input values and fit polynomial chaos to the answers."""
    if not inputs:
        getattr(solve, "cleanup", lambda: None)()
        return None
    n = solves or uq.design_size(len(inputs))
    X = uq.sample(inputs, n, "lhs", seed=7)
    Y = []
    try:
        for k, row in enumerate(X):
            if progress:
                progress("solve", f"{what} {k + 1} of {n}")
            Y.append(solve({i.name: float(v) for i, v in zip(inputs, row)}))
    finally:
        getattr(solve, "cleanup", lambda: None)()
    return uq.fit_pce(inputs, X, np.array(Y), names)


def _surrogate_info(sur, solved: list[Input], what: str) -> dict:
    if sur is None:
        return {"kind": "exact", "method": "no input needs a new solve, so the converged answer is used as it is"}
    worst = max(sur.loo) if len(sur.loo) else 0.0
    return {"kind": "polynomial chaos", "surrogate": sur.summary(),
            "method": f"polynomial chaos of degree {sur.degree} fitted to {sur.n_train} {what} across "
                      f"{', '.join(i.name for i in solved)}, leave one out error {100 * worst:.2g}% of the variance"}


__all__ = ["for_result", "Model", "StrengthModel", "ThermalModel", "CircuitModel", "ModelError"]
