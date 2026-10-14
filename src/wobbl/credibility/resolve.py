"""New solves on a strength run's converged mesh, for the inputs that do not scale.

The run keeps its final mesh in work/fields.npz. Rebuilding it, and the CAD faces the loads and supports were
picked on, gives back the exact model the answer came from, so a different Poisson's ratio or one load at a time
can be solved on it without meshing or refining again. Each solve costs one CalculiX run on the final mesh.

One load at a time is what lets loads scatter independently. A linear model adds up the fields of its loads, so
with a field per load every combination of load factors is exact, and the governing stress of each sample is the
largest failure measure of the summed field, away from the zones excluded in the run.
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


class ResolveError(ValueError):
    pass


def final_mesh(fields: dict):
    """The run's converged mesh, rebuilt from the arrays saved with its fields."""
    from ..mesh.mesher import Mesh, _boundary

    X, tets, tris = fields["X"], fields["tets"], fields["tris"]
    order = 2 if tets.shape[1] == 10 else 1
    _, elem, face, key = _boundary(tets, X, order)
    n = len(X)
    s = np.sort(tris[:, :3], axis=1).astype(np.int64)
    want = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    order_k = np.argsort(key)
    pos = np.searchsorted(key, want, sorter=order_k)
    idx = order_k[np.clip(pos, 0, len(key) - 1)]
    if not np.array_equal(key[idx], want):
        raise ResolveError("the saved surface does not match the saved elements, so the mesh cannot be rebuilt")
    return Mesh(X, tets, tris, fields["tri_surf"], elem[idx], face[idx], order)


class FinalMesh:
    """Solves of variants of a strength study on its converged mesh."""

    def __init__(self, result_dir):
        from ..structural.run import describe_part
        from ..structural.spec import StructuralStudy
        from .probes import load_fields

        self.dir = Path(result_dir)
        f = load_fields(self.dir)
        if f is None or "tets" not in f or "tri_surf" not in f:
            raise ResolveError("this run kept no fields (work/fields.npz), so its final mesh cannot be solved again; "
                               "run the study again and keep its work folder")
        d = json.loads((self.dir / "study.json").read_text())
        part = d["part"]
        d["part"] = str((self.dir / part).resolve()) if not Path(part).is_absolute() else part
        self.study = StructuralStudy.from_dict(d)
        self.fields = f
        self.mesh = final_mesh(f)
        self.work = Path(tempfile.mkdtemp(prefix="wobbl-uq-"))
        self.prep, self.features = describe_part(self.study.part, self.work / "geometry", self.study.scale,
                                                 self.study.body, self.study.params)
        self.excluded = f["excluded"].astype(bool) if "excluded" in f else np.zeros(len(f["X"]), bool)

    def cleanup(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def solve(self, loads=None, nu: float | None = None):
        """Solve the study with these loads (default all) and Poisson's ratio. Returns (result, failure measure)."""
        from ..materials import resolve_material
        from ..solvers import base as solvers
        from ..structural.build import build_static
        from ..structural.run import _solve, failure_measure, resolve_selections

        st = copy.deepcopy(self.study)
        if loads is not None:
            st.loads = list(loads)
        mat = resolve_material(st.material)
        if nu is not None:
            mat = mat.with_overrides(nu=float(nu))
        sel_loads, sel_fix, _ = resolve_selections(st, self.features)
        built = build_static(self.mesh, sel_loads, sel_fix, mat, self.features.faces)
        out = self.work / "solve"
        res, _, _ = _solve(solvers.for_domain("structural"), built, out, None)
        shutil.rmtree(out, ignore_errors=True)
        axis = None
        if st.print_direction:
            from ..geometry.select import parse_vector
            axis = parse_vector(st.print_direction)
        return res, failure_measure(res, mat, axis)[0], mat, axis

    def governing(self, measure: np.ndarray) -> float:
        valid = ~self.excluded if (~self.excluded).any() else np.ones(len(measure), bool)
        return float(measure[valid].max())


class Superposition:
    """The governing stress and largest deflection for any combination of load factors, from one field per load.

    Von Mises stress squared and deflection squared are quadratic in the load factors, so each node keeps the
    products of its load fields once, and every sample's value at every node is one matrix product away, exact.
    A brittle measure (the largest principal stress) or one judged across print layers is not quadratic. For
    those, each batch of samples bounds every node from above, takes the exact value at the nodes with the
    highest bounds as a floor, and sums exactly only where a bound reaches the floor.
    """

    def __init__(self, fm: FinalMesh, progress=None):
        from ..fe import StaticResult

        self.fm = fm
        st = fm.study
        if any(fx.kind == "wall" for fx in st.fixtures):
            raise ResolveError("a wall support pushes but cannot pull, so its loads do not add up and cannot scatter "
                               "one at a time; vary them together with 'load'")
        S, U, m = [], [], []
        for k, ld in enumerate(st.loads):
            if progress:
                progress("solve", f"load {k + 1} of {len(st.loads)} alone on the final mesh")
            res, meas, self.mat, self.axis = fm.solve([ld])
            S.append(res.S)
            U.append(res.U)
            m.append(meas)
        self.S, self.U = np.array(S), np.array(U)
        self.valid = np.nonzero(~fm.excluded if (~fm.excluded).any() else np.ones(len(fm.mesh.X), bool))[0]
        self.Result = StaticResult
        n = len(S)
        self.pairs = [(i, j) for i in range(n) for j in range(i, n)]
        self.Qu = _products(self.U, self.pairs, np.ones(3))
        self.quadratic = self.mat.ductile and self.axis is None
        if self.quadratic:   # von Mises squared as sum_k w_k v_k^2, v the stress differences and shears
            s = self.S[:, self.valid]
            v = np.stack([s[..., 0] - s[..., 1], s[..., 1] - s[..., 2], s[..., 2] - s[..., 0], s[..., 3], s[..., 4],
                          s[..., 5]], axis=-1)
            self.Qs = _products(v, self.pairs, np.array([0.5, 0.5, 0.5, 3.0, 3.0, 3.0]))
        else:   # every measure is under the Frobenius norm of the stress, over the layer fraction when it applies
            s = self.S
            fro = np.sqrt((s[..., :3] ** 2).sum(-1) + 2 * (s[..., 3:] ** 2).sum(-1))
            self.bound = (fro / min(self.mat.layer_factor, 1.0) if self.axis is not None else fro)[:, self.valid]

    def __call__(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Governing stress and largest deflection for each row of load factors A, shape (n, loads)."""
        A = np.atleast_2d(A)
        gov = (np.sqrt(self._quadmax(A, self.Qs)) if self.quadratic else
               self._max(A, self.valid, self.bound, self._measure))
        return gov, np.sqrt(self._quadmax(A, self.Qu))

    def _quadmax(self, A, Q):
        terms = np.column_stack([A[:, i] * A[:, j] * (1.0 if i == j else 2.0) for i, j in self.pairs])
        step = max(1, 4_000_000 // Q.shape[1])
        return np.concatenate([np.maximum((terms[i:i + step] @ Q).max(1), 0.0) for i in range(0, len(A), step)])

    def _max(self, A, nodes, bound, value):
        out = np.empty(len(A))
        step = max(1, 4_000_000 // max(len(nodes), 1))
        k = min(16, len(nodes))
        for i in range(0, len(A), step):
            a = A[i:i + step]
            B = np.abs(a) @ bound
            top = np.unique(np.argpartition(-B, k - 1, axis=1)[:, :k])
            floor = value(a, nodes[top]).max(1)
            keep = (B >= floor[:, None] * (1 - 1e-9)).any(0)
            out[i:i + step] = self._chunked(a, nodes[keep], value)
        return out

    @staticmethod
    def _chunked(A, nodes, value):
        step = max(1, 1_000_000 // max(len(nodes), 1))
        return np.concatenate([value(A[i:i + step], nodes).max(1) for i in range(0, len(A), step)])

    def _measure(self, a, nodes):
        from ..structural.run import failure_measure

        S = _combine(a, self.S[:, nodes, :]).reshape(-1, 6)
        res = self.Result(np.zeros((len(S), 3)), S, np.zeros((len(S), 3)))
        return failure_measure(res, self.mat, self.axis)[0].reshape(len(a), len(nodes))


def _products(v: np.ndarray, pairs, w: np.ndarray) -> np.ndarray:
    """For fields v (loads, nodes, k), the weighted products sum_k w_k v_ik v_jk of each pair of loads, shaped
    (pairs, nodes)."""
    return np.array([(v[i] * v[j] * w).sum(-1) for i, j in pairs])


def _combine(a: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """Sum of the per load fields (loads, nodes, k) with factors a (samples, loads), as one matrix product."""
    lo, n, k = fields.shape
    return (a @ fields.reshape(lo, n * k)).reshape(len(a), n, k)
