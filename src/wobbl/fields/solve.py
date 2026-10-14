"""Run a GetDP problem on a mesh and read its result files back."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..solvers import base as solvers


def getdp():
    a = solvers.get("getdp")
    a.require()
    return a


def run(pro: str, mesh_path: str | Path, work: str | Path, name: str = "model", resolution: str = "M",
        postop: str = "P", timeout: float = 600, verbosity: int = 3) -> str:
    return getdp().run(pro, Path(mesh_path), work, resolution, postop, name, timeout, verbosity)


def read_global(work: str | Path, fname: str) -> np.ndarray:
    """Values of a global quantity printed with Format Table, without the leading time column."""
    vals = Path(work, fname).read_text().split()
    return np.array([float(v) for v in vals[1:]])


def read_complex(work: str | Path, fname: str) -> complex:
    v = read_global(work, fname)
    return complex(v[0], v[1] if len(v) > 1 else 0.0)


def read_nodes(work: str | Path, fname: str, node_ids: np.ndarray, comps: int = 1) -> np.ndarray:
    """A node table (one or more values per node) in the order of the mesh's nodes; NaN where absent."""
    tok = Path(work, fname).read_text().split()
    n = int(tok[0])
    data = np.array(tok[1:1 + n * (comps + 1)], float).reshape(n, comps + 1)
    pos = {int(t): i for i, t in enumerate(node_ids)}
    out = np.full((len(node_ids), comps), np.nan)
    for row in data:
        i = pos.get(int(row[0]))
        if i is not None:
            out[i] = row[1:]
    return out[:, 0] if comps == 1 else out


def read_elements(work: str | Path, fname: str, nodes_per_elem: int = 3, comps: int = 1) -> np.ndarray:
    """Values printed with OnElementsOf in Table format: one row per element, the value at each of its nodes.
    Each line holds the element type and number, node coordinates and then the values, which come last."""
    rows = []
    for ln in Path(work, fname).read_text().splitlines():
        t = ln.split()
        if t:
            rows.append([float(v) for v in t[-nodes_per_elem * comps:]])
    return np.array(rows)


def element_map(work: str | Path, fname: str, X: np.ndarray, tris: np.ndarray, nodes_per_elem: int = 3) -> np.ndarray:
    """An element table's mean value on each of the mesh's triangles, matched by centroid (NaN where absent)."""
    from scipy.spatial import cKDTree

    cents, vals = [], []
    for ln in Path(work, fname).read_text().splitlines():
        t = ln.split()
        if not t:
            continue
        xyz = np.array(t[2:2 + 3 * nodes_per_elem], float).reshape(nodes_per_elem, 3)
        cents.append(xyz[:, :2].mean(0))
        vals.append(np.mean([float(v) for v in t[-nodes_per_elem:]]))
    out = np.full(len(tris), np.nan)
    if not cents:
        return out
    dist, idx = cKDTree(np.array(cents)).query(X[tris].mean(1))
    ok = dist < 1e-9 + 1e-6 * np.ptp(X, axis=0).max()
    out[ok] = np.array(vals)[idx[ok]]
    return out


def peak_element(work: str | Path, fname: str, nodes_per_elem: int = 3) -> tuple[float, tuple[float, float]]:
    """The largest value in an element table and the centroid of the element it sits in."""
    best, where = -np.inf, (0.0, 0.0)
    for ln in Path(work, fname).read_text().splitlines():
        t = ln.split()
        if not t:
            continue
        v = max(float(x) for x in t[-nodes_per_elem:])
        if v > best:
            xyz = np.array(t[2:2 + 3 * nodes_per_elem], float).reshape(nodes_per_elem, 3)
            best, where = v, (float(xyz[:, 0].mean()), float(xyz[:, 1].mean()))
    return float(best), where


def read_point(work: str | Path, fname: str) -> np.ndarray:
    """A value printed OnPoint in Table format: the components after the point's coordinates."""
    t = Path(work, fname).read_text().split()
    return np.array([float(v) for v in t[8:]])


def residual(log: str) -> tuple[int, float]:
    """Iterations and final relative residual of a nonlinear solve (0, 0 for a linear one)."""
    hits = re.findall(r"Residual\s+(\d+): abs\s+(\S+) rel\s+(\S+)", log)
    if not hits:
        return 0, 0.0
    it, _, rel = hits[-1]
    return int(it), float(rel)


def eigenvalues(log: str) -> list[complex]:
    """Eigenvalues (omega squared) GetDP reports, in its order."""
    out = []
    for re_, im in re.findall(r"Eigenvalue\s+\d+: w\^2 = (\S+) ([+-] \S+) \* i", log):
        out.append(complex(float(re_), float(im.replace(" ", ""))))
    return out
