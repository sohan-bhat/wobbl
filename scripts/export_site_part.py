"""Export the website's motor mount as three unit loads the browser combines.

The hero lets visitors point a load on the motor shaft any way in 3D and change
its size. The model is linear elastic, so a load of F newtons along the unit
direction u is exactly F * (u1 * case1 + u2 * case2 + u3 * case3). This script
takes the three unit cases from scripts/solve_site_directions.py (same mesh,
node for node), keeps the outer surface, and writes the full stress tensor and
the displacement of each case at every surface node, so the page can rebuild
the tensor for any direction and take von Mises from it. Von Mises itself does
not add up across loads.

    python scripts/export_site_part.py --cases /tmp/dirs \\
        --example examples/cast-motor-mount --name cast-motor-mount --out site/public/data

Nothing is interpolated or decimated: the surface is the solver's own second
order surface, every node keeps its own values, and the browser splits each
curved triangle into four as VTK does. What makes it small is coding:

- Triangles are stored by their three corner nodes only, in a walk across the
  surface, as distances back from the next new node (a new node is 0). Each
  edge's midside node is numbered the first time the walk crosses that edge.
- Every field is an integer on a fixed step. A midside node is stored as its
  difference from the mean of its edge's two corners, which on a smooth field
  is close to zero. A corner is stored as its difference from the corners
  already known in the triangle that brings it in.
- The nodes that are, or come within a few percent of, the peak for some
  direction carry a second, finer step for the stress, so the readout matches
  the solver's peak to a hundredth of a megapascal.

All streams are zigzag coded int32 split into four byte planes, then the file
is gzipped. Offsets are in the JSON manifest; the binary file's name carries a
hash of its contents, so a browser never shows an old mesh from its cache.

The peak table goes in its own small file, also named by its hash: the
governing von Mises and its node for directions every 2 degrees over half the
sphere, since a load and its opposite give the same von Mises. The page reads
the reading and the peak marker from it for any direction.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import warnings
from collections import deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# corner triples and their midside nodes for the four faces of a VTK quadratic tetra
FACES = ((0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8), (2, 0, 3, 6, 7, 9), (0, 2, 1, 6, 5, 4))


def read_case(path: Path):
    import pyvista as pv

    g = pv.read(path)
    tets = np.asarray(g.cells_dict[pv.CellType.QUADRATIC_TETRA])
    return (np.asarray(g.points, float), tets, np.asarray(g.point_data["displacement_mm"], float),
            np.asarray(g.point_data["stress_MPa"], float), np.asarray(g.point_data["excluded_zone"]) > 0.5)


def boundary(X: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Outward quadratic boundary triangles (a, b, c, ab, bc, ca) of a tet10 mesh."""
    tris = np.concatenate([tets[:, list(f)] for f in FACES])
    opp = np.concatenate([tets[:, ({0, 1, 2, 3} - set(f[:3])).pop()] for f in FACES])
    key = np.sort(tris[:, :3], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.ravel()] == 1
    tris, opp = tris[once], opp[once]
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), X[opp] - a) > 0
    tris[inward] = tris[inward][:, [0, 2, 1, 5, 4, 3]]  # flip: (a, c, b, ca, bc, ab)
    return tris


def walk(tris: np.ndarray) -> np.ndarray:
    """Triangle order of a breadth first walk across shared edges."""
    edges = {}
    for t, (a, b, c) in enumerate(tris[:, :3]):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault((min(u, v), max(u, v)), []).append(t)
    seen = np.zeros(len(tris), bool)
    order = []
    for start in range(len(tris)):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        while q:
            t = q.popleft()
            order.append(t)
            a, b, c = tris[t, :3]
            for u, v in ((a, b), (b, c), (c, a)):
                for s in edges[(min(u, v), max(u, v))]:
                    if not seen[s]:
                        seen[s] = True
                        q.append(s)
    return np.asarray(order)


def number(tris: np.ndarray):
    """Node numbering the browser rebuilds: corners, then midside nodes, each by first use in the walk.

    Returns (corner triangles in new numbers, old node id of every new node, corner count, index codes,
    corner predictors as lists of earlier corners, midside edge ends)."""
    new = {}
    codes, preds = [], []
    ct = np.zeros((len(tris), 3), np.int64)
    for t, row in enumerate(tris):
        known = []
        for k in range(3):
            v = int(row[k])
            if v not in new:
                codes.append(0)
                new[v] = len(new)
                preds.append(list(known))
            else:
                codes.append(len(new) - new[v])
            ct[t, k] = new[v]
            known.append(new[v])
    nc = len(new)
    old = np.empty(nc, np.int64)
    for v, i in new.items():
        old[i] = v
    mids, ends = [], []
    seen = set()
    for t, row in enumerate(tris):
        for k in range(3):
            a, b = ct[t, k], ct[t, (k + 1) % 3]
            e = (min(a, b), max(a, b))
            if e not in seen:
                seen.add(e)
                mids.append(int(row[3 + k]))
                ends.append(e)
    return ct, np.concatenate([old, mids]), nc, np.asarray(codes, np.int64), preds, np.asarray(ends, np.int64)


def residuals(q: np.ndarray, nc: int, preds: list[list[int]], ends: np.ndarray) -> np.ndarray:
    """Integer field (nodes in new order, any trailing shape) to prediction residuals."""
    r = np.empty_like(q)
    for i in range(nc):
        p = preds[i]
        r[i] = q[i] - (q[i - 1] if not p and i else 0 if not p else (q[p].sum(0) // len(p)))
    r[nc:] = q[nc:] - (q[ends[:, 0]] + q[ends[:, 1]]) // 2
    return r


def decode(r: np.ndarray, nc: int, preds: list[list[int]], ends: np.ndarray) -> np.ndarray:
    """The browser's side of residuals(), to check the round trip."""
    q = np.empty_like(r)
    for i in range(nc):
        p = preds[i]
        q[i] = r[i] + (q[i - 1] if not p and i else 0 if not p else (q[p].sum(0) // len(p)))
    q[nc:] = r[nc:] + (q[ends[:, 0]] + q[ends[:, 1]]) // 2
    return q


def planes(v: np.ndarray) -> bytes:
    """Zigzag int32, split into four byte planes (low bytes of every value first)."""
    v = np.asarray(v, np.int64).reshape(len(v), -1).T.ravel()  # channel by channel
    if np.abs(v).max(initial=0) >= 2 ** 30:
        raise ValueError("a residual does not fit 31 bits; use a coarser step")
    z = ((v << 1) ^ (v >> 63)).astype("<u4")
    return z.view(np.uint8).reshape(-1, 4).T.copy().tobytes()


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = np.moveaxis(S, -1, 0)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def directions(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Unit directions in the case basis (pulley plane e1, e2 and the shaft e3) for azimuth and elevation."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def sweep(S: np.ndarray, ok: np.ndarray, dirs: np.ndarray):
    """Governing von Mises (outside excluded zones) and its node for each direction, per newton, plus how
    close each allowed node comes to the governing value for any of them. S holds the three cases'
    stresses side by side, 18 channels."""
    idx = np.flatnonzero(ok)
    A, B, C = S[idx, :6], S[idx, 6:12], S[idx, 12:]
    gov, arg, near = np.empty(len(dirs)), np.empty(len(dirs), np.int64), np.zeros(len(idx))
    for k, d in enumerate(dirs):
        vm = von_mises(d[0] * A + d[1] * B + d[2] * C)
        j = int(np.argmax(vm))
        gov[k], arg[k] = vm[j], idx[j]
        near = np.maximum(near, vm / vm[j])
    return gov, arg, idx, near


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, required=True, help="output of scripts/solve_site_directions.py")
    ap.add_argument("--example", type=Path, required=True, help="the example's output folder parent")
    ap.add_argument("--name", required=True, help="file name prefix")
    ap.add_argument("--out", type=Path, default=Path("site/public/data"))
    ap.add_argument("--stress-step", type=float, default=0.02, help="MPa at the example's load")
    ap.add_argument("--fine-every", type=int, default=20, help="fine stress step is the step over this")
    ap.add_argument("--disp-step", type=float, default=2e-4, help="mm at the example's load")
    ap.add_argument("--hot", type=float, default=0.04, help="nodes within this fraction of a peak get the fine step")
    ap.add_argument("--grid", type=float, default=2.0, help="degrees between the peak table's directions")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    info = json.loads((a.cases / "cases.json").read_text())
    cases = [read_case(a.cases / f"case{k}.vtu") for k in (1, 2, 3)]
    X, tets, ex = cases[0][0], cases[0][1], cases[0][4]
    for c in cases[1:]:
        assert np.array_equal(X, c[0]) and np.array_equal(tets, c[1]), "the cases must share one mesh"
    F = float(info["load_N"])

    tris = boundary(X, tets)
    tris = tris[walk(tris)]
    ct, old, nc, codes, preds, ends = number(tris)
    n = len(old)
    print(f"surface: {len(tris)} curved triangles, {nc} corner and {n - nc} midside nodes")

    blocks, layout = [], {}

    def add(key: str, data: bytes, **meta) -> None:
        layout[key] = {"offset": sum(len(b) for b in blocks), **meta}
        blocks.append(data + b"\0" * (-len(data) % 4))

    add("index", planes(codes), count=len(codes))
    P = X[old]
    lo, hi = P.min(0), P.max(0)
    qp = np.round((P - lo) / (hi - lo) * 65535).astype(np.int64)
    add("position", planes(residuals(qp, nc, preds, ends)), channels=3)

    # unit cases per newton, coded on steps given at the example's load
    s_step, d_step = a.stress_step / F, a.disp_step / F
    fine = s_step / a.fine_every
    S = np.concatenate([c[3] for c in cases], axis=1)[old]
    U = np.concatenate([c[2] for c in cases], axis=1)[old]
    qs = np.round(S / s_step).astype(np.int64)
    qu = np.round(U / d_step).astype(np.int64)
    add("stress", planes(residuals(qs, nc, preds, ends)), channels=18, step=s_step)
    add("disp", planes(residuals(qu, nc, preds, ends)), channels=9, step=d_step)
    excl = ex[old]
    add("excluded", excl.astype(np.uint8).tobytes(), count=n)

    # Von Mises is the same for u and -u, so half the sphere covers every direction: azimuth from 0 up
    # to 180 degrees and elevation from -90 to 90.
    g = a.grid
    az_grid, el_grid = np.arange(0, 180, g), np.arange(-90, 90 + g / 2, g)
    AZ, EL = np.meshgrid(az_grid, el_grid, indexing="ij")
    table = directions(AZ.ravel(), EL.ravel())

    # nodes near the peak for some direction get the fine step
    ok = ~excl
    caz, cel = np.meshgrid(np.arange(0, 180, 3.0), np.arange(-90, 91, 3.0), indexing="ij")
    _, _, idx, near = sweep(S, ok, directions(caz.ravel(), cel.ravel()))
    hot = idx[near >= 1 - a.hot]
    qf = np.round(S[hot] / fine).astype(np.int64) - qs[hot] * a.fine_every
    add("hot", planes(np.diff(hot, prepend=0)), count=len(hot))
    add("hot_stress", planes(qf), channels=18, step=fine)
    print(f"{len(hot)} nodes carry the fine stress step")

    # what the browser will compute, from the coded values
    assert np.array_equal(decode(residuals(qs, nc, preds, ends), nc, preds, ends), qs)
    Sb = qs * s_step
    Sb[hot] = (qs[hot] * a.fine_every + qf) * fine
    g_true, i_true, _, _ = sweep(S, ok, table)
    g_web, i_web, _, _ = sweep(Sb, ok, table)
    err = np.abs(g_web - g_true).max() * F
    print(f"coded peaks differ from the solver's by at most {err:.4f} MPa at {F:g} N over {len(table)} directions; "
          f"same node for {np.mean(i_web == i_true):.0%} of them")
    t0 = math.radians(info["theta0_deg"])
    vm0 = von_mises(math.cos(t0) * Sb[:, :6] + math.sin(t0) * Sb[:, 6:12]) * F
    print(f"default pull: {vm0[ok].max():.4f} MPa (solver {info['governing_MPa']:.4f})")
    # the peak's node for every direction in the table, with its position and bending per newton as the
    # page draws them, so the readout and the marker cost nothing while the load turns
    spots, at = np.unique(i_web, return_inverse=True)
    assert len(spots) < 256
    Pb = lo + qp[spots] / 65535 * (hi - lo)
    Ub = qu[spots] * d_step
    peak_nodes = [[round(float(v), 4) for v in Pb[k]] + [float(f"{v:.6g}") for v in Ub[k]] for k in range(len(spots))]

    blob = b"".join(blocks)
    data = gzip.compress(blob, compresslevel=9, mtime=0)
    tag = hashlib.sha256(data).hexdigest()[:10]
    for stale in list(a.out.glob(f"{a.name}.*.bin.gz")) + list(a.out.glob(f"{a.name}.peaks.*.json")):
        stale.unlink()
    path = a.out / f"{a.name}.{tag}.bin.gz"
    path.write_bytes(data)
    sizes = {k: len(gzip.compress(b, 9, mtime=0)) for k, b in zip(layout, blocks)}
    print(f"{path.name}: {len(data) / 1000:.0f} KB gzipped, {len(blob) / 1000:.0f} KB raw; "
          + ", ".join(f"{k} {v / 1000:.0f}" for k, v in sizes.items()))

    # the peak table: hundredths of an MPa at load_N, and the index of the peak's node, azimuth major
    peaks = {
        "grid_deg": g, "azimuths": len(az_grid), "elevations": len(el_grid),
        "mpa_x100": np.round(g_web * F * 100).astype(int).tolist(), "node": at.ravel().tolist(),
    }
    text = json.dumps(peaks, separators=(",", ":"))
    ptag = hashlib.sha256(text.encode()).hexdigest()[:10]
    ppath = a.out / f"{a.name}.peaks.{ptag}.json"
    ppath.write_text(text)
    print(f"{ppath.name}: {len(text) / 1000:.0f} KB, {len(gzip.compress(text.encode(), 9)) / 1000:.0f} KB gzipped")

    outcome = json.loads((a.example / "output" / "outcome.json").read_text())
    acc = json.loads((a.cases / "accuracy.json").read_text()) if (a.cases / "accuracy.json").exists() else []
    manifest = {
        "name": a.name,
        "file": path.name,
        "peaks": ppath.name,
        "bytes_gzip": len(data),
        "load_N": F,
        "allowable_MPa": outcome["allowable_MPa"],
        "target_sf": outcome["target_sf"],
        "measure": outcome["measure"],
        "verdict": outcome["verdict"],
        "governing_MPa": round(float(outcome["governing_MPa"]), 3),
        "governing_xyz": [round(v, 3) for v in outcome["governing_xyz"]],
        "pull": {
            "point": info["point"], "axis": info["axis"], "e1": info["e1"], "e2": info["e2"], "e3": info["e3"],
            "default_deg": round(info["theta0_deg"], 3),
            "note": "Linear elastic: a load of F N along u is F * (u1 * case1 + u2 * case2 + u3 * case3), "
                    "u in the basis e1, e2 (pulley plane) and e3 (shaft).",
        },
        "mesh": {"triangles": len(tris), "corners": nc, "nodes": n, "bounds_min": lo.round(5).tolist(),
                 "bounds_max": hi.round(5).tolist()},
        "layout": layout,
        # the default pull's governing von Mises and the range over every direction, MPa at load_N
        "default_peak_MPa": round(float(vm0[ok].max()), 3),
        "peak_range_MPa": [round(float(g_web.min() * F), 3), round(float(g_web.max() * F), 3)],
        # where the peak can be: x, y, z in mm, then case 1, 2 and 3 displacement in mm per newton
        "peak_nodes": peak_nodes,
        "converged": acc,
    }
    (a.out / f"{a.name}.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
