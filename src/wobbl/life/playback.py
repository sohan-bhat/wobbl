"""playback.json: what a viewer needs to animate a life result truthfully, written by the run itself.

A few hundred kilobytes at most: the part's surface, decimated to a few thousand vertices that are all original
mesh nodes, so every field is sampled exactly, plus the motion to play.

    vibration   each mode shape that matters, with its frequency, for x(t) = x + a phi sin(2 pi f t),
                and the excitation bands, the sine sweep response and the mass that each mode moves
    buckling    the first buckling shape and how it grows with load, from an initial crookedness that the
                amplification factor 1 / (1 - P / Pcr) magnifies, the classic estimate for an imperfect part
    fatigue     damage as it builds up over the duty cycle by Miner's rule, block by block

Arrays are base64 of little endian typed data, described by dtype, shape and, for quantized ones, the scale and
offset that turn the stored integers back into values: value = offset + scale * stored.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

MAX_VERTICES = 2500


def encode(arr: np.ndarray, dtype: str, scale=None, offset=None) -> dict:
    a = np.ascontiguousarray(arr).astype("<" + np.dtype(dtype).str[1:])
    d = {"dtype": dtype, "shape": list(arr.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}
    if scale is not None:
        d["scale"] = scale
    if offset is not None:
        d["offset"] = offset
    return d


def quantize(values: np.ndarray, dtype: str = "int16") -> dict:
    """Symmetric quantization to signed integers: value = scale * stored."""
    top = float(np.abs(values).max()) if values.size else 0.0
    lim = np.iinfo(dtype).max
    scale = top / lim if top > 0 else 1.0
    return encode(np.round(values / scale).astype(dtype), dtype, scale=scale)


def surface(mesh, max_vertices: int = MAX_VERTICES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The outer surface as (vertices, triangles, node index of each vertex), decimated when it is large. Decimation
    only removes vertices, so each one left is an original mesh node and carries its fields exactly."""
    tris = mesh.tris[:, :3]
    nodes, inv = np.unique(tris.ravel(), return_inverse=True)
    faces = inv.reshape(-1, 3)
    pts = mesh.X[nodes]
    if len(nodes) > max_vertices:
        try:
            import pyvista as pv

            poly = pv.PolyData(pts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
            red = poly.decimate_pro(1.0 - max_vertices / len(nodes), preserve_topology=True)
            from scipy.spatial import cKDTree

            _, idx = cKDTree(pts).query(np.asarray(red.points))
            keep = red.faces.reshape(-1, 4)[:, 1:]
            nodes, pts, faces = nodes[idx], pts[idx], keep
        except Exception:  # noqa: BLE001  a viewer can still play the full surface, only a larger file
            pass
    return pts, faces, nodes


def mesh_block(pts: np.ndarray, faces: np.ndarray) -> dict:
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    q = np.round((pts - lo) / span * 65535).astype("uint16")
    return {"vertices": encode(q, "uint16", scale=(span / 65535).round(9).tolist(), offset=lo.round(6).tolist()),
            "triangles": encode(faces, "uint16" if faces.max() < 65535 else "uint32"),
            "units": "mm"}


def write(path: Path, doc: dict) -> str:
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))
    return str(path)


def modes_doc(mesh, modes, picks: list[int], labels: dict, values: dict) -> dict:
    pts, faces, nodes = surface(mesh)
    out = []
    for i in picks:
        phi = modes.shapes[i, nodes].astype(float)
        phi = phi / max(float(np.linalg.norm(phi, axis=1).max()), 1e-30)
        out.append({"mode": i + 1, "frequency_Hz": round(float(modes.freq[i]), 4), "status": labels.get(i, ""),
                    "mass_share_xyz": [round(float(x), 4) for x in values["mass_fraction"][i]],
                    "shape": quantize(phi, "int8")})
    doc = {"format": "wobbl-playback", "version": 1, "kind": "vibration",
           "about": "Mode shapes of the part as it is held, each scaled to a largest motion of one. Play a mode as "
                    "x(t) = x + a * shape * sin(2 pi f t), with a about amplitude_mm and f slowed down for the eye. "
                    "the real motion is far smaller and far faster.",
           "units": {"length": "mm", "frequency": "Hz", "stress": "MPa"},
           "mesh": mesh_block(pts, faces), "amplitude_mm": round(0.06 * float(np.linalg.norm(np.ptp(pts, 0))), 3),
           "modes": out, "excitations": values.get("resonance_bands", [])}
    h = values.get("harmonic")
    if h and "curve" in values:
        doc["sine_sweep"] = {"frequency_Hz": values["curve"]["f"], "stress_MPa": values["curve"]["stress"],
                             "peak_Hz": round(h["frequency_Hz"], 4), "peak_MPa": round(h["peak_MPa"], 4),
                             "at_mm": [round(v, 3) for v in h["xyz"]]}
    if values.get("random"):
        r = values["random"]
        doc["random"] = {"three_sigma_MPa": round(r["three_sigma_MPa"], 4), "at_mm": [round(v, 3) for v in r["xyz"]],
                         "grms": round(r["grms"], 4)}
    return doc


def buckling_doc(mesh, shape: np.ndarray, factor: float, design: float, yield_factor: float, length: float,
                 imperfection: float, why: str) -> dict:
    """The first buckling shape, and its growth with the load factor lam from an initial crookedness e0 shaped like
    it: w(lam) = e0 / (1 - lam / lam1), drawn up to 95% of the buckling load."""
    pts, faces, nodes = surface(mesh)
    phi = shape[nodes].astype(float)
    phi = phi / max(float(np.linalg.norm(phi, axis=1).max()), 1e-30)
    lam = np.linspace(0.0, 0.95 * factor, 40)
    grow = imperfection / (1.0 - lam / factor)
    return {"format": "wobbl-playback", "version": 1, "kind": "buckling",
            "about": "The first buckling shape, scaled to a largest motion of one, and how far a part that starts "
                     f"crooked by {imperfection:.3g} mm in that shape ({why}) has bowed at each load factor, "
                     "e0 / (1 - P / Pcr). That is the classic estimate for an imperfect part and not a collapse "
                     "simulation, and it grows without bound at the buckling load.",
            "units": {"length": "mm", "load_factor": "times the applied load"},
            "mesh": mesh_block(pts, faces), "shape": quantize(phi, "int8"),
            "frames": {"load_factor": [round(float(x), 5) for x in lam],
                       "bow_mm": [round(float(x), 5) for x in grow]},
            "buckling_factor": round(factor, 5), "design_factor": round(design, 5),
            "yield_factor": None if not np.isfinite(yield_factor) else round(float(yield_factor), 5),
            "length_mm": round(length, 3)}


def fatigue_doc(blocks: list, lives: list, repeats: float, single: bool, required: float, values: dict) -> dict:
    """Damage over the duty cycle: each block adds n / N in turn, repeated, until the duty is done or D = 1."""
    t, d, label = [0.0], [0.0], []
    total, cycles = 0.0, 0.0
    runs = int(min(max(1, round(repeats)), 200)) if not single else 1
    for r in range(runs):
        for b, n in zip(blocks, lives):
            steps = 5
            for k in range(1, steps + 1):
                dn = b.cycles / steps
                cycles += dn
                total += dn / n if n > 0 and np.isfinite(n) else 0.0
                t.append(cycles)
                d.append(total)
                label.append(b.label or "load")
    return {"format": "wobbl-playback", "version": 1, "kind": "fatigue",
            "about": "Fatigue damage by Miner's rule as the cycles add up, block after block. A crack is predicted "
                     "to start where the damage reaches one, and lives scatter by a factor of three to ten around it.",
            "units": {"cycles": "load cycles", "damage": "fraction of life", "stress": "MPa"},
            "frames": {"cycles": [round(x, 3) for x in t], "damage": [round(x, 8) for x in d], "block": label},
            "blocks": [{"label": b.label or "load", "amplitude_MPa": round(b.amplitude, 4), "mean_MPa": round(b.mean, 4),
                        "cycles": b.cycles, "life": None if not np.isfinite(n) else round(n, 1)}
                       for b, n in zip(blocks, lives)],
            "required": required, "safety_factor": round(values["sf"], 4) if np.isfinite(values["sf"]) else None,
            "where_mm": values.get("xyz")}
