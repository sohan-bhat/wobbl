"""Export each strength and heat example as a small playback file for its page.

Every strength and heat run on the examples pages plays back from its own results.
This script reads the fields Wobbl saved for each run (work/fields.npz for a strength
run, fields.npz for a heat run; both stay out of git, so run the examples first or
point --examples at a checkout that has), keeps the part's outer surface, and writes
it with its fields in the coding scripts/export_site_part.py uses for the landing
page's part: a walk across the surface, integers on fixed steps, prediction residuals
split into byte planes, then gzip. The page decodes it with part.js.

    python scripts/sitegen/export_playback.py --examples ~/code/wobbl/examples

What each kind of run plays, and why it is exact:

- Strength. The models are linear elastic (the shelf bracket's wall contact has no gap
  and no friction, which scales the same way), so at a fraction k of the run's load
  every stress and displacement is k times the run's. The file holds the run's own
  stress measure and displacement at every surface node, and the page multiplies them
  by k while the load ramps from zero.
- Heat over time (the battery pack). The run saved the whole temperature field at 40
  times and starts at the air temperature, as Wobbl's own history does. The file holds
  every saved frame and the page blends between neighbors.
- Heat at steady state (the heat sink). The run solved the steady state only. The warm
  up comes from Wobbl's lumped check of the same sink (convection and radiation on a
  body of the run's volume, surface and material), which this script first checks
  against the steady skin temperature the run reported. The page scales the steady
  temperature rise by the lumped rise at each moment; with a Biot number of 3e-5 the sink
  stays within a few degrees of uniform all the way.

Where a heat run's hottest point is inside the part (the battery pack's is in the middle
of the block), the surface alone looks one temperature, so the file also holds a section:
the corner of the part toward the camera is cut away at the hottest point, and the three
cut faces carry the run's own volume field, sampled on a grid with the elements' second
order shape functions at every saved time. Every heat run also gets an isotherm spacing,
about a tenth of its final spread, so the page can draw the gradient as lines.

A surface with more than --max-nodes second order nodes keeps its corner nodes only,
as flat triangles; the parts here are mostly flat faces, so they look the same.

Writes site/public/data/play/<example>.json with what the page needs to know and one
<example>-<run>.<hash>.bin.gz per run, named by its contents so browsers never keep an
old one.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "sitegen"), str(ROOT / "scripts"), str(ROOT / "src")]
from export_site_part import boundary, decode, number, planes, residuals, walk  # noqa: E402
from wobbl.report.render import _surface_normal_at, _view_dir  # noqa: E402

STRENGTH = ["cast-motor-mount", "shelf-bracket", "motor-mount", "enclosure", "drone-arm"]
HEAT = ["heat-sink", "battery-pack"]
NICE = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000)
SIGMA = 5.670374e-8


def commands(folder: Path) -> list[list[str]]:
    """The wobbl commands in a run.sh as argument lists, with its shell variables filled in."""
    text = (folder / "run.sh").read_text().replace("\\\n", " ")
    env: dict[str, str] = {}
    out = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.+)$", line)
        if m:
            env[m[1]] = shlex.split(m[2])[0]
            continue
        if not line.startswith("wobbl "):
            continue
        for k, v in env.items():
            line = re.sub(r"\$\{?%s\}?" % k, lambda _m, v=v: v, line)
        out.append(shlex.split(line))
    return out


def opt(argv: list[str], *names: str) -> str | None:
    for i, a in enumerate(argv[:-1]):
        if a in names:
            return argv[i + 1]
    return None


def number_of(text: str, unit: str) -> float:
    m = re.match(r"\s*(-?[\d.]+)\s*" + unit, text)
    if not m:
        raise ValueError(f"expected a value in {unit}: {text}")
    return float(m[1])


def surface(X: np.ndarray, tets: np.ndarray, max_nodes: int):
    """The outer surface in walk order: node ids (old numbering), corner count, the index codes,
    the corner predictors and the midside edge ends, and whether it was cut to corners only."""
    tris = boundary(X, tets)
    tris = tris[walk(tris)]
    ct, old, nc, codes, preds, ends = number(tris)
    linear = len(old) > max_nodes
    if linear:
        old, ends = old[:nc], np.zeros((0, 2), np.int64)
    return {"tris": len(tris), "old": old, "nc": nc, "codes": codes, "preds": preds, "ends": ends, "linear": linear}


class Blob:
    """Blocks of a playback file, each 4 byte aligned, with their offsets for the manifest."""

    def __init__(self):
        self.blocks: list[bytes] = []
        self.layout: dict[str, dict] = {}

    def add(self, key: str, data: bytes, **meta) -> None:
        self.layout[key] = {"offset": sum(len(b) for b in self.blocks), **meta}
        self.blocks.append(data + b"\0" * (-len(data) % 4))

    def field(self, key: str, q: np.ndarray, s: dict, **meta) -> None:
        """An integer field on the surface's nodes, coded as residuals; checks the round trip."""
        r = residuals(q, s["nc"], s["preds"], s["ends"])
        assert np.array_equal(decode(r, s["nc"], s["preds"], s["ends"]), q), key
        self.add(key, planes(r), channels=1 if q.ndim == 1 else q.shape[1], **meta)

    def write(self, out: Path, stem: str) -> tuple[str, int]:
        data = gzip.compress(b"".join(self.blocks), compresslevel=9, mtime=0)
        name = f"{stem}.{hashlib.sha256(data).hexdigest()[:10]}.bin.gz"
        (out / name).write_bytes(data)
        return name, len(data)


def geometry(blob: Blob, X: np.ndarray, s: dict) -> dict:
    P = X[s["old"]]
    lo, hi = P.min(0), P.max(0)
    blob.add("index", planes(s["codes"]), count=len(s["codes"]))
    qp = np.round((P - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype(np.int64)
    blob.field("position", qp, s)
    return {"triangles": s["tris"], "corners": s["nc"], "nodes": len(s["old"]), "linear": s["linear"],
            "bounds_min": lo.round(5).tolist(), "bounds_max": hi.round(5).tolist()}


def camera(X: np.ndarray, tets: np.ndarray, focus) -> dict:
    """The direction Wobbl's own heatmaps look from (report/render.py), the point they look at and the
    part's radius, so the page frames the part as the report does."""
    tris = boundary(X, tets)
    g = np.asarray(focus, float)
    view = _view_dir(X, g, _surface_normal_at(X, tris, g))
    lo, hi = X.min(0), X.max(0)
    return {"view": np.round(view / np.linalg.norm(view), 5).tolist(), "center": np.round(X.mean(0), 3).tolist(),
            "radius": round(float(np.linalg.norm(hi - lo) / 2), 3)}


def nearest(X: np.ndarray, p) -> int:
    return int(np.argmin(np.linalg.norm(X - np.asarray(p, float), axis=1)))


def arrow(o: dict, X: np.ndarray, U: np.ndarray) -> dict | None:
    """Where the load goes in and which way, for the arrow: the run's total applied force, at the load's
    point ("at x,y,z", a pull from a point off the part) or the middle of its patch (a push on a face)."""
    f = np.asarray(o["levels"][-1]["applied"], float)
    if np.linalg.norm(f) < 1e-9:
        return None
    text = next((t for t in o["loads"] if re.match(r"\s*[\d.]+\s*N\b(?!\*)", t)), o["loads"][0])
    m = re.search(r"\bat\s+(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)", text)
    if m:
        point, pull = [float(v) for v in m.groups()], True
    else:
        sel = o["selections"].get("load1", {})
        sphere = next((p for p in sel.get("patch", []) if p[0] == "sphere"), None)
        if sphere is None:
            return None
        point, pull = sphere[1], False
    i = nearest(X, point)
    return {"point": [round(v, 3) for v in point], "dir": np.round(f / np.linalg.norm(f), 5).tolist(), "pull": pull,
            "disp": [float(f"{v:.6g}") for v in U[i]]}


def strength_run(folder: Path, rel: str, out: Path, stem: str, max_nodes: int, exag: int | None, label: str) -> dict:
    o = json.loads((folder / "outcome.json").read_text())
    z = np.load(folder / "work" / "fields.npz")
    X, tets, U, M, ex = z["X"], z["tets"], z["U"], z["measure"], z["excluded"].astype(bool)
    gov = float(o["governing_MPa"])
    gi = int(np.argmax(np.where(ex, -np.inf, M)))
    if abs(M[gi] - gov) > 1e-4 * gov:
        raise SystemExit(f"{rel}: fields.npz peaks at {M[gi]:.4f} MPa but outcome.json says {gov:.4f}; run it again")
    s = surface(X, tets, max_nodes)
    blob = Blob()
    mesh = geometry(blob, X, s)
    old = s["old"]
    v_step, u_top = gov / 1000, float(np.abs(U).max())
    u_step = u_top / 2000
    blob.field("value", np.round(M[old] / v_step).astype(np.int64), s, step=v_step)
    blob.field("disp", np.round(U[old] / u_step).astype(np.int64), s, step=u_step)
    blob.add("excluded", ex[old].astype(np.uint8).tobytes(), count=len(old))
    name, size = blob.write(out, stem)
    span = float(np.linalg.norm(X.max(0) - X.min(0)))
    umax = float(np.linalg.norm(U, axis=1).max())
    if exag is None:  # the largest round factor that bends the part by at most 6% of its size
        exag = max([n for n in NICE if n * umax <= 0.06 * span] or [1])
    gxyz = o["governing_xyz"]
    loads = [t.split(" @ ")[0] for t in o["loads"]]
    return {
        "kind": "strength", "label": label, "file": name, "bytes": size, "mesh": mesh, "layout": blob.layout,
        "governing_MPa": round(gov, 4), "governing_xyz": [round(v, 3) for v in gxyz],
        "governing_disp": [float(f"{v:.6g}") for v in U[nearest(X, gxyz)]],
        "allowable_MPa": o["allowable_MPa"], "target_sf": o["target_sf"], "safety_factor": round(o["safety_factor"], 4),
        "measure": o["measure"], "verdict": o["verdict"], "max_disp_mm": round(float(o["max_disp_mm"]), 4),
        "loads": [re.sub(r"^(\S+ \S+).*", r"\1", t).replace("N*m", "N m") for t in loads],
        "exag": exag, "camera": camera(X, tets, gxyz), "arrow": arrow(o, X, U),
    }


def lumped_warmup(folder: Path, argv: list[str], X: np.ndarray, tets: np.ndarray, o: dict) -> dict:
    """Wobbl's lumped check of a steady heat run, integrated over time from the air temperature."""
    film = opt(argv, "--film") or ""
    if not film.endswith("@ all"):
        raise SystemExit(f"{folder}: the warm up needs one film on every face, not {film!r}")
    h = number_of(film, "W/m2K")
    power = number_of(opt(argv, "--power") or "", "W")
    air = number_of(opt(argv, "--ambient") or "", "C")
    eps = float(opt(argv, "--radiation") or 0)
    m = re.search(r"specific heat ([\d.]+) J/\(kg K\), density ([\d.]+) kg/m3", o["assumptions"][0])
    cp, rho = float(m[1]), float(m[2])
    tris = boundary(X, tets)
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 1e-6       # m2
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6 * 1e-9                  # m3

    def shed(t):
        return h * area * (t - air) + eps * SIGMA * area * ((t + 273.15) ** 4 - (air + 273.15) ** 4)

    steady = air + power / (h * area)
    for _ in range(60):
        steady -= (shed(steady) - power) / (h * area + 4 * eps * SIGMA * area * (steady + 273.15) ** 3)
    want = next(c["detail"]["lumped_surface_C"] for c in o["checks"] if c["name"] == "hand calculation")
    if abs(steady - want) > 0.01:
        raise SystemExit(f"{folder}: the lumped model gives {steady:.4f} C, the run's check {want:.4f} C")
    cap, temp, t, dt = rho * cp * vol, air, 0.0, 1.0
    ts, theta = [0.0], [0.0]
    while (temp - air) / (steady - air) < 0.999:
        k1 = (power - shed(temp)) / cap
        k2 = (power - shed(temp + 0.5 * dt * k1)) / cap
        temp += dt * k2
        t += dt
        if t % 15 == 0:
            ts.append(t)
            theta.append(round((temp - air) / (steady - air), 5))
    ts.append(t)
    theta.append(round((temp - air) / (steady - air), 5))
    return {"t_s": ts, "theta": theta, "lumped_C": round(steady, 4), "tau_s": round(cap / (h * area), 1)}


def locate(X: np.ndarray, tets: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the second order tet that holds it and the ten shape function weights there (VTK node
    order: four corners, then the midsides of 01, 12, 20, 03, 13 and 23); tet -1 for a point outside the part."""
    x0 = X[tets[:, 0]]
    M = np.stack([X[tets[:, k]] - x0 for k in (1, 2, 3)], axis=2)        # columns are the edges from corner 0
    Minv = np.linalg.inv(M)
    owner = np.full(len(P), -1, np.int64)
    bary = np.zeros((len(P), 4))
    for a in range(0, len(tets), 400):
        b = min(a + 400, len(tets))
        L = np.einsum("tij,tpj->tpi", Minv[a:b], P[None, :, :] - x0[a:b, None, :])
        L4 = np.concatenate([1 - L.sum(2, keepdims=True), L], axis=2)
        inside = (L4 >= -1e-7).all(2)
        for t, pt in zip(*np.nonzero(inside)):
            if owner[pt] < 0:
                owner[pt], bary[pt] = a + t, L4[t, pt]
    l0, l1, l2, l3 = bary.T
    W = np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                  4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0, 4 * l0 * l3, 4 * l1 * l3, 4 * l2 * l3], axis=1)
    return owner, W


def section(X: np.ndarray, tets: np.ndarray, hist: np.ndarray, at, view, spacing: float):
    """Cut away the corner of the part on the camera's side of the point at, and sample the volume field on the
    three cut faces: a regular grid on each, values for every frame of hist (frames by nodes)."""
    p = np.asarray(at, float)
    sign = np.where(np.asarray(view) >= 0, 1, -1)
    lo, hi = X.min(0), X.max(0)
    faces, pts = [], []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        span = lambda k: (p[k], hi[k]) if sign[k] > 0 else (lo[k], p[k])  # noqa: E731
        (u0, u1), (v0, v1) = span(b), span(c)
        nu, nv = max(2, math.ceil((u1 - u0) / spacing) + 1), max(2, math.ceil((v1 - v0) / spacing) + 1)
        U, V = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0, v1, nv), indexing="ij")
        G = np.empty((nu * nv, 3))
        G[:, a], G[:, b], G[:, c] = p[a], U.ravel(), V.ravel()
        faces.append({"axis": a, "at": round(float(p[a]), 4), "u": [b, round(float(u0), 4), round(float(u1), 4), nu],
                      "v": [c, round(float(v0), 4), round(float(v1), 4), nv]})
        pts.append(G)
    P = np.concatenate(pts)
    owner, W = locate(X, tets, P)
    ok = owner >= 0
    vals = np.zeros((len(hist), len(P)))
    vals[:, ok] = np.einsum("pk,fpk->fp", W[ok], hist[:, tets[owner[ok]]])
    return {"point": p.round(4).tolist(), "sign": sign.tolist(), "faces": faces, "points": len(P)}, P, vals, ok


def heat_run(folder: Path, argv: list[str], rel: str, out: Path, stem: str, max_nodes: int, label: str) -> dict:
    o = json.loads((folder / "outcome.json").read_text())
    z = np.load(folder / "fields.npz")
    X, tets, T = z["X"], z["tets"], z["T"]
    v = o["values"]
    if abs(T.max() - v["t_max_C"]) > 1e-3:
        raise SystemExit(f"{rel}: fields.npz peaks at {T.max():.4f} C but outcome.json says {v['t_max_C']:.4f}")
    air = number_of(opt(argv, "--ambient") or "25 C", "C")
    start = number_of(opt(argv, "--initial") or f"{air} C", "C")
    s = surface(X, tets, max_nodes)
    blob = Blob()
    mesh = geometry(blob, X, s)
    old = s["old"]
    step = 0.01
    transient = len(z["hist_t"]) > 1
    run = {"kind": "heat", "label": label, "mesh": mesh, "air_C": air, "limit_C": v["limit_C"],
           "t_max_C": round(float(v["t_max_C"]), 4), "hot_xyz": [round(c, 3) for c in v["hot_xyz"]],
           "verdict": o["verdict"], "camera": camera(X, tets, v["hot_xyz"]),
           # the hottest point gets a marker only where it is on the surface the page draws
           "marker": bool(np.linalg.norm(X[old] - np.asarray(v["hot_xyz"]), axis=1).min() < 1e-3)}
    # the run's own history: the starting state, then every saved time, as its history plot has it
    times = np.concatenate([[0.0], z["hist_t"]])
    hist = np.vstack([np.full((1, len(X)), start), z["hist_T"]]) if transient else T[None, :]
    spread = float(T.max() - T.min())
    run["iso_C"] = next(n for n in (0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50) if n >= spread / 10)
    if not run["marker"]:
        # the hottest point is inside: cut the corner away there and carry the field on the cut faces
        cut, P, vals, ok = section(X, tets, hist, v["hot_xyz"], run["camera"]["view"], run["camera"]["radius"] / 45)
        # the cut faces meet at the hottest point, where they must read what the run found
        corner = vals[-1, np.argmin(np.linalg.norm(P - np.asarray(v["hot_xyz"]), axis=1))]
        if abs(corner - v["t_max_C"]) > 0.01:
            raise SystemExit(f"{rel}: the cut reads {corner:.4f} C at the hottest point, the run {v['t_max_C']:.4f} C")
        qc = np.round((vals.T - air) / step).astype(np.int64)
        blob.add("cut_frames", planes(np.diff(qc, axis=1, prepend=0)), count=cut["points"], channels=len(hist), step=step)
        blob.add("cut_inside", ok.astype(np.uint8).tobytes(), count=cut["points"])
        run["cut"] = cut
        run["marker"] = True
    if transient:
        q = np.round((hist[:, old].T - air) / step).astype(np.int64)
        dq = np.diff(q, axis=1, prepend=0)  # each frame as its change from the one before
        blob.field("frames", dq, s, step=step, base=air)
        cross = v.get("time_over_limit_s")
        run.update({"transient": True, "times_s": times.round(3).tolist(),
                    "hottest_C": hist.max(axis=1).round(4).tolist(),
                    "cross_s": round(float(cross), 2) if cross is not None else None,
                    "scale_C": [air, float(math.ceil(max(v["t_max_C"], v["limit_C"])))]})
    else:
        blob.field("frames", np.round((T[old] - air) / step).astype(np.int64), s, step=step, base=air)
        run.update({"transient": False, "warmup": lumped_warmup(folder, argv, X, tets, o),
                    "scale_C": [air, float(math.ceil(max(v["t_max_C"], v["limit_C"])))]})
    air = still_air(folder, run["mesh"])
    if air:
        blob.add("air", air.pop("data"), count=air["points"])
        run["air"] = air
    run["file"], run["bytes"] = blob.write(out, stem)
    run["layout"] = blob.layout
    return run


def nice(x: float) -> float:
    """1, 2 or 5 times a power of ten, nearest x on a log scale, as the page's play-motion.js nice."""
    p = 10.0 ** math.floor(math.log10(x))
    return min((m * p for m in (1, 2, 5, 10)), key=lambda c: abs(math.log(c / x)))


def still_air(folder: Path, mesh: dict) -> dict | None:
    """The air a heat run in still air warms, as Wobbl's scene of the run estimates it (the lines named air, from
    wobbl/thermal/plume.py): every point's place, the air's speed there at the steady state in mm/s and its rise
    above the air in C, as float32 x, y, z, speed, rise. The page sends tracers along the lines as the part warms
    up: the plume's speed goes as the convected heat to the 1/3 and its warming to the 2/3, and that heat as the
    part's rise, so both follow the warm up's share of the steady rise. slow is how many times slower than real
    time the tracers move, the factor the page shows (play-motion.js flowSlow)."""
    p = folder / "scene.wobbl"
    if not p.exists():
        return None
    from wobbl.viz.scene import array, load
    h, b = load(p)
    o = next((o for o in h["objects"] if o.get("kind") == "lines" and o.get("name") == "air" and "speed" in o), None)
    if o is None or "field" not in o:
        return None
    P = array(h, b, o["pos"])
    sp = array(h, b, o["speed"]).ravel()
    rise = array(h, b, o["field"]["values"]).ravel()
    sc = o["field"].get("scale")
    if sc:
        # the scene's values are at its asked power; the steady rise the page warms toward is the same
        design = h["timeline"].get("design")
        t = h["timeline"]["t"]
        k = float(np.interp(design, t, sc)) if design is not None and len(t) > 1 else 1.0
        rise = rise * k
    lo, hi = np.asarray(mesh["bounds_min"]), np.asarray(mesh["bounds_max"])
    R = float(np.linalg.norm(hi - lo) / 2)
    data = np.c_[P, sp, rise].astype("<f4").tobytes()
    top = float(h["fields"][o["field"]["f"]]["range"][1])
    return {"runs": o["runs"], "points": int(len(P)), "slow": nice(float(sp.max()) / (0.3 * R)), "top": top,
            "speed_exp": 1 / 3, "rise_exp": 2 / 3, "data": data}


def runs_of(folder: Path) -> list[tuple[str, list[str], str]]:
    """(output folder, command, label) for every run in an example, fixes after the run they fix."""
    found = []
    for argv in commands(folder):
        if argv[1] == "fix":
            o = json.loads((folder / argv[2] / "outcome.json").read_text())
            for f in o.get("fixes", []):
                if f.get("after"):
                    found.append((f"{argv[2]}/{f['after']['report'].split('/')[0]}", argv, f["kind"]))
            continue
        found.append((opt(argv, "-o", "--out") or "output", argv, ""))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="examples to export (default: every strength and heat example)")
    ap.add_argument("--examples", type=Path, default=ROOT / "examples", help="where the runs' outputs are")
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "public" / "data" / "play")
    ap.add_argument("--max-nodes", type=int, default=60000, help="above this, keep corner nodes only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.names or STRENGTH + HEAT:
        src = a.examples / name
        man = {"example": name, "runs": {}}
        exag = None
        for out_rel, argv, label in runs_of(src):
            rel = f"{name}/{out_rel}"
            stem = f"{name}-{out_rel.replace('/', '-')}"
            folder = src / out_rel
            # a heat run by its command (wobbl thermal) or by what it ran (wobbl simulate --physics heat)
            done = folder / "outcome.json"
            domain = json.loads(done.read_text()).get("domain", "") if done.exists() else ""
            if argv[1] == "thermal" or domain == "thermal":
                run = heat_run(folder, argv, rel, a.out, stem, a.max_nodes, label)
            else:
                # a fix is bent as much as the part it fixes, so the two compare
                run = strength_run(folder, rel, a.out, stem, a.max_nodes, exag if label else None, label)
                if not label:
                    exag = run["exag"]
            man["runs"][rel] = run
            m = run["mesh"]
            print(f"{rel}: {m['triangles']} triangles, {m['nodes']} nodes{' (corners only)' if m['linear'] else ''}, "
                  f"{run['bytes'] / 1000:.0f} KB")
        keep = {r["file"] for r in man["runs"].values()}
        for stale in a.out.glob(f"{name}-*.bin.gz"):
            if stale.name not in keep and re.fullmatch(rf"{re.escape(name)}-[a-z0-9-]+\.[0-9a-f]{{10}}\.bin\.gz", stale.name):
                stale.unlink()
        total = sum(r["bytes"] for r in man["runs"].values())
        (a.out / f"{name}.json").write_text(json.dumps(man, separators=(",", ":")) + "\n")
        print(f"{name}: {total / 1000:.0f} KB in all")


if __name__ == "__main__":
    main()
