"""A strength run as a scene: the part in its setting, the load coming on, the stress rising and the part bending.

The model is linear elastic, so stress and displacement at a fraction of the load are exactly that fraction of the
solved ones, and every frame of the ramp is exact. Supports become the wall, the bolts or the plate they stand
for, and loads become arrows at their real faces, with a shelf, a weight or a motor where the question names one.
Where the stress passes the material's limit the part turns white, at the place the solver found.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene

FRAMES = 60


def _tris(mesh, mask):
    t = mesh.tris[mask, :3]
    X = mesh.X
    a, b, c = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(2 * area, 1e-30)[:, None]
    return t, n, area, (a + b + c) / 3


def _components(t: np.ndarray) -> list[np.ndarray]:
    """Groups of triangles that share nodes, such as the separate holes of one selection."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in t:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(a) for a in t[:, 0]])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _cylinder(X, t, n, area):
    """Axis, center, radius and length of a hole's wall from its triangles: normals lie square to the axis."""
    M = (n * area[:, None]).T @ n
    w, V = np.linalg.eigh(M)
    axis = V[:, 0]
    pts = X[np.unique(t)]
    c0 = pts.mean(0)
    s = (pts - c0) @ axis
    radial = (pts - c0) - np.outer(s, axis)
    # the center is where the radial offsets balance; a least squares circle fit on the projected points
    u, v, _ = P.basis(axis)
    x, y = radial @ u, radial @ v
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    center = c0 + sol[0] * u + sol[1] * v + axis * (s.min() + s.max()) / 2
    r = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-12)))
    return axis, center, r, float(np.ptp(s)), float(w[0] / max(w[2], 1e-30))


def _selection(run, info: dict):
    from ...geometry.select import Selection
    regions = [tuple([r[0], *[np.asarray(x) if isinstance(x, list) else x for x in r[1:]]]) for r in info.get("patch", [])]
    return Selection(info.get("selector", ""), list(info.get("faces", [])), regions).tri_mask(run.mesh)


def write_scene(run, o, out_dir) -> Path | None:
    """Build and save the scene of a finished strength run. Returns the file written."""
    mesh = run.mesh
    X = mesh.X
    lo, hi = X.min(0), X.max(0)
    span = float(np.linalg.norm(hi - lo))
    size = float(np.ptp(X, 0).max())
    centre = (lo + hi) / 2
    words = set(re.findall(r"[a-z]+", (o.question or "").lower()))
    ramp = np.linspace(0.0, 1.0, FRAMES)
    measure = np.asarray(run.measure, float)
    U = np.asarray(run.result.U, float)
    gov, allow = float(o.governing_MPa), float(o.allowable_MPa or 0) or float(measure.max())
    g = np.asarray(o.governing_xyz, float)
    node = int(np.argmin(np.linalg.norm(X - g, axis=1)))
    brittle = "principal" in (o.measure or "")
    limit_word = "strength" if brittle else "yield"

    loads = [ld for ld in run.study.loads]
    first = next((ld for ld in loads if ld.kind in ("force", "remote", "bolt")), None)
    total = sum(ld.magnitude for ld in loads if ld.kind in ("force", "remote", "bolt"))
    unit = "N" if first is not None else ""
    name = (Path(o.part).stem or "part").replace("_", " ").replace("-", " ")
    s = Scene(f"{name.capitalize()} under {total:g} N" if total > 0 else f"{name.capitalize()} under its load",
              domain="strength", question=o.question,
              source="Frames scale the solved stress and displacement by the load, which is exact for this linear "
                     "elastic model.")
    if total > 0:
        s.timeline(total * ramp, unit="N", label="load", seconds=6, between="exact")
    else:
        s.timeline(100 * ramp, unit="%", label="load", seconds=6, between="exact")
    part = s.mesh(X, mesh.tris, name="part", keep=[node], max_vertices=14000)
    part.color_by(o.measure.split(" (")[0] if o.measure else "stress", measure, unit="MPa", scale=ramp,
                  range=(0.0, allow), limit=allow, limit_label=limit_word)
    umax = float(np.linalg.norm(U, axis=1).max())
    ex = 0.08 * span / umax if umax > 0 else 1.0
    ex = float(f"{max(ex, 1.0):.2g}")
    part.deform(U, scale=ramp, exaggerate=ex)

    supports(s, run, X, size, words)
    load_props(s, run, X, size, ramp, words)

    if total > 0:
        s.readout("load", total * ramp, unit="N", digits=3)
    s.readout("peak stress", gov * ramp, unit="MPa", limit=allow)
    s.readout("safety factor", np.minimum(np.where(ramp > 0, allow / np.maximum(gov * ramp, 1e-30), 99), 99), digits=3)
    s.readout("deflection", float(o.max_disp_mm) * ramp, unit="mm")
    for h in o.hotspots or []:
        if h.get("excluded") and h.get("values") and h["values"][-1] > gov:
            s.label("excluded corner, a mesh artifact", h["xyz"], kind="note")
            break
    sf = float(o.safety_factor)
    if sf < 1 and gov > 0:
        at = sf * (total if total > 0 else 100)
        s.event(at, "first yield" if not brittle else "crack starts", f"{o.governing_where or 'at the peak'}",
                kind="fail")
        s.label(f"{'yields' if not brittle else 'breaks'} here, {{peak stress}}", g, kind="fail", t0=at)
        s.label("peak {peak stress}", g, kind="peak", t1=at)
    else:
        s.label("peak {peak stress}", g, kind="peak")
    word = {"holds": "HOLDS", "marginal": "MARGINAL", "fails": "FAILS"}.get(o.verdict, o.verdict.upper())
    tone = {"holds": "pass", "marginal": "warn", "fails": "fail"}.get(o.verdict, "info")
    s.verdict(word, _line(o, total, sf, brittle), tone)
    from ...report.render import _surface_normal_at
    from ..camera import clear_direction
    view = clear_direction(s, g, _surface_normal_at(X, mesh.tris, g), X, mesh.tris, size)
    s.camera(target=centre, direction=view, distance=0.5 * span / np.sin(np.radians(15)) * 1.15, fov=30)
    return s.save(out_dir)


def _line(o, total, sf, brittle) -> str:
    """The verdict in one plain sentence."""
    place = re.sub(r"\s*\(face \d+\)|\s*at \([^)]*\)", "", o.governing_where or "").strip()
    place = place if place and not re.fullmatch(r"on face \d+", place) else "at its most stressed spot"
    load = f"the {total:g} N load" if total > 0 else "the load"
    if o.verdict == "fails":
        return f"It {'cracks' if brittle else 'yields'} {place} at about {sf * 100:.0f}% of {load}."
    if o.verdict == "marginal":
        return f"It holds {load}, but its safety factor of {sf:.2f} is under the target of {o.target_sf:g}."
    return f"It holds {load} with a safety factor of {sf:.2f}, peaking {place}."


def supports(s, run, X, size, words):
    """The wall, plate or bolts each support stands for, placed on its real faces."""
    wall_side = None
    holes = []
    for k, fx in enumerate(run.study.fixtures):
        info = run.selections.get(f"fix{k + 1}")
        if not info:
            continue
        mask = _selection(run, info)
        if not mask.any():
            continue
        t, n, area, c = _tris(run.mesh, mask)
        mean = (n * area[:, None]).sum(0) / max(area.sum(), 1e-30)
        flat = np.linalg.norm(mean) > 0.9
        if flat and fx.kind != "bolt":
            nn = P.unit(mean)
            at = (c * area[:, None]).sum(0) / area.sum()
            vertical = abs(nn[2]) > 0.8
            if not vertical:
                word = "wall"
            elif nn[2] < 0:
                word = next((w for w in ("table", "desk", "bench", "floor", "ground") if w in words), "plate")
            else:
                word = "ceiling"
            big = size * (2.2 if word in ("wall", "floor", "ground", "ceiling") else 1.4)
            thick = size * (0.12 if word in ("wall", "floor", "ground", "ceiling") else 0.06)
            # the support's face meets the part's, facing back at it, and its body runs away from the part
            P.slab(s, at, -nn, big, big * (0.8 if not vertical else 1), thick,
                   color=P.WOOD if word in ("table", "desk", "bench") else P.WALL, name=word)
            if not vertical:
                wall_side = (nn, at)
        else:
            for comp in _components(t):
                if area[comp].sum() <= 0:
                    continue
                axis, center, r, length, flat_ratio = _cylinder(run.mesh.X, t[comp], n[comp], area[comp])
                if flat_ratio < 0.2 and 0 < r < size:
                    holes.append((axis, center, r, length))
    for axis, center, r, length in holes:
        # the head goes on the side away from the wall, or from the part's own middle when there is no wall
        side = axis if wall_side is None else -wall_side[0]
        away = np.sign(np.dot(side, axis)) or 1.0
        if wall_side is None:
            mid = X.mean(0)
            away = np.sign(np.dot(center - mid, axis)) or 1.0
            if abs(axis[2]) > 0.8:          # vertical holes: the part sits on a plate below and the heads are on top
                away = 1.0 if axis[2] > 0 else -1.0
        extra = r * 2.5
        P.bolt(s, center - P.unit(axis) * away * extra / 2, axis, r, length + extra, head=away)
    if holes and wall_side is None:
        axis = holes[0][0]
        if abs(axis[2]) > 0.8:
            zmin = float(X[:, 2].min())
            lo, hi = X.min(0), X.max(0)
            P.slab(s, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin], [0, 0, 1], size * 1.2, size * 1.2, size * 0.05,
                   color=P.STEEL, name="mounting plate")


def load_props(s, run, X, size, ramp, words):
    """Each load as an arrow at its faces, with what the question says it is: a shelf, a weight or a motor."""
    for k, ld in enumerate(run.study.loads):
        if ld.kind == "gravity":
            P.arrow(s, X.mean(0), [0, 0, -1], size * 0.35, ramp, "own weight", name="gravity")
            continue
        info = run.selections.get(f"load{k + 1}")
        mask = _selection(run, info) if info else None
        if mask is None or not mask.any():
            continue
        t, n, area, c = _tris(run.mesh, mask)
        at = (c * area[:, None]).sum(0) / area.sum()
        nn = P.unit((n * area[:, None]).sum(0))
        d = P.unit(ld.direction) if ld.direction is not None else -nn
        label = "{load}"
        if ld.kind == "remote" and ld.point is not None:
            p = np.asarray(ld.point, float)
            if words & {"motor", "belt", "pulley", "stepper", "nema"}:
                P.motor(s, at, nn, p, float(np.ptp(X[np.unique(t)], 0).max()) or size * 0.42)
            else:
                s.lines([np.array([at, p])], color="#a3a7ad", width=1.5, name="rigid link", prop=True)
                s.body("sphere", [size * 0.025], pos=p, color=P.STEEL, name="load point", prop=True)
            P.arrow(s, p, d, size * 0.45, ramp, label)
            continue
        if ld.kind == "pressure":
            pts = c[np.argsort(-area)[: min(9, len(c))]]
            for i, q in enumerate(pts):
                P.arrow(s, q, -nn, size * 0.18, ramp, label if i == 0 else "", name="pressure")
            continue
        down = d[2] < -0.7 and nn[2] > 0.7
        ext = np.ptp(X[np.unique(t)], 0)
        if down and words & {"shelf", "shelves", "board", "plank"}:
            top = float(X[np.unique(t), 2].max())
            shelf(s, X, at, top, size, ext)
            board = size * 0.12
            P.block(s, [at[0], at[1], top + board], [0, 0, 1], size * 0.12, size * 0.16, size * 0.22,
                    color=P.WEIGHT, name="load on the shelf")
            P.arrow(s, [at[0], at[1], top + board + size * 0.22], d, size * 0.35, ramp, label)
        elif down:
            top = float(X[np.unique(t), 2].max())
            hu, hv = max(ext[0] / 2, size * 0.06), max(ext[1] / 2, size * 0.06)
            h = size * 0.18
            P.block(s, [at[0], at[1], top], [0, 0, 1], hu, hv, h, name="weight")
            P.arrow(s, [at[0], at[1], top + h], d, size * 0.35, ramp, label)
        else:
            P.arrow(s, at, d, size * 0.4, ramp, label)


def shelf(s, X, at, top, size, ext):
    """A shelf board resting on the loaded top face, running long across the bracket and as deep as the arm, kept
    clear of any part of the bracket that rises above the face."""
    along = 1 if ext[1] <= ext[0] else 0          # the board runs along the bracket's narrow side
    deep = 1 - along
    on = np.abs(X[:, 2] - top) < 0.02 * size
    a0, a1 = float(X[on, deep].min()), float(X[on, deep].max())
    above = X[:, 2] > top + 0.05 * size
    if above.any():
        b0, b1 = float(X[above, deep].min()), float(X[above, deep].max())
        a0, a1 = (b1, a1) if a1 - b1 >= b0 - a0 else (a0, b0)
    c = np.array([0.0, 0.0, top])
    c[deep] = (a0 + a1) / 2
    c[along] = float(X[:, along].mean())
    half = [0.0, 0.0]
    half[deep], half[along] = (a1 - a0) / 2, size * 1.6
    # block() lays its first size along -y and its second along x when standing on +z with the x hint
    # see through, so the bracket under it shows
    P.block(s, c, [0, 0, 1], half[1], half[0], size * 0.12, color=P.WOOD, name="shelf", hint=(1, 0, 0), opacity=0.45)
