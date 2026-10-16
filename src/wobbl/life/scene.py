"""3D scenes of buckling, vibration and fatigue runs, written through wobbl.viz while the mesh and fields exist.

    vibration   the modes that matter play one after another, each oscillating in its own shape, with its natural
                frequency as a live readout and an event saying how near it sits to what shakes the part
    buckling    the load climbs to 95% of the buckling load: the stress, exact for the linear model, colors the part
                and the first buckling shape grows from a starting crookedness as e0 / (1 - P / Pcr)
    fatigue     the part colored by the fatigue damage each spot has taken as the duty cycle runs, when the strength
                result it reads kept its full field (result.vtu)

Every scene is a convenience: building one never fails the run.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..units import fmt

TONE = {"fails": "fail", "marginal": "warn", "holds": "pass", "ok": "pass"}


def _debug() -> None:
    """Re-raise the error being handled when WOBBL_DEBUG is set, as the viewer does."""
    import os

    if os.environ.get("WOBBL_DEBUG"):
        raise


def clear(out: Path) -> None:
    """Remove a scene left by an earlier run in the same folder, so the report never shows a stale one."""
    from ..viz import FILE

    (Path(out) / FILE).unlink(missing_ok=True)


def _finish(s, out: Path, verdict: str, headline: str) -> Path:
    """Save the scene next to the outcome. Writing the outcome afterwards puts the viewer into its report."""
    s.verdict(verdict.upper(), headline.split(". Confidence")[0], TONE.get(verdict, "info"))
    return s.save(out)


def _surface(mesh):
    n = mesh.n_nodes if hasattr(mesh, "n_nodes") else len(mesh.X)
    return np.asarray(mesh.X[:n], float), mesh.tris


def amplification(r, zeta: float):
    """Steady response of one mode against its static one, driven at r times its natural frequency."""
    r = np.asarray(r, float)
    return 1.0 / np.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2)


def speed_sweep(s, X, tris, shapes, freqs, numbers, bands, *, zeta: float, margin: float = 0.2,
                quiet=(), n: int = 41):
    """Fill scene s with the part's steady response as what shakes it speeds up. bands are (lo, hi, label) in Hz,
    each excitation moving from lo to hi together, as a rotor's once per turn and blade pass do. A band labeled
    once per turn makes the input the rotor's speed in rpm; with no bands the input is the shaking frequency from
    half the lowest mode to past the highest. shapes are mode shapes scaled to a largest motion of one.

    At each input the color is each spot's motion against its static motion, the largest amplification of any
    mode times how much that mode moves there, and the part takes the shape of the mode driven hardest. The limit
    is the amplification at half the margin from a resonance and the target at the margin, the run's own lines
    for fails and marginal. Modes in quiet move little mass and are drawn but not counted as driven."""
    freqs = np.asarray(freqs, float)
    if not bands:
        bands = [(0.5 * float(freqs.min()), 1.3 * float(freqs.max()), "shaking frequency")]
    # an even spread of speeds plus each one where a mode meets an excitation, so every resonance peak is a frame
    hits = [(f - lo) / (hi - lo) for lo, hi, _ in bands if hi > lo for f in freqs if lo <= f <= hi]
    p = np.unique(np.r_[np.linspace(0.0, 1.0, n), hits])
    n = len(p)
    fe = np.array([lo + p * (hi - lo) for lo, hi, _ in bands])                        # bands x frames
    turn = next((b for b in bands if "once per turn" in b[2]), None)
    if turn is not None:
        x, unit, label = np.round(60 * (turn[0] + p * (turn[1] - turn[0])), 1), "rpm", "rotor speed"
    else:
        x, unit, label = fe[0], "Hz", bands[0][2]
    D = amplification(fe[:, None, :] / freqs[None, :, None], zeta).max(0)              # modes x frames
    for i in range(len(freqs)):
        if numbers[i] in quiet:
            D[i] = np.minimum(D[i], 1.0)
    phi = np.stack([np.asarray(f, float) for f in shapes])                             # modes x verts x 3
    mag = np.linalg.norm(phi, axis=2)
    field = (D[:, :, None] * mag[:, None, :]).max(0)                                   # frames x verts
    lead = D.argmax(0)
    top = float(D.max())
    lim = float(amplification(1 + 0.5 * margin, zeta))
    tgt = float(amplification(1 + margin, zeta))
    worst = int(np.argmax(D.max(0)))
    s.timeline(x, unit=unit, label=label, between="interpolated", kind="input", design=float(x[worst]))
    span = float(np.linalg.norm(np.ptp(X, 0)))
    a = 0.08 * span / max(top, 1.0)
    U = np.stack([phi[lead[k]] * D[lead[k], k] * a for k in range(n)])
    part = s.mesh(X, tris, name="part", max_vertices=8000)
    part.color_by("amplification", field, unit="x static", colormap="turbo", range=[0, max(top, lim * 1.2)],
                  limit=lim, limit_label="resonant", digits=3)
    s.fields[-1]["target"] = tgt
    part.deform(U)
    s.readout("amplification", D.max(0), unit="x static", digits=3, limit=lim)
    s.readout("mode driven hardest", freqs[lead], unit="Hz", digits=4)
    for (lo, hi, lab), f in zip(bands, fe):
        if lab != label:
            s.readout(lab, f, unit="Hz", digits=4)
    for i, f in enumerate(freqs):
        for b, (lo, hi, lab) in enumerate(bands):
            if lo <= f <= hi and hi > lo and numbers[i] not in quiet:
                at = float(np.interp((f - lo) / (hi - lo), p, x))
                s.event(at, f"mode {numbers[i]} meets the {lab.split(' from ')[0]}", f"at {f:.4g} Hz", "fail")
    s.note(f"The motion is drawn as large as the amplification, up to {100 * a * top / span:.2g}% of the part, "
           "because its real size depends on how hard the rotor shakes it. Damping is "
           f"{100 * zeta:.3g}% of critical.")
    return part


def vibration(out: Path, mesh, modes, picks: list[int], labels: dict, verdict: str, headline: str, title: str,
              question: str = "", excite=(), zeta: float = 0.03, margin: float = 0.2) -> Path | None:
    """The part's steady response as the excitation speeds up through its range (see speed_sweep), from the
    modes that matter."""
    try:
        from ..viz import Scene

        X, tris = _surface(mesh)
        shapes = []
        for i in picks:
            phi = modes.shapes[i, :len(X)].astype(float)
            shapes.append(phi / max(float(np.linalg.norm(phi, axis=1).max()), 1e-30))
        s = Scene(title, domain="vibration", question=question)
        bands = [(float(e.lo), float(e.hi), e.label) for e in excite]
        speed_sweep(s, X, tris, shapes, [float(modes.freq[i]) for i in picks], [i + 1 for i in picks], bands,
                    zeta=zeta, margin=margin, quiet=[i + 1 for i in picks if labels.get(i) == "barely excited"])
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None


def load_sweep(s, X, tris, phi, lam: float, design: float, load_N: float, e0: float, *, measure=None,
               yield_factor: float = math.inf, target: float | None = None, n: int = 48):
    """Fill scene s with the part under each load from none to 95% of its buckling load. The stress is the solved
    one scaled with the load, exact for the linear model before buckling, and the first buckling shape (scaled to
    a largest motion of one) grows from a starting crookedness e0 as e0 / (1 - P / Pcr). design is the load factor
    the knockdown and yielding allow: the limit sits there and the target at design / target, so past the target
    factor the part is marginal and past the allowed load it fails."""
    top = 0.95 * lam
    P = np.unique(np.r_[np.linspace(0.0, top, n), min(1.0, top)]) * load_N
    f = P / load_N
    bow = e0 / (1.0 - f / lam)
    span = float(np.linalg.norm(np.ptp(X, 0)))
    phi = np.asarray(phi, float)
    phi = phi / max(float(np.linalg.norm(phi, axis=1).max()), 1e-30)
    word = "buckling limit" if design < yield_factor else "yield"
    s.timeline(P / 1000, unit="kN", label="load", between="interpolated", kind="input",
               design=float(f"{min(1.0, top) * load_N / 1000:.6g}"))
    part = s.mesh(X, tris, name="part")
    part.deform(phi, scale=bow, exaggerate=max(1.0, 0.05 * span / max(float(bow[-1]), 1e-30)))
    if measure is not None and np.max(measure) > 0:
        m = np.asarray(measure[:len(X)], float)
        peak = float(m.max())
        part.color_by("stress", m, scale=f, unit="MPa", range=[0, peak * design], limit=peak * design,
                      limit_label=word)
        lim_t = peak * design / target if target else None
        stress = peak * f
    else:
        b_lim = e0 / (1.0 - min(design, top) / lam)
        part.color_by("bow", np.linalg.norm(phi, axis=1), scale=bow, unit="mm", range=[0, b_lim], limit=b_lim,
                      limit_label=word)
        lim_t = e0 / (1.0 - design / target / lam) if target and design / target < lam else None
        stress = None
    if lim_t:
        s.fields[-1]["target"] = float(lim_t)
    with np.errstate(divide="ignore"):
        s.readout("factor against buckling", np.minimum(np.where(f > 0, design / np.maximum(f, 1e-30), 99), 99),
                  digits=3)
    s.readout("bow", bow, unit="mm", digits=3)
    if stress is not None:
        s.readout("peak stress", stress, unit="MPa")
    s.readout("buckling load", np.full(len(P), lam * load_N / 1000), unit="kN", digits=4)
    if design * load_N <= P[-1]:
        s.event(design * load_N / 1000, "allowed load", "the buckling load with the knockdown and yielding", "fail")
    if target and design / target * load_N <= P[-1]:
        s.event(design / target * load_N / 1000, f"target factor of {target:g}", "", "warn")
    s.note(f"The bow starts from {fmt(e0)} mm of crookedness and grows as 1 / (1 - P / Pcr), the classic estimate "
           "for an imperfect part, not a collapse simulation.")
    _end_plates(s, X)
    return part


def _end_plates(s, X) -> None:
    """A steel plate square across each end of the part's longest dimension, where a strut is held and pushed."""
    from ..viz import props as P

    lo, hi = X.min(0), X.max(0)
    ax = int(np.argmax(hi - lo))
    L = float(hi[ax] - lo[ax])
    for end in (lo[ax], hi[ax]):
        c = (lo + hi) / 2
        c[ax] = end
        nrm = np.zeros(3)
        nrm[ax] = 1 if end == lo[ax] else -1
        P.slab(s, c, nrm, L * 0.12, L * 0.12, L * 0.03, color=P.STEEL, name="end plate")


def buckling(out: Path, mesh, mode1: np.ndarray, measure: np.ndarray, lam: float, design: float,
             yield_factor: float, load_N: float, e0: float, why: str, allowable: float, verdict: str, headline: str,
             title: str, question: str = "", target: float | None = None) -> Path | None:
    """The part under each load up to 95% of its buckling load (see load_sweep)."""
    try:
        from ..viz import Scene

        X, tris = _surface(mesh)
        s = Scene(title, domain="buckling", question=question)
        load_sweep(s, X, tris, np.asarray(mode1[:len(X)], float), lam, design, load_N, e0, measure=measure,
                   yield_factor=yield_factor, target=target)
        s.note(f"The starting crookedness is {why}.")
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None


def damage_table(blocks, curve, model: str, repeats: float, sf: float, single: bool = False,
                 need: float = 1.0) -> dict:
    """The fatigue damage over the required duty with every stress scaled, from none to twice what a crack needs,
    the table a load slider reads. A single block's damage is the cycles needed over its life."""
    from . import fatigue as F

    top = 2.0 * max(1.5, 1.3 * sf) if math.isfinite(sf) else 3.0
    grid = np.linspace(0.0, top, 81)[1:]
    if single:
        d = [need / max(F.damage(blocks, curve, model, scale=float(r))[1][0], 1e-300) for r in grid]
    else:
        d = [repeats * F.damage(blocks, curve, model, scale=float(r))[0] for r in grid]
    return {"stress_scale": [round(float(r), 5) for r in grid],
            "damage": [float(f"{min(x, 1e12):.6g}") for x in d]}


def duty_sweep(s, X, tris, ratio, table: dict, *, sf: float, target: float | None = None, required: float = 1.0,
               unit: str = "repeats", where=None, n: int = 41):
    """Fill scene s with the part under every load of the duty scaled, from a fifth of it to past the load a crack
    needs. ratio is each spot's stress over the judged peak's. The color is each spot's stress over the stress
    that starts a crack in the required duty (one over its safety factor), and the readouts are the damage at the
    peak by Miner's rule, read from the table, its safety factor and its life. Past the share the target factor
    allows the part is marginal."""
    g = np.asarray(table["stress_scale"], float)
    logd = np.log10(np.maximum(np.asarray(table["damage"], float), 1e-30))

    def dmg(x):
        return 10 ** np.interp(x, np.r_[0.0, g], np.r_[-30.0, logd])

    hi = max(1.5, 1.3 * sf) if math.isfinite(sf) else 1.5
    L = np.unique(np.r_[np.linspace(0.2, min(hi, g[-1]), n), 1.0])
    ratio = np.clip(np.asarray(ratio, float), 0.0, None)
    d = dmg(L)
    s.timeline(100 * L, unit="%", label="duty load", between="interpolated", kind="input", design=100.0)
    part = s.mesh(X, tris, name="part")
    if math.isfinite(sf) and sf > 0:
        # each spot's stress over the stress that starts a crack in the duty: linear in the load, so exact
        part.color_by("share of fatigue strength", ratio / sf, scale=L, colormap="heat", range=[0, 1], limit=1.0,
                      limit_label="a crack starts", digits=3)
        if target:
            s.fields[-1]["target"] = float(1.0 / target)
    else:
        part.color_by("fatigue damage", np.minimum(dmg(L[:, None] * ratio[None, :]), 2.0), colormap="heat",
                      range=[0, 1], limit=1.0, limit_label="a crack starts", digits=3)
    s.readout("damage", d, digits=3, limit=1.0)
    if math.isfinite(sf):
        s.readout("safety factor", sf / L, digits=3)
    with np.errstate(divide="ignore"):
        s.readout("life", np.minimum(required / np.maximum(d, 1e-30), 1e12), unit=unit, digits=3)
    if math.isfinite(sf) and L[0] < sf < L[-1]:
        s.event(100 * sf, "a crack starts", "Miner's sum reaches one at the judged peak", "fail")
    if target and math.isfinite(sf) and L[0] < sf / target < L[-1]:
        s.event(100 * sf / target, f"target factor of {target:g}", "", "warn")
    if where is not None:
        s.label("damage {damage}", where, kind="peak")
    s.note("Away from the judged peak each spot's stress is scaled from the strength result with the same notch "
           "sensitivity, and lives scatter by a factor of three to ten.")
    return part


def source_surface(source, where=None):
    """The part's surface from the strength run a fatigue study reads, and each vertex's stress over the judged
    peak's: from its result.vtu when kept, else from its scene, peaked at the vertex nearest where. Returns X, tris,
    ratio and the source scene's camera (or None), or None when neither is there."""
    src = Path(source)
    vtu = src / "result.vtu"
    if vtu.exists():
        import pyvista as pv

        grid = pv.read(str(vtu))
        try:
            surf = grid.extract_surface(algorithm="dataset_surface")
        except TypeError:      # an older pyvista without the keyword
            surf = grid.extract_surface()
        ids = np.asarray(surf.point_data["vtkOriginalPointIds"])
        meas = np.asarray(grid.point_data["failure_measure_MPa"], float)[ids]
        X, tris, cam = np.asarray(surf.points, float), surf.faces.reshape(-1, 4)[:, 1:], None
    else:
        from ..viz.scene import FILE, array, load

        if not (src / FILE).exists():
            return None
        h, blob = load(src / FILE)
        part = next((ob for ob in h["objects"] if ob["kind"] == "mesh" and ob.get("name") == "part"), None)
        if part is None or not part.get("field"):
            return None
        X, tris = array(h, blob, part["pos"]), array(h, blob, part["tri"])
        meas = np.asarray(array(h, blob, part["field"]["values"]), float)
        if meas.ndim > 1:
            meas = meas[-1]
        sc = part["field"].get("scale")
        if isinstance(sc, dict):
            sc = array(h, blob, sc)
        if sc is not None and np.size(sc):
            meas = meas * float(np.ravel(sc)[-1])
        cam = h.get("camera") or None
    if where is not None:
        ref = float(meas[int(np.argmin(np.linalg.norm(X - np.asarray(where, float), axis=1)))])
    else:
        ref = float(meas.max())
    return X, tris, meas / max(ref, 1e-30), cam


def fatigue(out: Path, source: str, table: dict, sf: float, target: float | None, required: float, unit: str,
            where, verdict: str, headline: str, title: str, question: str = "") -> Path | None:
    """The part colored by the damage each spot takes over the required duty, as the duty's loads are scaled (see
    duty_sweep), on the strength result the study reads."""
    try:
        from ..viz import Scene

        got = source_surface(source, where) if source else None
        if got is None:
            return None
        X, tris, ratio, cam = got
        s = Scene(title, domain="fatigue", question=question)
        duty_sweep(s, X, tris, ratio, table, sf=sf, target=target, required=required, unit=unit, where=where)
        if cam:
            s.camera(target=cam.get("target"), direction=cam.get("dir"), distance=cam.get("dist"),
                     fov=cam.get("fov", 30))
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None
