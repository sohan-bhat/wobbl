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


def vibration(out: Path, mesh, modes, picks: list[int], labels: dict, verdict: str, headline: str, title: str,
              question: str = "") -> Path | None:
    """Each picked mode plays two cycles in turn. A mode shape has no size of its own, so each is drawn with its
    largest motion at 6% of the part and colored by how much each spot moves relative to that."""
    try:
        from ..viz import Scene

        X, tris = _surface(mesh)
        span = float(np.linalg.norm(np.ptp(X, 0)))
        F = 32
        m = len(picks)
        t = np.concatenate([k + np.arange(F) / F for k in range(m)])
        s = Scene(title, domain="vibration", question=question)
        s.timeline(t, unit="", label="mode", seconds=3.0 * m, between="interpolated",
                   note="each mode plays two cycles, far slower than it really moves")
        wave = np.sin(2 * np.pi * 2 * np.arange(F) / F)
        for k, i in enumerate(picks):
            phi = modes.shapes[i, :len(X)].astype(float)
            phi /= max(float(np.linalg.norm(phi, axis=1).max()), 1e-30)
            scale = np.zeros(len(t))
            scale[k * F:(k + 1) * F] = wave
            part = s.mesh(X, tris, name=f"mode {i + 1}", max_vertices=5000)   # one copy of the part per mode
            part.deform(phi * 0.06 * span, scale=scale)
            part.color_by("relative motion", np.linalg.norm(phi, axis=1), scale=np.abs(scale), colormap="viridis",
                          range=[0, 1], digits=2)
            part.show(k, k + 1 - 0.5 / F)
            status = labels.get(i, "")
            kind = {"resonant": "fail", "close": "warn"}.get(status, "info")
            word = {"resonant": "inside or right next to an excitation", "close": "close to an excitation",
                    "clear": "clear of every excitation", "barely excited": "near an excitation but barely driven"}
            s.event(float(k), f"mode {i + 1}, {fmt(modes.freq[i])} Hz", word.get(status, ""), kind)
        s.readout("natural frequency", np.repeat([float(modes.freq[i]) for i in picks], F), unit="Hz")
        s.note("A mode shape has no size of its own, so each is drawn with its largest motion at 6% of the part")
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None


def buckling(out: Path, mesh, mode1: np.ndarray, measure: np.ndarray, lam: float, design: float,
             yield_factor: float, load_N: float, e0: float, why: str, allowable: float, verdict: str, headline: str,
             title: str, question: str = "") -> Path | None:
    """The load climbing to 95% of the buckling load, the stress scaled with it exactly, and the first buckling
    shape growing from a starting crookedness e0 as e0 / (1 - P / Pcr)."""
    try:
        from ..viz import Scene

        X, tris = _surface(mesh)
        span = float(np.linalg.norm(np.ptp(X, 0)))
        lamf = np.linspace(0.0, 0.95 * lam, 48)
        bow = e0 / (1.0 - lamf / lam)
        phi = np.asarray(mode1[:len(X)], float)
        phi /= max(float(np.linalg.norm(phi, axis=1).max()), 1e-30)
        s = Scene(title, domain="buckling", question=question)
        s.timeline(lamf, unit="times the load", label="load factor", seconds=8.0, between="interpolated")
        part = s.mesh(X, tris, name="part")
        part.deform(phi, scale=bow, exaggerate=max(1.0, 0.05 * span / max(float(bow[-1]), 1e-30)))
        part.color_by("stress", np.asarray(measure[:len(X)], float), scale=lamf, unit="MPa", limit=allowable,
                      limit_label="strength limit")
        s.readout("load", lamf * load_N, unit="N")
        s.readout("load factor", lamf, digits=3)
        s.readout("bow", bow, unit="mm", digits=3)
        if math.isfinite(yield_factor) and yield_factor < 0.95 * lam:
            s.event(float(yield_factor), "first yield", "the most stressed point reaches its limit", "warn")
        if design < 0.95 * lam:
            s.event(float(design), "design limit", "the load the knockdown and yielding allow", "info")
        s.note(f"The bow starts from {fmt(e0)} mm of crookedness ({why}) and grows as 1 / (1 - P / Pcr), the classic "
               "estimate for an imperfect part, not a collapse simulation")
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None


def fatigue(out: Path, source: str, peak: float, blocks: list, curve, model: str, repeats: float, frames: dict,
            verdict: str, headline: str, title: str, question: str = "") -> Path | None:
    """The part colored by the damage each spot takes as the duty runs. A spot's stresses are the judged peak's
    scaled by its share of the peak in the strength result, with the same notch sensitivity, and its damage grows
    over the duty in the same proportion as the peak's."""
    try:
        import pyvista as pv

        from ..viz import Scene
        from . import fatigue as F

        vtu = Path(source) / "result.vtu"
        if not source or not vtu.exists() or peak <= 0:
            return None
        grid = pv.read(str(vtu))
        try:
            surf = grid.extract_surface(algorithm="dataset_surface")
        except TypeError:      # an older pyvista without the keyword
            surf = grid.extract_surface()
        ids = np.asarray(surf.point_data["vtkOriginalPointIds"])
        meas = np.asarray(grid.point_data["failure_measure_MPa"], float)[ids]
        X = np.asarray(surf.points, float)
        tris = surf.faces.reshape(-1, 4)[:, 1:]
        ratio = np.clip(meas / peak, 0.0, None)
        # damage over the whole duty at a grid of stress ratios, then per spot by interpolation in log space
        grid_r = np.linspace(0.0, max(1.2, float(ratio.max())), 64)
        d_grid = repeats * np.array([F.damage(blocks, curve, model, scale=r)[0] for r in grid_r])
        logd = np.log10(np.maximum(d_grid, 1e-30))
        d_total = 10 ** np.interp(ratio, grid_r, logd)
        d_total[d_total < 1e-29] = 0.0
        cyc, dmg = np.asarray(frames["cycles"], float), np.asarray(frames["damage"], float)
        keep = np.unique(np.rint(np.linspace(0, len(cyc) - 1, min(60, len(cyc)))).astype(int))
        share = dmg[keep] / max(float(dmg[-1]), 1e-300)
        s = Scene(title, domain="fatigue", question=question)
        s.timeline(cyc[keep], unit="cycles", label="cycles", seconds=8.0, between="interpolated")
        part = s.mesh(X, tris, name="part")
        part.color_by("fatigue damage", np.minimum(d_total, 2.0), scale=share, colormap="heat", range=[0, 1],
                      limit=1.0, limit_label="a crack starts")
        s.readout("damage at the peak", dmg[keep], digits=3, limit=1.0)
        if dmg[-1] >= 1:
            k = int(np.argmax(dmg >= 1))
            s.event(float(cyc[k]), "a crack starts", "Miner's sum reaches one at the judged peak", "fail")
        s.note("Away from the judged peak each spot's stress is scaled from the strength result with the same notch "
               "sensitivity, and lives scatter by a factor of three to ten")
        return _finish(s, out, verdict, headline)
    except Exception:  # noqa: BLE001  a scene is a convenience, never a reason to fail
        _debug()
        return None
