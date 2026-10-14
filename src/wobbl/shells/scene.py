"""Shell runs as 3D scenes: the load coming on, the walls bending and the stress, or each ply's share of its
strength, rising until the end of the ramp. The models are linear, so every frame is the solved field scaled by
the load and the frames in between are exact."""
from __future__ import annotations

from pathlib import Path

import numpy as np

FRAMES = 40


def ramp_scene(out: Path, *, title: str, domain: str, question: str, model, values: np.ndarray, field: str,
               unit: str, U: np.ndarray, end: float, limit: float, limit_label: str, keep: int, peak_label: str,
               readouts: list[tuple], events: list[tuple], verdict: tuple[str, str, str], source: str) -> Path:
    """values and U are per shell node at the applied load; frames run from no load to end times it."""
    from ..viz import Scene
    from ..viz import props as P

    mesh = model.mesh
    lam = np.linspace(0.0, end, FRAMES)
    s = Scene(title, domain=domain, question=question, source=source)
    s.timeline(lam, unit="x load", label="load factor", seconds=6, between="exact")
    part = s.mesh(mesh.X, mesh.tris, name="walls", keep=[keep], max_vertices=12000)
    part.color_by(field, values, unit=unit, scale=lam, range=(0.0, limit), limit=limit, limit_label=limit_label)
    span = float(np.ptp(mesh.X, axis=0).max())
    umax = float(np.linalg.norm(U, axis=1).max())
    ex = float(f"{max(1.0, 0.08 * span / umax):.2g}") if umax > 0 else 1.0
    part.deform(U, scale=lam, exaggerate=ex)
    # loads as arrows at the nodes they act on, and rigid attachments at their points
    F = model.forces if model.forces is not None else np.zeros((mesh.n_nodes, 3))
    on = np.abs(F).sum(1) > 0
    if on.any() and np.linalg.norm(F.sum(0)) > 0:
        tot = F.sum(0)
        at = (mesh.X[on] * np.linalg.norm(F[on], axis=1)[:, None]).sum(0) / np.linalg.norm(F[on], axis=1).sum()
        P.arrow(s, at - 0.25 * span * P.unit(tot), tot, 0.25 * span, scale=lam / max(end, 1e-12),
                label=f"{np.linalg.norm(tot):.3g} N")
    for r in model.rigid:
        if np.linalg.norm(r.force) > 0:
            p = np.asarray(r.ref_point, float)
            P.arrow(s, p - 0.25 * span * P.unit(r.force), r.force, 0.25 * span, scale=lam / max(end, 1e-12),
                    label=f"{np.linalg.norm(r.force):.3g} N")
    for name, nodes, _ in model.constraints:
        nodes = nodes[nodes < mesh.n_nodes]
        if len(nodes):
            s.label("held", mesh.X[nodes].mean(0), kind="note")
    for name, full, u, lim in readouts:
        s.readout(name, np.asarray(full, float) * lam, unit=u, limit=lim)
    for at, name, detail, kind in events:
        s.event(at, name, detail, kind=kind)
    s.label(peak_label, mesh.X[keep], kind="peak")
    s.verdict(*verdict)
    centre = 0.5 * (mesh.X.min(0) + mesh.X.max(0))
    s.camera(target=centre, direction=(1.0, -1.3, 0.8), distance=1.9 * span, fov=30)
    return s.save(out)


__all__ = ["ramp_scene"]
