"""Turn a study's loads and supports, picked on the solid's faces, into a shell model.

Loads are spread over the solid's faces by the same code the solid study uses (even forces, pressures, bolt
bearing, distributed remote loads), then carried onto the midsurface (see mapping.py), so a load means the same
thing on either model. Supports hold the shell nodes on the picked faces. A fixed face also stops the nodes
turning, as a solid face held over the whole wall does, and a roller holds only the direction off its face.
"""
from __future__ import annotations

import numpy as np

from ..fe import RigidCoupling
from ..structural.build import bolt_bearing, distributed_remote, traction_forces, washer_nodes
from ..structural.spec import SpecError
from .ccx import Layer, ShellModel
from .mapping import SolidSurface, nodes_on, transfer_forces
from .mesh import ShellMesh


def build_shell(shell: ShellMesh, loads, fixtures, faces, E: float = 0.0, nu: float = 0.0, density: float = 0.0,
                layers: list[Layer] | None = None, ref_dir=None, posts: list | None = None
                ) -> tuple[ShellModel, dict[str, np.ndarray]]:
    """The shell model and the node sets where each support, rigid attachment or post joint acts (its idealized
    zone)."""
    model = ShellModel(shell, E, nu, density, list(layers or []), ref_dir, posts=list(posts or []), post_h=shell.h)
    post_X = model.place_posts()
    solid = SolidSurface(shell.solid)
    zones: dict[str, np.ndarray] = {}
    for k, (fx, sel) in enumerate(fixtures):
        name = f"fix{k + 1}"
        mask = sel.tri_mask(solid)
        if fx.kind == "bolt":
            ring = np.zeros(solid.n_nodes, bool)
            ring[washer_nodes(solid, sel, faces)] = True
            mask = ring[solid.tris].all(axis=1)
        if not mask.any():
            raise SpecError(f"fixture '{fx.describe()}' covers no part of the surface; widen the selection")
        nodes = np.union1d(nodes_on(shell, solid, mask), _post_nodes_on(model, post_X, solid, mask))
        if not len(nodes):
            raise SpecError(f"fixture '{fx.describe()}' does not touch the shell model's midsurface. Supports on "
                            "rims, holes and wall faces carry over; supports on bosses and other chunky features do "
                            "not, so solve this part without --shell")
        if fx.kind in ("fixed", "bolt"):
            dofs = (1, 2, 3, 4, 5, 6) if getattr(fx, "dofs", "xyz") == "xyz" else \
                tuple(" xyz".index(c) for c in fx.dofs)
        elif fx.kind == "roller":
            n_, a_, _ = solid.tri_geometry(mask)
            nrm = (n_ * a_[:, None]).sum(0)
            nrm /= np.linalg.norm(nrm)
            ax = int(np.argmax(np.abs(nrm)))
            if abs(nrm[ax]) < 0.999:
                raise SpecError(f"'{fx.describe()}': the shell model holds rollers only on faces square to x, y or "
                                "z; solve this part without --shell")
            dofs = (ax + 1,)
        else:
            raise SpecError(f"'{fx.describe()}': wall supports that can lift off are solved on the solid; solve "
                            "this part without --shell")
        model.constraints.append((name, nodes, dofs))
        zones[name] = nodes
    F_solid = np.zeros((solid.n_nodes, 3))
    for k, (ld, sel) in enumerate(loads):
        name = f"load{k + 1}"
        if ld.kind == "gravity":
            g = ld.vector()
            model.gravity = g if model.gravity is None else model.gravity + g
            continue
        mask = sel.tri_mask(solid)
        if not mask.any():
            raise SpecError(f"load '{ld.describe()}' covers no part of the surface; widen the selection")
        n_, a_, c_ = solid.tri_geometry(mask)
        if ld.kind == "force":
            tr = -n_ * (ld.magnitude / a_.sum()) if ld.direction is None else ld.vector() / a_.sum()
            F_solid += traction_forces(solid, mask, tr)
        elif ld.kind == "pressure":
            F_solid += traction_forces(solid, mask, -n_ * ld.magnitude)
        elif ld.kind == "bolt":
            F_solid += bolt_bearing(solid, sel, faces, ld.vector(), [])
        elif ld.kind in ("remote", "moment"):
            vec = ld.vector()
            force = vec if ld.kind == "remote" else np.zeros(3)
            moment = vec if ld.kind == "moment" else np.zeros(3)
            point = np.asarray(ld.point, float) if ld.point is not None else (c_ * a_[:, None]).sum(0) / a_.sum()
            if ld.coupling == "rigid":
                nodes = nodes_on(shell, solid, mask)
                if len(nodes) < 3:
                    raise SpecError(f"load '{ld.describe()}' does not touch the shell model's midsurface; pick wall "
                                    "faces, or solve this part without --shell")
                model.rigid.append(RigidCoupling(name, nodes, point, force, moment))
                zones[name] = nodes
            else:
                F_solid += distributed_remote(solid, mask, force, moment, point)
        else:
            raise SpecError(f"unknown load type {ld.kind}")
    model.forces = transfer_forces(shell, solid.X, F_solid)
    for k, p in enumerate(model.posts):
        zones[f"post{k + 1}"] = p.footprint
    if not model.constraints:
        raise SpecError("nothing holds the part. Add a fixture, for example --fix 'x=min'")
    if model.rigid:
        held = np.unique(np.concatenate([n for _, n, _ in model.constraints]))
        for r in model.rigid:
            r.nodes = np.setdiff1d(r.nodes, held)
            if len(r.nodes) < 3:
                raise SpecError(f"the faces of {r.name} are all held by supports; a rigid attachment needs free faces")
    return model, zones


def _post_nodes_on(model: ShellModel, post_X: np.ndarray, solid: SolidSurface, mask: np.ndarray) -> np.ndarray:
    """Beam nodes of the posts that lie within a post's radius of the picked faces (the end of a boss standing
    on a held face)."""
    from .mapping import point_triangle_distance

    out = []
    n = model.mesh.n_nodes
    idx = np.flatnonzero(mask)
    T = solid.tris[idx]
    A, B, C = (solid.X[T[:, k]] for k in range(3))
    for p in model.posts:
        for i in p.nodes:
            x = post_X[i - n]
            d, _, _ = point_triangle_distance(np.repeat(x[None], len(idx), 0), A, B, C)
            if d.min() <= p.r_out:
                out.append(int(i))
    return np.asarray(out, dtype=np.int64)


def excluded_nodes(shell: ShellMesh, zones: dict[str, np.ndarray], radius: float) -> np.ndarray:
    """Nodes within radius of any support or rigid attachment, where the idealization sets the stress."""
    from scipy.spatial import cKDTree

    out = np.zeros(shell.n_nodes, bool)
    pts = [shell.X[n[n < shell.n_nodes]] for n in zones.values() if (n < shell.n_nodes).any()]
    if not pts:
        return out
    d, _ = cKDTree(np.vstack(pts)).query(shell.X)
    return d <= radius


__all__ = ["build_shell", "excluded_nodes"]
