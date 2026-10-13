"""Turn loads and fixtures into nodal forces and constraints on a specific mesh."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fe import NodeConstraint, NormalConstraint, RigidCoupling, StaticModel
from ..geometry.faces import FaceInfo
from ..geometry.select import Selection
from ..materials import Material
from ..mesh.mesher import Mesh
from .spec import Fixture, Load, SpecError


@dataclass
class Wall:
    name: str
    nodes: np.ndarray
    normal: np.ndarray        # outward normal of the part face (the wall sits on that side)


@dataclass
class BuiltModel:
    model: StaticModel
    zones: dict[str, np.ndarray] = field(default_factory=dict)    # node sets where a modeling idealization acts
    zone_kind: dict[str, str] = field(default_factory=dict)       # name -> fixture, bolt, coupling, wall
    walls: list[Wall] = field(default_factory=list)
    load_patches: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def tri_area_weights(mesh: Mesh, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consistent nodal weights of a uniform unit traction on the masked boundary faces.

    Returns (per node weight array (n,), per face areas). Second order faces put a third of the area on each
    midside node; first order faces a third on each corner.
    """
    _, a, _ = mesh.tri_geometry(mask)
    w = np.zeros(mesh.n_nodes)
    tris = mesh.tris[mask]
    cols = tris[:, 3:6] if tris.shape[1] == 6 else tris[:, :3]
    for k in range(3):
        np.add.at(w, cols[:, k], a / 3.0)
    return w, a


def traction_forces(mesh: Mesh, mask: np.ndarray, traction: np.ndarray) -> np.ndarray:
    """Nodal forces for a traction given per face ((k,3) or (3,)), N."""
    _, a, _ = mesh.tri_geometry(mask)
    tris = mesh.tris[mask]
    cols = tris[:, 3:6] if tris.shape[1] == 6 else tris[:, :3]
    t = np.broadcast_to(np.asarray(traction, float), (len(a), 3))
    F = np.zeros((mesh.n_nodes, 3))
    for k in range(3):
        np.add.at(F, cols[:, k], t * (a / 3.0)[:, None])
    return F


def distributed_remote(mesh: Mesh, mask: np.ndarray, force: np.ndarray, moment_at_point: np.ndarray,
                       point: np.ndarray) -> np.ndarray:
    """Spread a force and moment acting at a point over faces (an RBE3 style distributing coupling).

    The nodal forces reproduce the exact resultant force and moment and do not stiffen the part.
    """
    w, _ = tri_area_weights(mesh, mask)
    idx = np.nonzero(w > 0)[0]
    wi = w[idx]
    W = wi.sum()
    x = mesh.X[idx]
    c = (x * wi[:, None]).sum(0) / W
    r = x - c
    M = np.asarray(moment_at_point, float) + np.cross(np.asarray(point, float) - c, force)
    J = (wi[:, None, None] * (np.einsum("ij,ij->i", r, r)[:, None, None] * np.eye(3)[None]
                              - np.einsum("ij,ik->ijk", r, r))).sum(0)
    try:
        theta = np.linalg.solve(J, M)
    except np.linalg.LinAlgError:
        if np.linalg.norm(M) > 1e-9 * (1 + np.linalg.norm(force)):
            raise SpecError("the faces picked for a remote load or moment are too small or a single line; "
                            "pick a larger face patch") from None
        theta = np.zeros(3)
    F = np.zeros((mesh.n_nodes, 3))
    F[idx] = wi[:, None] * force / W + wi[:, None] * np.cross(theta, r)
    return F


def _hole_groups(sel: Selection, faces: dict[int, FaceInfo]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for fid in sel.faces:
        f = faces[fid]
        key = f.group if f.group is not None else fid
        groups.setdefault(key, []).append(fid)
    return list(groups.values())


def bolt_bearing(mesh: Mesh, sel: Selection, faces: dict[int, FaceInfo], force: np.ndarray,
                 notes: list[str]) -> np.ndarray:
    """A pin or bolt pushing on holes: cosine pressure on the loaded half of each hole, shared equally."""
    groups = _hole_groups(sel, faces)
    F = np.zeros((mesh.n_nodes, 3))
    share = np.asarray(force, float) / len(groups)
    for g in groups:
        f0 = faces[g[0]]
        if f0.axis is None:
            raise SpecError(f"bolt loads need hole faces; face {f0.id} is not cylindrical")
        a = np.asarray(f0.axis)
        cp = np.asarray(f0.axis_point)
        mask = mesh.surface_mask(g) & sel.tri_mask(mesh)
        _, area, cen = mesh.tri_geometry(mask)
        f_ax = np.dot(share, a) * a
        f_r = share - f_ax
        F += traction_forces(mesh, mask, f_ax / area.sum())
        if np.linalg.norm(f_r) < 1e-12:
            continue
        d = f_r / np.linalg.norm(f_r)
        rad = cen - cp
        rad -= np.outer(rad @ a, a)
        rad /= np.maximum(np.linalg.norm(rad, axis=1), 1e-12)[:, None]
        cosv = np.clip(rad @ d, 0, None)
        trac = cosv[:, None] * rad
        raw = (trac * area[:, None]).sum(0)
        scale = np.linalg.norm(f_r) / max(np.dot(raw, d), 1e-300)
        trac *= scale
        resid = f_r - (trac * area[:, None]).sum(0)          # tiny, from faceting; spread it evenly
        trac += resid / area.sum()
        F += traction_forces(mesh, mask, trac)
    if len(groups) > 1:
        notes.append(f"The bolt load is shared equally by {len(groups)} holes.")
    return F


def washer_nodes(mesh: Mesh, sel: Selection, faces: dict[int, FaceInfo]) -> np.ndarray:
    """Hole surface plus the flat ring a washer or bolt head would clamp (outer radius 2 x hole radius)."""
    nodes = [mesh.surface_nodes(sel.faces)]
    nrm, area, cen = mesh.tri_geometry()
    for g in _hole_groups(sel, faces):
        f0 = faces[g[0]]
        if f0.axis is None:
            continue
        a = np.asarray(f0.axis)
        cp = np.asarray(f0.axis_point)
        r = f0.radius
        half = 0.5 * (f0.length or 0.0)
        rel = cen - cp
        ax = rel @ a
        radial = np.linalg.norm(rel - np.outer(ax, a), axis=1)
        flat = np.abs(nrm @ a) > 0.97
        near_end = np.abs(np.abs(ax) - half) < max(0.05 * half, 0.02 * r) + 1e-6
        ring = flat & near_end & (radial <= 2.0 * r) & (radial >= 0.9 * r)
        if ring.any():
            nodes.append(np.unique(mesh.tris[ring].ravel()))
    return np.unique(np.concatenate(nodes))


def build_static(mesh: Mesh, loads: list[tuple[Load, Selection | None]], fixtures: list[tuple[Fixture, Selection]],
                 material: Material, faces: dict[int, FaceInfo]) -> BuiltModel:
    p = material.internal()
    model = StaticModel(mesh, p["E"], p["nu"], p["density"])
    built = BuiltModel(model)
    F = np.zeros((mesh.n_nodes, 3))
    for k, (fx, sel) in enumerate(fixtures):
        name = f"fix{k + 1}"
        mask = sel.tri_mask(mesh)
        if not mask.any():
            raise SpecError(f"fixture '{fx.describe()}' covers no part of the mesh; widen the selection")
        if fx.kind == "fixed":
            nodes = np.unique(mesh.tris[mask].ravel())
            dofs = tuple(" xyz".index(c) for c in getattr(fx, "dofs", "xyz"))
            model.constraints.append(NodeConstraint(name, nodes, dofs))
        elif fx.kind == "bolt":
            nodes = washer_nodes(mesh, sel, faces)
            model.constraints.append(NodeConstraint(name, nodes))
        elif fx.kind in ("roller", "wall"):
            n_, a_, _ = mesh.tri_geometry(mask)
            nrm = (n_ * a_[:, None]).sum(0)
            nrm /= np.linalg.norm(nrm)
            if (n_ @ nrm).min() < 0.995:
                raise SpecError(f"'{fx.describe()}' needs flat faces that share one normal")
            nodes = np.unique(mesh.tris[mask].ravel())
            if fx.kind == "roller":
                model.normal_constraints.append(NormalConstraint(name, nodes, nrm))
            else:
                built.walls.append(Wall(name, nodes, nrm))
        else:
            raise SpecError(f"unknown fixture type {fx.kind}")
        built.zones[name] = model.constraints[-1].nodes if fx.kind in ("fixed", "bolt") else nodes
        built.zone_kind[name] = {"fixed": "fixture"}.get(fx.kind, fx.kind)
    for k, (ld, sel) in enumerate(loads):
        name = f"load{k + 1}"
        if ld.kind == "gravity":
            g = ld.vector()
            model.gravity = g if model.gravity is None else model.gravity + g
            continue
        mask = sel.tri_mask(mesh)
        if not mask.any():
            raise SpecError(f"load '{ld.describe()}' covers no part of the mesh; widen the selection")
        built.load_patches[name] = mask
        if sel.faces and all(faces[f].role == "hole" for f in sel.faces if f in faces):
            # a load pushed into holes (bolts, pins, a flange) peaks where the idealized load meets the hole
            # edge; like a support, that spot is reported separately rather than as the answer
            built.zones[name] = np.unique(mesh.tris[mask].ravel())
            built.zone_kind[name] = "load"
        n_, a_, c_ = mesh.tri_geometry(mask)
        if ld.kind == "force":
            if ld.direction is None:  # along the outward normal of each face (a push into the part if negative)
                tr = -n_ * (ld.magnitude / a_.sum())
                F += traction_forces(mesh, mask, tr)
            else:
                F += traction_forces(mesh, mask, ld.vector() / a_.sum())
        elif ld.kind == "pressure":
            F += traction_forces(mesh, mask, -n_ * ld.magnitude)
        elif ld.kind == "bolt":
            F += bolt_bearing(mesh, sel, faces, ld.vector(), built.notes)
        elif ld.kind in ("remote", "moment"):
            vec = ld.vector()
            force = vec if ld.kind == "remote" else np.zeros(3)
            moment = vec if ld.kind == "moment" else np.zeros(3)
            if ld.point is not None:
                point = np.asarray(ld.point, float)
            else:
                point = (c_ * a_[:, None]).sum(0) / a_.sum()
            if ld.coupling == "rigid":
                nodes = np.unique(mesh.tris[mask].ravel())
                same = next((r for r in model.rigid if np.intersect1d(r.nodes, nodes).size > 0), None)
                if same is not None:
                    # one rigid attachment carrying several loads: move them to its reference point
                    same.nodes = np.union1d(same.nodes, nodes)
                    same.force = same.force + force
                    same.moment = same.moment + moment + np.cross(point - same.ref_point, force)
                    built.zones[same.name] = same.nodes
                    continue
                model.rigid.append(RigidCoupling(name, nodes, point, force, moment))
                built.zones[name] = nodes
                built.zone_kind[name] = "coupling"
            else:
                F += distributed_remote(mesh, mask, force, moment, point)
        else:
            raise SpecError(f"unknown load type {ld.kind}")
    model.forces = F
    _check_held(model, built)
    return built


def _check_held(model: StaticModel, built: BuiltModel):
    """Refuse to solve a part that can drift or spin: say which motion is free."""
    X = model.mesh.X
    origin = X.mean(0)
    rows = []
    for c in model.constraints:
        for d in c.dofs:
            e = np.zeros(3)
            e[d - 1] = 1.0
            rows += _rbm_rows(X[c.nodes] - origin, e)
    for nc in model.normal_constraints:
        rows += _rbm_rows(X[nc.nodes] - origin, np.asarray(nc.normal))
    for w in built.walls:
        rows += _rbm_rows(X[w.nodes] - origin, np.asarray(w.normal))
    if not rows:
        raise SpecError("nothing holds the part. Add a fixture, for example --fix 'holes' or --fix 'x=min'")
    A = np.array(rows)
    A[:, 3:] /= max(float(np.abs(A[:, 3:]).max()), 1e-12)  # balance translation and rotation columns
    _, sv, vt = np.linalg.svd(A, full_matrices=False)
    if len(sv) < 6 or sv.min() < 1e-7 * sv.max():
        names = ["slide along x", "slide along y", "slide along z", "spin about x", "spin about y", "spin about z"]
        motion = names[int(np.argmax(np.abs(vt[-1])))]
        raise SpecError(f"the fixtures leave the part free to {motion}. Add a fixture that stops it, for "
                        "example fix a hole or a face that is not parallel to the existing ones")


def _rbm_rows(pts: np.ndarray, e: np.ndarray) -> list[np.ndarray]:
    """Rows of the constraint matrix acting on the 6 rigid body modes (3 translations, 3 rotations)."""
    if len(pts) > 400:
        pts = pts[np.linspace(0, len(pts) - 1, 400).astype(int)]
    return [np.concatenate([e, np.cross(p, e)]) for p in pts]
