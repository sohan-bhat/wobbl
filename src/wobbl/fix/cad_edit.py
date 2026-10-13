"""Edit prepared CAD with OpenCASCADE (through gmsh): bigger fillets and corner gussets.

Each edit runs in a child process like every other CAD operation, writes a new
BREP and a STEP the user can open in their CAD tool, and reports what it did.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry.cad import PreparedGeometry, gmsh_session
from ..mesh.isolate import run_isolated


class EditError(RuntimeError):
    pass


def _finish(gmsh, prep: PreparedGeometry, out_dir: Path, name: str) -> PreparedGeometry:
    out_dir.mkdir(parents=True, exist_ok=True)
    brep = out_dir / f"{name}.brep"
    gmsh.write(str(brep))
    gmsh.write(str(out_dir / f"{name}.step"))
    vols = gmsh.model.getEntities(3)
    bb = gmsh.model.getBoundingBox(-1, -1)
    vol = float(sum(gmsh.model.occ.getMass(3, v) for _, v in vols))
    return PreparedGeometry(prep.source, brep, "cad", (tuple(bb[:3]), tuple(bb[3:])), vol, len(vols),
                            len(gmsh.model.getEntities(2)), list(prep.notes))


def _concave_curves_near(gmsh, point: np.ndarray, reach: float) -> list[int]:
    """CAD edges close to a point where two faces meet at an inside angle."""
    from ..mesh.mesher import _surface_from_session

    sm = _surface_from_session(gmsh, reach)
    from ..geometry.faces import find_sharp_corners

    out = []
    for c in find_sharp_corners(sm, min_angle_deg=20):
        d = np.linalg.norm(c.points - point, axis=1).min()
        if d <= reach:
            out += c.curves
    return sorted(set(out))


def _fillet(prep: PreparedGeometry, out_dir: str, point, radius: float, fillet_face: int | None,
            reach: float) -> PreparedGeometry:
    point = np.asarray(point, float)
    with gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(prep.path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [v for _, v in gmsh.model.getEntities(3)]
        if fillet_face is not None:
            try:
                gmsh.model.occ.defeature(vols, [fillet_face])
                gmsh.model.occ.synchronize()
            except Exception as e:  # noqa: BLE001
                raise EditError(f"could not remove the old fillet: {e}") from e
            vols = [v for _, v in gmsh.model.getEntities(3)]
        curves = _concave_curves_near(gmsh, point, reach)
        if not curves:
            raise EditError("no inside corner edge was found next to the peak")
        try:
            gmsh.model.occ.fillet(vols, curves, [radius], removeVolume=True)
            gmsh.model.occ.synchronize()
        except Exception as e:  # noqa: BLE001
            raise EditError(f"OpenCASCADE could not make a {radius:g} mm fillet there ({e}); the walls may be "
                            "too short for that radius") from e
        return _finish(gmsh, prep, Path(out_dir), f"fillet_r{radius:g}")


def fillet(prep: PreparedGeometry, out_dir: str | Path, point, radius: float, fillet_face: int | None = None,
           reach: float = 3.0) -> PreparedGeometry:
    return run_isolated(_fillet, prep, str(out_dir), list(map(float, point)), float(radius), fillet_face,
                        float(reach), timeout=180)


def _gusset(prep: PreparedGeometry, out_dir: str, corner, n_a, n_b, edge_dir, edge_mid, leg: float,
            thickness: float) -> PreparedGeometry:
    corner = np.asarray(corner, float)
    n_a, n_b, e = (np.asarray(v, float) for v in (n_a, n_b, edge_dir))
    e /= np.linalg.norm(e)
    mid = np.asarray(edge_mid, float)
    base = corner + ((mid - corner) @ e) * e - e * thickness / 2.0
    with gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(prep.path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        vols = [(3, v) for _, v in gmsh.model.getEntities(3)]
        # sink the gusset slightly into both walls so the union is solid
        sink = 0.05 * thickness
        p0 = base - n_a * sink - n_b * sink
        p1 = base + n_b * leg - n_a * sink
        p2 = base + n_a * leg - n_b * sink
        pts = [gmsh.model.occ.addPoint(*p) for p in (p0, p1, p2)]
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % 3]) for i in range(3)]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        ext = gmsh.model.occ.extrude([(2, surf)], *(e * thickness))
        wedge = [x for x in ext if x[0] == 3]
        try:
            gmsh.model.occ.fuse(vols, wedge)
            gmsh.model.occ.synchronize()
        except Exception as ex:  # noqa: BLE001
            raise EditError(f"could not merge the gusset into the part: {ex}") from ex
        if len(gmsh.model.getEntities(3)) != 1:
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
        return _finish(gmsh, prep, Path(out_dir), f"gusset_t{thickness:g}")


def gusset(prep: PreparedGeometry, out_dir: str | Path, corner, n_a, n_b, edge_dir, edge_mid, leg: float,
           thickness: float) -> PreparedGeometry:
    args = [list(map(float, v)) for v in (corner, n_a, n_b, edge_dir, edge_mid)]
    return run_isolated(_gusset, prep, str(out_dir), *args, float(leg), float(thickness), timeout=180)
