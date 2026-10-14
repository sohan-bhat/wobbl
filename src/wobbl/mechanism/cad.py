"""CAD parts as MuJoCo meshes, for a mechanism assembled from a STEP file or one file per part.

Each solid is meshed on its surface with gmsh in a separate process, the way every CAD import in Wobbl is, and
written as an STL in millimeters, which the MJCF scales to meters. MuJoCo then works out each part's volume,
mass and inertia from its mesh and the part's density, and collides parts through their convex hulls.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CAD = (".step", ".stp", ".iges", ".igs", ".brep")
MESHES = (".stl", ".obj")


def part_mesh(path: str | Path, solid: int | None, cache: Path) -> Path:
    """A part as an STL in millimeters: a mesh file as it is, or one solid (counted from 1) of a CAD file meshed
    on its surface. Converted parts are kept in ``cache`` under a name made from the file and the solid."""
    p = Path(path)
    if p.suffix.lower() in MESHES:
        return p
    if p.suffix.lower() not in CAD:
        raise ValueError(f"'{p.name}' is not a part Wobbl can read (STEP, IGES, BREP, STL or OBJ)")
    key = hashlib.sha1(f"{p.resolve()}:{p.stat().st_mtime_ns}:{solid}".encode()).hexdigest()[:12]
    out = Path(cache) / f"{p.stem}-{solid or 'all'}-{key}.stl"
    if not out.exists():
        from ..mesh.isolate import run_isolated

        out.parent.mkdir(parents=True, exist_ok=True)
        run_isolated(_mesh_solid, str(p), solid, str(out))
    return out


def _mesh_solid(path: str, solid: int | None, out: str) -> None:
    from ..geometry.cad import _set_import_options, gmsh_session

    with gmsh_session() as gmsh:
        _set_import_options(gmsh, False)
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        if not vols:
            raise ValueError(f"{Path(path).name} holds no solid")
        if solid is not None:
            if not 1 <= solid <= len(vols):
                raise ValueError(f"{Path(path).name} holds {len(vols)} solids, so there is no solid {solid}")
            keep = vols[solid - 1]
            gmsh.model.occ.remove([v for v in vols if v != keep], recursive=True)
            gmsh.model.occ.synchronize()
        lo = gmsh.model.getBoundingBox(-1, -1)
        size = max(lo[3] - lo[0], lo[4] - lo[1], lo[5] - lo[2])
        gmsh.option.setNumber("Mesh.MeshSizeMax", size / 25)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(out)
