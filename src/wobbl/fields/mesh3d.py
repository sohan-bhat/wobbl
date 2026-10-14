"""3D air volumes for acoustic modes: a union of boxes, or a STEP file of the air inside an enclosure."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Model3D:
    boxes: list[tuple[float, float, float, float, float, float]] = field(default_factory=list)   # m
    step: str = ""                  # a STEP file of the air volume, read in mm
    size: float = 0.5               # element size, m

    def scaled(self, k: float) -> "Model3D":
        from dataclasses import replace

        return replace(self, size=self.size * k)


@dataclass
class Mesh3D:
    path: str
    n_nodes: int
    volume: float                   # m3
    area: float                     # m2 of boundary
    bbox: tuple


def mesh(model: Model3D, path: str | Path, timeout: float = 300.0) -> Mesh3D:
    from ..mesh.isolate import run_isolated

    return Mesh3D(**run_isolated(_build, model, str(path), timeout=timeout))


def _build(model: Model3D, path: str) -> dict:
    import gmsh

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        occ = gmsh.model.occ
        if model.step:
            gmsh.option.setNumber("Geometry.OCCScaling", 1e-3)
            vols = occ.importShapes(model.step)
            vols = [v for v in vols if v[0] == 3]
        else:
            vols = [(3, occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)) for x0, y0, z0, x1, y1, z1 in model.boxes]
        if len(vols) > 1:
            vols, _ = occ.fuse(vols[:1], vols[1:])
        occ.synchronize()
        tags = [t for _, t in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, tags, 1)
        vol = sum(occ.getMass(3, t) for t in tags)
        area = sum(occ.getMass(2, t) for _, t in gmsh.model.getEntities(2))
        gmsh.option.setNumber("Mesh.MeshSizeMax", model.size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", model.size * 0.3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
        n = len(gmsh.model.mesh.getNodes()[0])
        bb = gmsh.model.getBoundingBox(-1, -1)
        return {"path": path, "n_nodes": n, "volume": vol, "area": area, "bbox": tuple(bb)}
    finally:
        gmsh.finalize()
