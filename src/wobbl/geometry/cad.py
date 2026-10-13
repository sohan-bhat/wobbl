"""Load CAD into gmsh's OpenCASCADE kernel, heal it, and save a clean copy.

Every later step (face analysis, meshing at each refinement level, geometry
edits) reloads the prepared copy, so surface ids stay identical across meshes.
STEP, IGES and BREP go through OpenCASCADE. STL is triangle soup: it is
classified into patches by feature angle and turned into a volume.
"""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAD_EXT = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}
MESH_EXT = {".stl", ".obj", ".ply"}
SUPPORTED = CAD_EXT | MESH_EXT

_lock = threading.RLock()  # gmsh is a process wide singleton


class GeometryError(RuntimeError):
    """A CAD problem Wobbl could not fix by itself. The message says what to do."""


@dataclass
class PreparedGeometry:
    source: Path
    path: Path                    # prepared .brep (CAD) or the original STL
    kind: str                     # "cad" or "stl"
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    volume: float                 # mm3
    n_bodies: int
    n_surfaces: int
    notes: list[str] = field(default_factory=list)
    stl_angle: float = 30.0

    @property
    def size(self) -> np.ndarray:
        lo, hi = np.array(self.bbox[0]), np.array(self.bbox[1])
        return hi - lo

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.size))

    def to_dict(self) -> dict:
        return {"source": str(self.source), "path": str(self.path), "kind": self.kind,
                "bbox": [list(self.bbox[0]), list(self.bbox[1])], "volume": self.volume,
                "n_bodies": self.n_bodies, "n_surfaces": self.n_surfaces, "notes": self.notes,
                "stl_angle": self.stl_angle}

    @classmethod
    def from_dict(cls, d: dict) -> "PreparedGeometry":
        return cls(Path(d["source"]), Path(d["path"]), d["kind"], (tuple(d["bbox"][0]), tuple(d["bbox"][1])),
                   d["volume"], d["n_bodies"], d["n_surfaces"], list(d.get("notes", [])), d.get("stl_angle", 30.0))


@contextlib.contextmanager
def gmsh_session(verbose: bool = False):
    import gmsh
    with _lock:
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize(readConfigFiles=False, run=False, interruptible=False)
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("General.NumThreads", max(1, (os.cpu_count() or 2)))
        try:
            gmsh.model.add("wobbl")
            yield gmsh
        finally:
            gmsh.finalize()


def _set_import_options(gmsh, heal: bool):
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    gmsh.option.setNumber("Geometry.OCCImportLabels", 0)
    if heal:
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)


def _load_stl(gmsh, path: Path, angle: float):
    gmsh.merge(str(path))
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.classifySurfaces(angle * np.pi / 180.0, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    loop = gmsh.model.geo.addSurfaceLoop(surfs)
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


def load_into_session(gmsh, prep: PreparedGeometry):
    """Load a prepared geometry into an open gmsh session."""
    if prep.kind == "stl":
        _load_stl(gmsh, prep.path, prep.stl_angle)
    else:
        gmsh.model.occ.importShapes(str(prep.path), highestDimOnly=True)
        gmsh.model.occ.synchronize()


def prepare(source: str | os.PathLike, workdir: str | os.PathLike, heal: bool = True,
            defeature: bool = True, stl_angle: float = 30.0, body: int | None = None) -> PreparedGeometry:
    """Import, heal, unify bodies and defeature. Returns a PreparedGeometry saved in workdir.

    Runs in a child process so a pathological file cannot hang or crash the caller.
    """
    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    try:
        return run_isolated(_prepare, source, workdir, heal, defeature, stl_angle, body, timeout=180.0)
    except IsolatedTimeout:
        if heal:
            try:
                prep = run_isolated(_prepare, source, workdir, False, False, stl_angle, body, timeout=180.0)
                prep.notes.append("Healing the CAD timed out, so Wobbl used it as is.")
                return prep
            except (IsolatedTimeout, IsolatedCrash):
                pass
        raise GeometryError(f"reading {Path(source).name} took more than 3 minutes; the file may be corrupt "
                            "or far more detailed than needed. Try exporting a simplified STEP.") from None
    except IsolatedCrash as e:
        raise GeometryError(f"the CAD kernel crashed reading {Path(source).name} ({e}). Re-export the file, "
                            "preferably as STEP AP214 or AP242, and try again.") from None


def _prepare(source, workdir, heal, defeature, stl_angle, body) -> PreparedGeometry:
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise GeometryError(f"file not found: {src}")
    ext = src.suffix.lower()
    if ext not in SUPPORTED:
        raise GeometryError(f"unsupported file type '{ext}'. Wobbl reads STEP, IGES, BREP and STL.")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    with gmsh_session() as gmsh:
        if ext in MESH_EXT:
            if ext != ".stl":
                raise GeometryError("only STL is supported among mesh formats; export your part as STEP or STL")
            try:
                _load_stl(gmsh, src, stl_angle)
            except Exception as e:  # noqa: BLE001
                raise GeometryError(f"could not turn the STL into a solid ({e}). The STL must be watertight; "
                                    "repair it (for example in PrusaSlicer, Meshmixer or Blender 3D Print Toolbox) "
                                    "or export STEP from your CAD tool instead.") from e
            notes.append("STL has no units; Wobbl assumes millimeters.")
            vols = gmsh.model.getEntities(3)
            bb = gmsh.model.getBoundingBox(-1, -1)
            vol = _stl_volume(gmsh)
            prep = PreparedGeometry(src, src, "stl", (tuple(bb[:3]), tuple(bb[3:])), vol, len(vols),
                                    len(gmsh.model.getEntities(2)), notes, stl_angle)
            _check_size(prep)
            return prep
        _set_import_options(gmsh, False)
        try:
            ents = gmsh.model.occ.importShapes(str(src), highestDimOnly=True)
        except Exception as e:  # noqa: BLE001
            raise GeometryError(f"OpenCASCADE could not read {src.name}: {e}") from e
        gmsh.model.occ.synchronize()
        if not _valid_solids(gmsh) and heal:
            # only repair what is broken: the import time fixes can damage good solids
            gmsh.model.remove()
            gmsh.model.add("wobbl")
            _set_import_options(gmsh, True)
            ents = gmsh.model.occ.importShapes(str(src), highestDimOnly=True)
            gmsh.model.occ.synchronize()
            if not gmsh.model.getEntities(3) and gmsh.model.getEntities(2):
                try:
                    loop = gmsh.model.occ.addSurfaceLoop([t for _, t in gmsh.model.getEntities(2)], sewing=True)
                    gmsh.model.occ.addVolume([loop])
                    gmsh.model.occ.synchronize()
                    notes.append("The file held surfaces only; Wobbl sewed them into a solid.")
                except Exception:  # noqa: BLE001
                    pass
            try:
                gmsh.model.occ.healShapes()
                gmsh.model.occ.synchronize()
                notes.append("The CAD had defects (open or invalid solids); Wobbl repaired them.")
            except Exception:  # noqa: BLE001
                notes.append("Shape healing failed; continuing with the original geometry.")
        if not _valid_solids(gmsh):
            raise GeometryError(f"{src.name} has no valid closed solid (found {len(ents)} entities). "
                                "Export a closed solid body from your CAD tool, not surfaces.")
        vols = [t for d, t in gmsh.model.getEntities(3)]
        if body is not None:
            if body < 1 or body > len(vols):
                raise GeometryError(f"--body {body} is out of range; the file has {len(vols)} bodies")
            keep = vols[body - 1]
            gmsh.model.occ.remove([(3, v) for v in vols if v != keep], recursive=True)
            gmsh.model.occ.synchronize()
            notes.append(f"Using body {body} of {len(vols)}.")
            vols = [keep]
        if len(vols) > 1:
            vols = _unify_bodies(gmsh, vols, notes)
        if defeature:
            _defeature(gmsh, notes)
        vols = [t for d, t in gmsh.model.getEntities(3)]
        bb = gmsh.model.getBoundingBox(-1, -1)
        vol = float(sum(gmsh.model.occ.getMass(3, v) for v in vols))
        out = work / "prepared.brep"
        gmsh.write(str(out))
        prep = PreparedGeometry(src, out, "cad", (tuple(bb[:3]), tuple(bb[3:])), vol, len(vols),
                                len(gmsh.model.getEntities(2)), notes, stl_angle)
    _check_size(prep)
    return prep


def _valid_solids(gmsh) -> bool:
    vols = gmsh.model.getEntities(3)
    if not vols:
        return False
    return all(gmsh.model.occ.getMass(3, v) > 0 for _, v in vols)


def _stl_volume(gmsh) -> float:
    # divergence theorem over the surface triangles
    total = 0.0
    for _, s in gmsh.model.getEntities(2):
        types, _, nodes = gmsh.model.mesh.getElements(2, s)
        for t, nn in zip(types, nodes):
            if t != 2:
                continue
            tags = np.asarray(nn).reshape(-1, 3)
            xyz = {}
            ntags, coords, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
            for i, tg in enumerate(ntags):
                xyz[tg] = coords[3 * i:3 * i + 3]
            p = np.array([[xyz[a] for a in tri] for tri in tags])
            total += np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0
    return abs(total)


def _unify_bodies(gmsh, vols, notes):
    """Several solids: touching ones are bonded (shared mesh at the interface), far apart ones are an error."""
    before = len(vols)
    out, _ = gmsh.model.occ.fragment([(3, vols[0])], [(3, v) for v in vols[1:]])
    gmsh.model.occ.synchronize()
    vols = [t for d, t in gmsh.model.getEntities(3)]
    # connectivity check through shared surfaces
    owner = {}
    for v in vols:
        _, down = gmsh.model.getAdjacencies(3, v)
        for s in down:
            owner.setdefault(s, []).append(v)
    parent = {v: v for v in vols}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, vs in owner.items():
        for v in vs[1:]:
            parent[find(v)] = find(vs[0])
    groups = {}
    for v in vols:
        groups.setdefault(find(v), []).append(v)
    if len(groups) > 1:
        sizes = sorted(((sum(gmsh.model.occ.getMass(3, v) for v in g), k) for k, g in groups.items()), reverse=True)
        raise GeometryError(
            f"the file has {len(groups)} separate bodies that do not touch. Wobbl simulates one connected part; "
            f"pick one with --body N (1 to {before}) or export the part you want on its own.")
    notes.append(f"{before} touching bodies were bonded together into one part (glued at their interfaces).")
    return vols


def _defeature(gmsh, notes):
    """Remove features far below anything that matters for strength: tiny faces from text, logos and slivers.

    Concave fillets and holes are never removed, because they control the peak stress.
    """
    surfs = [t for _, t in gmsh.model.getEntities(2)]
    if not surfs:
        return
    areas = np.array([gmsh.model.occ.getMass(2, s) for s in surfs])
    total = areas.sum()
    bb = gmsh.model.getBoundingBox(-1, -1)
    diag = float(np.linalg.norm(np.array(bb[3:]) - np.array(bb[:3])))
    tiny = [s for s, a in zip(surfs, areas) if a < 1e-6 * total or a < (2e-3 * diag) ** 2]
    if not tiny:
        return
    vols = [t for _, t in gmsh.model.getEntities(3)]
    try:
        gmsh.model.occ.defeature(vols, tiny)
        gmsh.model.occ.synchronize()
        notes.append(f"Removed {len(tiny)} tiny faces (below {(2e-3 * diag) ** 2:.3g} mm2) that would only slow "
                     "meshing.")
    except Exception:  # noqa: BLE001
        notes.append(f"Found {len(tiny)} tiny faces but could not remove them; meshing may be slower.")


def _check_size(prep: PreparedGeometry):
    d = prep.diag
    if d < 1.0:
        prep.notes.append(f"The part is only {d:.3g} mm across. If it was modeled in meters or inches, "
                          "pass --scale (for example --scale 1000 for meters).")
    elif d > 20000:
        prep.notes.append(f"The part is {d / 1000:.3g} m across. If the file was in micrometers or the CAD unit "
                          "was wrong, pass --scale.")


def scale_geometry(prep: PreparedGeometry, factor: float, workdir: str | os.PathLike) -> PreparedGeometry:
    """Uniformly scale a prepared geometry (fixes unit mistakes such as an STL exported in inches)."""
    if prep.kind == "stl":
        with gmsh_session() as gmsh:
            gmsh.merge(str(prep.path))
            gmsh.model.mesh.affineTransform([factor, 0, 0, 0, 0, factor, 0, 0, 0, 0, factor, 0])
            out = Path(workdir) / "scaled.stl"
            gmsh.write(str(out))
        p2 = prepare(out, workdir, stl_angle=prep.stl_angle)
    else:
        with gmsh_session() as gmsh:
            load_into_session(gmsh, prep)
            gmsh.model.occ.dilate(gmsh.model.getEntities(3), 0, 0, 0, factor, factor, factor)
            gmsh.model.occ.synchronize()
            out = Path(workdir) / "prepared.brep"
            gmsh.write(str(out))
            bb = gmsh.model.getBoundingBox(-1, -1)
            vol = float(sum(gmsh.model.occ.getMass(3, v) for _, v in gmsh.model.getEntities(3)))
        p2 = PreparedGeometry(prep.source, out, "cad", (tuple(bb[:3]), tuple(bb[3:])), vol, prep.n_bodies,
                              prep.n_surfaces, list(prep.notes))
    p2.source = prep.source
    p2.notes = [n for n in p2.notes if "across" not in n] + [f"Scaled by {factor:g}."]
    return p2
