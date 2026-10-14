"""General 3D scenes for the field solvers: any CAD bodies in a sphere of air, meshed for GetDP.

A scene is one or more CAD files (STEP, IGES, BREP, STL or a parametric .py part) whose solids are kept
as separate bodies, unlike a strength run, which bonds them into one part. Each body later gets a role
from its material. A volume body is meshed inside (steel, a magnet, a dielectric, air in a cavity), a hole
body is cut out and only its surface is kept (a perfect conductor held at a voltage, or a metal antenna),
and the air around everything ends in a spherical shell that maps the outside to infinity, or in an
absorbing sphere for waves.

Lengths in the CAD are millimeters, as everywhere in Wobbl, and the scene is scaled to meters for the
solvers. Elements are sized from each body's thickness, refined in gaps between bodies and grown out
into the air.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAD = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}


class SceneError(ValueError):
    pass


@dataclass
class Body:
    name: str
    index: int                    # 1 based across the whole scene
    file: str
    volume: float                 # m3
    area: float                   # m2
    bbox: tuple                   # x0, y0, z0, x1, y1, z1 in m
    center: tuple                 # center of mass, m
    hole_axis: tuple | None = None   # a ring's axis: the principal direction whose line through the center misses it

    @property
    def thickness(self) -> float:
        """A plate's thickness, a rod's diameter: twice volume over area, capped by the smallest box side."""
        side = min(self.bbox[3] - self.bbox[0], self.bbox[4] - self.bbox[1], self.bbox[5] - self.bbox[2])
        return float(min(2 * self.volume / max(self.area, 1e-30), side))

    @property
    def extent(self) -> float:
        return float(math.dist(self.bbox[:3], self.bbox[3:]))


@dataclass
class Scene:
    bodies: list[Body]
    brep: str                     # all bodies in meters, in index order
    notes: list[str] = field(default_factory=list)

    @property
    def bbox(self) -> tuple:
        b = np.array([x.bbox for x in self.bodies])
        return tuple(b[:, :3].min(axis=0)) + tuple(b[:, 3:].max(axis=0))

    @property
    def center(self) -> np.ndarray:
        b = self.bbox
        return 0.5 * (np.array(b[:3]) + np.array(b[3:]))

    @property
    def radius(self) -> float:
        b = self.bbox
        return 0.5 * math.dist(b[:3], b[3:])

    def select(self, text: str) -> list[Body]:
        """Bodies by name (part of a STEP label or file name), "body 2", "largest", "smallest", "all",
        "at x,y,z" (mm, the smallest body whose box holds the point) or several joined by "and"."""
        out: list[Body] = []
        for part in re.split(r"\s+and\s+|\s*\+\s*", text.strip()):
            out += [b for b in self._one(part) if b not in out]
        return out

    def _one(self, t: str) -> list[Body]:
        s = t.strip().lower()
        if s in ("all", "everything"):
            return list(self.bodies)
        if s == "largest":
            return [max(self.bodies, key=lambda b: b.volume)]
        if s == "smallest":
            return [min(self.bodies, key=lambda b: b.volume)]
        m = re.fullmatch(r"(?:body|solid|#)\s*(\d+)", s)
        if m:
            i = int(m.group(1))
            if not 1 <= i <= len(self.bodies):
                raise SceneError(f"there is no body {i}. The scene has {len(self.bodies)} bodies "
                                 f"({self.listing()})")
            return [self.bodies[i - 1]]
        m = re.fullmatch(r"at\s+([-\d.e+]+)\s*,\s*([-\d.e+]+)\s*,\s*([-\d.e+]+)", s)
        if m:
            p = np.array([float(v) for v in m.groups()]) * 1e-3
            hits = [b for b in self.bodies if np.all(p >= np.array(b.bbox[:3]) - 1e-9)
                    and np.all(p <= np.array(b.bbox[3:]) + 1e-9)]
            if not hits:
                raise SceneError(f"no body holds the point {t[2:].strip()} mm ({self.listing()})")
            return [min(hits, key=lambda b: b.volume)]
        exact = [b for b in self.bodies if b.name.lower() == s]
        hits = exact or [b for b in self.bodies if s in b.name.lower() or s == Path(b.file).stem.lower()]
        if not hits:
            raise SceneError(f"no body called '{t}'. The scene has {self.listing()}")
        return hits

    def listing(self) -> str:
        return ", ".join(f"body {b.index} '{b.name}' ({_mm(b.bbox)})" for b in self.bodies)


def _mm(bb) -> str:
    return " x ".join(f"{1e3 * (bb[i + 3] - bb[i]):.3g}" for i in range(3)) + " mm"


# --- loading ------------------------------------------------------------------------------------
def load(files: list[str] | str, workdir: str | Path) -> Scene:
    """Read every solid of every file as its own body, scaled to meters, and save them together."""
    from ..mesh.isolate import run_isolated

    files = [files] if isinstance(files, (str, Path)) else list(files)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    srcs = []
    for f in files:
        p = Path(f).expanduser()
        if not p.exists():
            raise SceneError(f"file not found: {p}")
        if p.suffix.lower() == ".py":
            from ..geometry.parametric import build_step
            p = Path(build_step(p, None, work / "built")[0])
        elif p.suffix.lower() not in CAD | {".stl"}:
            raise SceneError(f"{p.name} is not a CAD file. Field studies read STEP, IGES, BREP, STL and .py parts")
        srcs.append(str(p.resolve()))
    d = run_isolated(_load, srcs, str(work / "scene.brep"), timeout=300)
    return Scene([Body(**b) for b in d["bodies"]], d["brep"], d["notes"])


def _session():
    import gmsh

    from ..geometry.cad import _deterministic

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False, run=False)
    gmsh.option.setNumber("General.Terminal", 0)
    _deterministic(gmsh)
    return gmsh


def _load(srcs: list[str], out: str) -> dict:
    gmsh = _session()
    try:
        occ = gmsh.model.occ
        bodies, notes, names = [], [], set()
        for src in srcs:
            stem = Path(src).stem
            if src.lower().endswith(".stl"):
                vols = _stl_solid(gmsh, src)
                notes.append(f"{Path(src).name} has no units, so Wobbl reads it in millimeters.")
                labels = [""] * len(vols)
            else:
                gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
                gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
                ents = occ.importShapes(src, highestDimOnly=True)
                occ.synchronize()
                vols = [t for d, t in ents if d == 3]
                if not vols:
                    raise SceneError(f"{Path(src).name} holds no closed solid. Export solids, not surfaces")
                labels = [gmsh.model.getEntityName(3, v) for v in vols]
            for v, lab in zip(vols, labels):
                base = _label(lab) or (stem if len(vols) == 1 else f"{stem} {vols.index(v) + 1}")
                name, k = base, 2
                while name.lower() in names:
                    name, k = f"{base} {k}", k + 1
                names.add(name.lower())
                bodies.append((v, name, Path(src).name))
        tags = [(3, v) for v, _, _ in bodies]
        occ.dilate(tags, 0, 0, 0, 1e-3, 1e-3, 1e-3)
        occ.synchronize()
        meta = []
        for i, (v, name, f) in enumerate(bodies):
            bb = gmsh.model.getBoundingBox(3, v)
            surf = [abs(s) for _, s in gmsh.model.getBoundary([(3, v)], oriented=False)]
            com = occ.getCenterOfMass(3, v)
            meta.append({"name": name, "index": i + 1, "file": f, "volume": occ.getMass(3, v),
                         "area": sum(occ.getMass(2, s) for s in surf), "bbox": tuple(bb),
                         "center": tuple(com), "hole_axis": _hole_axis(gmsh, v, com, bb)})
        _overlaps(gmsh, bodies, meta, notes)
        gmsh.write(out)
        return {"bodies": meta, "brep": out, "notes": notes}
    finally:
        gmsh.finalize()


def _hole_axis(gmsh, v: int, com, bb) -> tuple | None:
    """For a ring (a coil, a washer), the principal axis whose line through the center of mass passes through the
    hole without touching the body. None for a solid body."""
    try:
        m = np.array(gmsh.model.occ.getMatrixOfInertia(3, v), float).reshape(3, 3)
        _, vecs = np.linalg.eigh(0.5 * (m + m.T))
    except Exception:  # noqa: BLE001
        vecs = np.eye(3)
    span = math.dist(bb[:3], bb[3:])
    c = np.array(com, float)
    found = []
    for k in range(3):
        d = vecs[:, k]
        flat = [float(x) for t in np.linspace(-0.6 * span, 0.6 * span, 41) for x in c + t * d]
        try:
            inside = gmsh.model.isInside(3, v, flat)
        except Exception:  # noqa: BLE001
            return None
        if not inside:
            found.append(tuple(float(x) for x in d))
    return found[0] if len(found) == 1 else None


def _label(lab: str) -> str:
    """A body's name from its STEP label, or nothing when the label is a CAD tool's placeholder."""
    name = (lab or "").split("/")[-1].strip()
    if re.match(r"(?i)^open cascade step translator", name) or \
            re.fullmatch(r"(?i)(shape|solid|body|part|component|compound|mesh)[\s_-]*\d*", name):
        return ""
    return name


def moved(scene: Scene, names: list[str], shift, path: str | Path) -> Scene:
    """The same scene with some bodies translated by shift (m), for virtual work checks."""
    from dataclasses import replace

    from ..mesh.isolate import run_isolated

    run_isolated(_move, scene.brep, [b.index for b in scene.bodies if b.name in names], tuple(float(x) for x in shift),
                 str(path), timeout=120)
    dx = np.array(shift, float)
    bodies = [replace(b, bbox=tuple(np.array(b.bbox) + np.concatenate([dx, dx])), center=tuple(np.array(b.center) + dx))
              if b.name in names else b for b in scene.bodies]
    return Scene(bodies, str(path), list(scene.notes))


def _move(brep: str, indices: list[int], shift: tuple, out: str) -> str:
    gmsh = _session()
    try:
        ents = gmsh.model.occ.importShapes(brep, highestDimOnly=True)
        vols = [t for d, t in ents if d == 3]
        gmsh.model.occ.translate([(3, vols[i - 1]) for i in indices], *shift)
        gmsh.model.occ.synchronize()
        gmsh.write(out)
        return out
    finally:
        gmsh.finalize()


def _stl_solid(gmsh, path: str) -> list[int]:
    gmsh.merge(path)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.classifySurfaces(math.radians(30), True, True, math.pi)
    gmsh.model.mesh.createGeometry()
    surfs = [s for _, s in gmsh.model.getEntities(2)]
    loop = gmsh.model.geo.addSurfaceLoop(surfs)
    vol = gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    # move the classified STL into OpenCASCADE so it can be fragmented with the air
    tmp = path + ".wobbl.brep"
    try:
        gmsh.write(tmp)
        out = gmsh.model.occ.importShapes(tmp)
    except Exception:  # noqa: BLE001
        raise SceneError(f"{Path(path).name} could not be turned into a solid. Export STEP instead") from None
    gmsh.model.occ.synchronize()
    _ = vol
    return [t for d, t in out if d == 3]


def _overlaps(gmsh, bodies, meta, notes) -> None:
    """Bodies that overlap in the CAD would share space; the later one takes it, and the run says so."""
    occ = gmsh.model.occ
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = meta[i]["bbox"], meta[j]["bbox"]
            if any(a[k + 3] <= b[k] or b[k + 3] <= a[k] for k in range(3)):
                continue
            try:
                common, _ = occ.intersect([(3, bodies[i][0])], [(3, bodies[j][0])], removeObject=False,
                                          removeTool=False)
                vol = sum(occ.getMass(3, t) for d, t in common if d == 3)
                if common:
                    occ.remove(common, recursive=True)
            except Exception:  # noqa: BLE001
                vol = 0.0
            if vol > 1e-6 * min(meta[i]["volume"], meta[j]["volume"]):
                notes.append(f"'{meta[i]['name']}' and '{meta[j]['name']}' overlap by {vol * 1e9:.3g} mm3 in the CAD, "
                             f"so the overlap counts as '{meta[j]['name']}'.")
    occ.synchronize()


# --- meshing --------------------------------------------------------------------------------------
@dataclass
class Domain:
    """How the space around the bodies is closed."""
    kind: str = "shell"           # shell (mapped to infinity), sphere (an absorbing boundary), none (bodies only)
    radius: float = 0.0           # m, 0 picks 2.5 times the scene's radius
    shell: float = 1.25           # outer radius over inner radius for a shell
    max_size: float = 0.0         # m, cap on the element size in air (waves need a fraction of a wavelength)
    gap_cells: float = 1.5        # elements across the narrowest gap between two bodies


@dataclass
class SceneMesh:
    path: str
    tags: dict                    # region name: physical tag. Volumes below 100, surfaces from 101
    n_nodes: int
    n_elements: int
    radius: float                 # inner air radius, m
    outer: float                  # outer radius (end of the shell), m
    center: tuple
    sizes: dict                   # body name: element size used at its surface, m
    surfaces: dict = field(default_factory=dict)  # hole body name: its surface area, m2
    touching: list = field(default_factory=list)  # pairs of bodies that share a face
    order: int = 1                                # element order of the geometry


def region(name: str) -> str:
    return re.sub(r"\W", "_", name.strip()) or "body"


def mesh(scene: Scene, holes: list[str], volumes: list[str], path: str | Path, k: float = 1.0,
         domain: Domain | None = None, sizes: dict | None = None, timeout: float = 900.0,
         sheets: dict | None = None, order: int = 1) -> SceneMesh:
    """Mesh the scene. holes are bodies cut out (their surfaces become regions), volumes are meshed inside,
    other bodies are left out entirely. sizes overrides the element size at a body's surface (m).
    sheets maps a name to a list of planar polygons (each (N, 3) in m), meshed as internal surfaces."""
    from ..mesh.isolate import run_isolated

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = run_isolated(_mesh, scene, list(holes), list(volumes), str(path), float(k), domain or Domain(), sizes or {},
                     sheets or {}, int(order), timeout=timeout)
    return SceneMesh(**d)


def body_size(b: Body) -> float:
    return max(min(b.thickness / 2.5, b.extent / 10), b.extent / 200)


def _mesh(scene: Scene, holes, volumes, path, k, dom: Domain, sizes, sheets, elem_order=1) -> dict:
    gmsh = _session()
    try:
        occ = gmsh.model.occ
        ents = occ.importShapes(scene.brep, highestDimOnly=True)
        occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        if len(vols) != len(scene.bodies):
            raise SceneError("the saved scene changed between loading and meshing; run again")
        by_name = {b.name: v for b, v in zip(scene.bodies, vols)}
        keep = set(holes) | set(volumes)
        for b, v in zip(scene.bodies, vols):
            if b.name not in keep:
                occ.remove([(3, v)], recursive=True)
        occ.synchronize()
        c = scene.center
        R = dom.radius or 2.5 * scene.radius
        Ro = R * dom.shell if dom.kind == "shell" else R
        objs, names = [], []
        if dom.kind != "none":
            objs.append([(3, occ.addSphere(*c, R))])
            names.append("air")
            if dom.kind == "shell":
                outer = occ.addSphere(*c, Ro)
                inner = occ.addSphere(*c, R)
                shell, _ = occ.cut([(3, outer)], [(3, inner)])
                objs.append(shell)
                names.append("shell")
        order = [n for n in volumes] + [n for n in holes if n not in volumes]
        for n in order:
            objs.append([(3, by_name[n])])
            names.append(n)
        sheet_names = []
        for n, polys in (sheets or {}).items():
            surfs = []
            for poly in polys:
                pts = [occ.addPoint(*p) for p in np.asarray(poly, float)]
                ls = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                surfs.append((2, occ.addPlaneSurface([occ.addCurveLoop(ls)])))
            objs.append(surfs)
            names.append(n)
            sheet_names.append(n)
        flat = [dt for o in objs for dt in o]
        if len(flat) > 1:
            _, mp = occ.fragment(flat, [])
        else:
            mp = [[flat[0]]]
        occ.synchronize()
        owner_v: dict[int, int] = {}
        owner_s: dict[int, int] = {}
        kk = 0
        for i, o in enumerate(objs):
            for _ in o:
                for dim, t in mp[kk]:
                    if dim == 3:
                        owner_v[t] = i
                    elif dim == 2 and names[i] in sheet_names:
                        owner_s[t] = i
                kk += 1
        hole_ids = {i for i, n in enumerate(names) if n in holes and n not in volumes}
        hole_surf: dict[str, set] = {}
        for t, i in list(owner_v.items()):
            if i in hole_ids:
                for _, s in gmsh.model.getBoundary([(3, t)], oriented=False):
                    hole_surf.setdefault(names[i], set()).add(abs(s))
        drop = [(3, t) for t, i in owner_v.items() if i in hole_ids]
        if drop:
            occ.remove(drop)
            occ.synchronize()
            owner_v = {t: i for t, i in owner_v.items() if i not in hole_ids}
        faces_of: dict[str, set] = {}
        for t, i in owner_v.items():
            if names[i] not in ("air", "shell"):
                faces_of.setdefault(names[i], set()).update(
                    abs(s) for _, s in gmsh.model.getBoundary([(3, t)], oriented=False))
        for n, ss in hole_surf.items():
            faces_of.setdefault(n, set()).update(ss)
        bn = sorted(faces_of)
        touching = [(a, b) for i, a in enumerate(bn) for b in bn[i + 1:] if faces_of[a] & faces_of[b]]
        tags, vtag = {}, 1
        for i, n in enumerate(names):
            if i in hole_ids or n in sheet_names:
                continue
            ts = [t for t, j in owner_v.items() if j == i]
            if not ts:
                continue
            gmsh.model.addPhysicalGroup(3, ts, vtag, name=region(n))
            tags[n] = vtag
            vtag += 1
        stag = 101
        areas = {}
        for n, ss in hole_surf.items():
            ss = sorted(ss)
            gmsh.model.addPhysicalGroup(2, ss, stag, name=region(n))
            tags[n] = stag
            areas[n] = sum(occ.getMass(2, s) for s in ss)
            stag += 1
        for n in sheet_names:
            ss = sorted(t for t, i in owner_s.items() if names[i] == n)
            if ss:
                gmsh.model.addPhysicalGroup(2, ss, stag, name=region(n))
                tags[n] = stag
                areas[n] = sum(occ.getMass(2, s) for s in ss)
                stag += 1
        if dom.kind != "none":
            outer_s = []
            for d_, t in gmsh.model.getEntities(2):
                bb = gmsh.model.getBoundingBox(2, t)
                if abs(max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]) - 2 * Ro) < 0.04 * Ro:
                    outer_s.append(t)
            if not outer_s:
                raise SceneError("the outer boundary of the air was not found, which is a Wobbl bug")
            gmsh.model.addPhysicalGroup(2, outer_s, stag, name="outer")
            tags["outer"] = stag
        used = _size_fields(gmsh, scene, by_name, keep, names, owner_v, hole_surf, k, R, dom, sizes, sheet_names,
                            owner_s, dom.gap_cells)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        if elem_order == 2:       # curved elements follow round parts and the spherical shell
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            gmsh.model.mesh.setOrder(2)
        gmsh.write(path)
        n = len(gmsh.model.mesh.getNodes()[0])
        ne = sum(len(e) for e in gmsh.model.mesh.getElements(3)[1])
        return {"path": path, "tags": tags, "n_nodes": n, "n_elements": ne, "radius": R, "outer": Ro,
                "center": tuple(c), "sizes": used, "surfaces": areas, "touching": touching,
                "order": elem_order}
    finally:
        gmsh.finalize()


def _size_fields(gmsh, scene, by_name, keep, names, owner_v, hole_surf, k, R, dom, sizes, sheet_names, owner_s,
                 gap_cells=1.5):
    """Each body's surface gets its own size, refined where another body comes close, growing into the air."""
    occ = gmsh.model.occ
    fl = gmsh.model.mesh.field
    bodies = [b for b in scene.bodies if b.name in keep]
    far = min(dom.max_size or R / 5, R / 5) * k
    used, fields = {}, []
    surf_of: dict[str, list[int]] = {}
    for n in keep:
        i = names.index(n)
        ss = set(hole_surf.get(n, set()))
        for t, j in owner_v.items():
            if j == i:
                ss |= {abs(s) for _, s in gmsh.model.getBoundary([(3, t)], oriented=False)}
        surf_of[n] = sorted(ss)
    for n in sheet_names:
        surf_of[n] = sorted(t for t, i in owner_s.items() if names[i] == n)
    boxes = []
    for b in bodies:
        used[b.name] = (sizes.get(b.name) or body_size(b)) * k
    for i, b in enumerate(bodies):
        for o in bodies[i + 1:]:
            a, c = b.bbox, o.bbox
            gap = max(max(a[j] - c[j + 3], c[j] - a[j + 3]) for j in range(3))
            h = min(used[b.name], used[o.name])
            if not 0 < gap < 3 * h / k:
                continue
            # refine only the box where the two bodies face each other across the gap
            lo = [min(max(a[j], c[j]), min(a[j + 3], c[j + 3])) - gap for j in range(3)]
            hi = [max(max(a[j], c[j]), min(a[j + 3], c[j + 3])) + gap for j in range(3)]
            # a few cells across the gap, but never more than about twenty thousand elements in the box
            vol = float(np.prod([hi[j] - lo[j] for j in range(3)]))
            boxes.append((lo, hi, max(gap / gap_cells, (vol / 20000) ** (1 / 3), min(b.extent, o.extent) / 400) * k))
    _ = occ
    for n, h in sizes.items():
        if n in sheet_names:
            used[n] = h * k
    growth = 0.35
    for n, ss in surf_of.items():
        if not ss or n not in used:
            continue
        f = fl.add("Distance")
        fl.setNumbers(f, "SurfacesList", ss)
        fl.setNumber(f, "Sampling", 20)
        t = fl.add("MathEval")
        fl.setString(t, "F", f"Min({far}, {used[n]} + {growth} * F{f})")
        fields.append(t)
    for lo, hi, h in boxes:
        f = fl.add("Box")
        fl.setNumber(f, "VIn", h)
        fl.setNumber(f, "VOut", far)
        for ax, a0, a1 in zip("XYZ", lo, hi):
            fl.setNumber(f, f"{ax}Min", a0)
            fl.setNumber(f, f"{ax}Max", a1)
        fl.setNumber(f, "Thickness", 3 * max(hi[j] - lo[j] for j in range(3)) * 0.1 + 4 * h)
        fields.append(f)
    if not fields:
        fields = [fl.add("MathEval")]
        fl.setString(fields[0], "F", str(far))
    mn = fl.add("Min")
    fl.setNumbers(mn, "FieldsList", fields)
    fl.setAsBackgroundMesh(mn)
    for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(opt, 0)
    return used
