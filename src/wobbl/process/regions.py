"""Meshes of several bodies that share their faces: a casting, its risers, chills and the sand around them.

The part comes from the prepared CAD. Risers (cylinders) and chills (boxes) are
added where the user puts them, cut back to the part's surface so they sit
on it, and a block of mold material is built around everything. One OpenCASCADE
fragment makes every touching pair share its faces, so heat flows between
bodies through common nodes, and every tetrahedron keeps the body it came from.
Meshing runs in a child process like every other gmsh call in Wobbl.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry.cad import GeometryError, PreparedGeometry


@dataclass
class Riser:
    diameter: float                    # mm
    height: float                      # mm
    at: tuple[float, float, float]     # center of its foot, on the part
    axis: str = "z"                    # it rises along +axis

    @property
    def volume(self) -> float:
        return np.pi * self.diameter ** 2 / 4 * self.height

    @property
    def modulus(self) -> float:
        """Volume over cooling area of a blind cylindrical riser whose foot sits on the casting, mm."""
        d, h = self.diameter, self.height
        return (np.pi * d * d / 4 * h) / (np.pi * d * h + np.pi * d * d / 4)

    def text(self) -> str:
        return f"riser d={self.diameter:g} h={self.height:g} at {','.join(f'{v:g}' for v in self.at)}"


@dataclass
class Chill:
    material: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def text(self) -> str:
        return (f"{self.material} chill {'x'.join(f'{v:g}' for v in self.size)} at "
                f"{','.join(f'{v:g}' for v in self.center)}")


_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_riser(text: str) -> Riser:
    """'d=40 h=60 at 50,0,30' (mm), optionally 'axis=x'. The point is the center of the riser's foot."""
    t = text.lower().replace("riser", " ")
    d = re.search(rf"\bd\s*=\s*({_NUM})", t)
    h = re.search(rf"\bh\s*=\s*({_NUM})", t)
    at = re.search(rf"\bat\s+({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})", t)
    ax = re.search(r"\baxis\s*=\s*([xyz])", t)
    if not (d and h and at):
        raise ValueError(f"could not read the riser '{text}'; write it like 'd=40 h=60 at 50,0,30' (mm, the point "
                         "is the center of the riser's foot on the casting)")
    return Riser(float(d.group(1)), float(h.group(1)), tuple(float(v) for v in at.groups()),
                 ax.group(1) if ax else "z")


def parse_chill(text: str) -> Chill:
    """'steel 30x30x15 at 50,0,-10' (mm): a block of chill material centered at the point, cut back to the part."""
    t = text.lower().replace("chill", " ").strip()
    m = re.match(rf"^([a-z\-]+)?\s*({_NUM})\s*x\s*({_NUM})\s*x\s*({_NUM})\s+at\s+({_NUM})\s*,\s*({_NUM})\s*,"
                 rf"\s*({_NUM})\s*$", t)
    if not m:
        raise ValueError(f"could not read the chill '{text}'; write it like 'steel 30x30x15 at 50,0,-10' (a block "
                         "centered at the point, in mm, overlapping the casting face it should cool)")
    g = m.groups()
    return Chill(g[0] or "steel", tuple(float(v) for v in g[4:7]), tuple(float(v) for v in g[1:4]))


@dataclass
class RegionMesh:
    X: np.ndarray                      # (n,3) mm
    tets: np.ndarray                   # (m,4)
    region: np.ndarray                 # (m,) index into names
    names: list[str]                   # casting, riser 1.., chill 1.., mold
    kinds: list[str]                   # casting, riser, chill, mold
    outer: np.ndarray                  # (k,3) triangles on the outside of the mold
    box: tuple[np.ndarray, np.ndarray]
    t_mesh: float = 0.0
    notes: list[str] = field(default_factory=list)

    def volumes(self) -> np.ndarray:
        c = self.X[self.tets]
        v = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6
        return np.array([v[self.region == r].sum() for r in range(len(self.names))])

    def interfaces(self) -> dict[tuple[int, int], float]:
        """Area shared by each pair of bodies, mm2, from the faces of the tetrahedra."""
        faces = np.concatenate([self.tets[:, [0, 1, 2]], self.tets[:, [0, 1, 3]], self.tets[:, [0, 2, 3]],
                                self.tets[:, [1, 2, 3]]])
        owner = np.tile(self.region, 4)
        key = np.sort(faces, axis=1)
        order = np.lexsort(key.T[::-1])
        key, owner, faces = key[order], owner[order], faces[order]
        same = (key[1:] == key[:-1]).all(axis=1)
        i = np.nonzero(same & (owner[1:] != owner[:-1]))[0]
        out: dict[tuple[int, int], float] = {}
        if len(i):
            p = self.X[faces[i]]
            a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
            for r1, r2, ai in zip(owner[i], owner[i + 1], a):
                k = (int(min(r1, r2)), int(max(r1, r2)))
                out[k] = out.get(k, 0.0) + float(ai)
        return out


def split_interfaces(X: np.ndarray, tets: np.ndarray, metal: np.ndarray):
    """Give the metal its own copy of every node it shares with the mold, so no node holds the heat of both.

    A node on the casting's surface would otherwise lump together a share of metal at the pouring temperature and
    a share of cold sand, and start somewhere in between, frozen before the metal has given up any heat. With the
    nodes split, the two sides meet through a contact conductance across each shared face. Returns the new nodes,
    the new tetrahedra and the face pairs (metal side, mold side) with matching corner order.
    metal: (m,) bool per tetrahedron."""
    n = len(X)
    on_metal = np.zeros(n, bool)
    on_metal[tets[metal].ravel()] = True
    on_other = np.zeros(n, bool)
    on_other[tets[~metal].ravel()] = True
    shared = np.nonzero(on_metal & on_other)[0]
    dup = np.arange(n)
    dup[shared] = n + np.arange(len(shared))
    X2 = np.vstack([X, X[shared]])
    tets2 = tets.copy()
    tets2[metal] = dup[tets[metal]]
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    owner = np.tile(metal, 4)
    key = np.sort(faces, axis=1)
    order = np.lexsort(key.T[::-1])
    key, owner, faces = key[order], owner[order], faces[order]
    i = np.nonzero((key[1:] == key[:-1]).all(axis=1) & (owner[1:] != owner[:-1]))[0]
    other = np.where(owner[i][:, None], faces[i + 1], faces[i])
    return X2, tets2, (dup[other], other)


def build(prep: PreparedGeometry, risers: list[Riser], chills: list[Chill], margin, h_metal: float,
          h_mold: float, band: float = 0.0) -> RegionMesh:
    """margin is the mold thickness around everything, mm, one value or one per axis (0 leaves those sides of the
    block flush with the part and insulated, as in a slice of a long casting). The mold keeps the metal's element
    size for band mm from the metal, the depth heat reaches while it freezes, and coarsens beyond."""
    from ..mesh.isolate import IsolatedCrash, IsolatedTimeout, run_isolated

    if prep.kind == "stl":
        raise GeometryError("a casting study needs a solid CAD part (STEP, IGES, BREP or .py) to build the mold "
                            "around; an STL mesh cannot be combined with risers and sand")
    try:
        return run_isolated(_build, prep, risers, chills, margin, h_metal, h_mold, band, timeout=600)
    except (IsolatedCrash, IsolatedTimeout) as e:
        raise GeometryError(f"could not mesh the casting with its mold ({e})") from e


def _build(prep, risers, chills, margin, h_metal, h_mold, band=0.0) -> RegionMesh:
    from ..geometry.cad import gmsh_session, load_into_session

    t0 = time.time()
    with gmsh_session() as gmsh:
        occ = gmsh.model.occ
        load_into_session(gmsh, prep)
        cast = [(3, t) for _, t in gmsh.model.getEntities(3)]
        if len(cast) > 1:
            cast, _ = occ.fuse(cast[:1], cast[1:])
        added = []
        for r in risers:
            d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[r.axis]
            cyl = occ.addCylinder(*r.at, *(r.height * np.asarray(d)), r.diameter / 2)
            added.append(("riser", [(3, cyl)]))
        for c in chills:
            lo = np.asarray(c.center) - np.asarray(c.size) / 2
            box = occ.addBox(*lo, *c.size)
            added.append(("chill", [(3, box)]))
        occ.synchronize()
        # cut each added body back to the part so it sits on the surface instead of inside it
        bodies = []
        for kind, ent in added:
            cut, _ = occ.cut(ent, cast, removeObject=True, removeTool=False)
            if not cut:
                raise GeometryError(f"a {kind} lies entirely inside the casting; move it onto the surface")
            bodies.append((kind, cut))
        occ.synchronize()
        allb = cast + [e for _, ents in bodies for e in ents]
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for dim, tag in allb:
            b = occ.getBoundingBox(dim, tag)
            lo, hi = np.minimum(lo, b[:3]), np.maximum(hi, b[3:])
        lo, hi = lo - np.asarray(margin, float), hi + np.asarray(margin, float)
        mold = occ.addBox(*lo, *(hi - lo))
        out, out_map = occ.fragment([(3, mold)], allb)
        occ.synchronize()
        # out_map[0] is what the mold box became; whatever of it is also a part or an added body is that body
        names, kinds, tag_region = [], [], {}
        groups = [("casting", "casting", [e[1] for e in cast])]
        n_r = n_c = 0
        for kind, ents in bodies:
            if kind == "riser":
                n_r += 1
                groups.append((f"riser {n_r}", "riser", [e[1] for e in ents]))
            else:
                n_c += 1
                groups.append((f"chill {n_c}", "chill", [e[1] for e in ents]))
        inputs = [e[1] for e in allb]
        for name, kind, tags in groups:
            ridx = len(names)
            names.append(name)
            kinds.append(kind)
            for t in tags:
                k = inputs.index(t)
                for e in out_map[1 + k]:
                    tag_region[e[1]] = ridx
        names.append("mold")
        kinds.append("mold")
        for _, t in gmsh.model.getEntities(3):
            tag_region.setdefault(t, len(names) - 1)
        # sizes: fine on the metal, coarse at the mold's outer corners, graded in between
        opt = gmsh.option.setNumber
        opt("Mesh.MeshSizeMax", h_mold)
        opt("Mesh.MeshSizeFromCurvature", 12)
        opt("Mesh.MeshSizeExtendFromBoundary", 1)
        opt("Mesh.MeshSizeFromPoints", 1)
        opt("Mesh.Algorithm", 6)
        opt("Mesh.Algorithm3D", 10)
        opt("Mesh.Optimize", 1)
        opt("Mesh.ElementOrder", 1)
        metal_tags = [t for t, r in tag_region.items() if kinds[r] in ("casting", "riser")]
        pts = set()
        for t in metal_tags:
            for e in gmsh.model.getBoundary([(3, t)], combined=False, recursive=True):
                if e[0] == 0:
                    pts.add(e)
        gmsh.model.mesh.setSize(list(pts), h_metal)
        field_ = gmsh.model.mesh.field
        dist = field_.add("Distance")
        surfs = []
        for t in metal_tags:
            surfs += [e[1] for e in gmsh.model.getBoundary([(3, t)], combined=False, oriented=False)]
        field_.setNumbers(dist, "SurfacesList", sorted(set(surfs)))
        thr = field_.add("Threshold")
        field_.setNumber(thr, "InField", dist)
        field_.setNumber(thr, "SizeMin", h_metal)
        field_.setNumber(thr, "SizeMax", h_mold)
        field_.setNumber(thr, "DistMin", band)
        field_.setNumber(thr, "DistMax", band + max(float(np.max(margin)) * 0.8, 3 * h_mold))
        inside = field_.add("Constant")
        field_.setNumber(inside, "VIn", h_metal)
        field_.setNumber(inside, "VOut", 1e22)
        field_.setNumbers(inside, "VolumesList", metal_tags)
        mn = field_.add("Min")
        field_.setNumbers(mn, "FieldsList", [thr, inside])
        field_.setAsBackgroundMesh(mn)
        opt("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(3)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        X_all = np.asarray(coords).reshape(-1, 3)
        pos = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        pos[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
        tets, region = [], []
        for _, t in gmsh.model.getEntities(3):
            types, _, enodes = gmsh.model.mesh.getElements(3, t)
            for ty, nn in zip(types, enodes):
                if ty == 4:
                    raw = pos[np.asarray(nn, dtype=np.int64).reshape(-1, 4)]
                    tets.append(raw)
                    region.append(np.full(len(raw), tag_region[t]))
        if not tets:
            raise GeometryError("the casting and mold produced no tetrahedra")
        tets = np.concatenate(tets)
        region = np.concatenate(region)
    used = np.unique(tets)
    lut = np.full(len(X_all), -1, dtype=np.int64)
    lut[used] = np.arange(len(used))
    X = X_all[used]
    tets = lut[tets]
    # outward faces of the whole block: faces used by one tetrahedron only
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    outer = faces[cnt[inv.ravel()] == 1]
    rm = RegionMesh(X, tets, region, names, kinds, outer, (lo, hi), time.time() - t0)
    vols = rm.volumes()
    for r, (nm, kd) in enumerate(zip(names, kinds)):
        if vols[r] <= 0:
            raise GeometryError(f"the {nm} came out empty after combining it with the casting")
    return rm
