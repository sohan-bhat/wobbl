"""3D grids around and inside CAD parts with snappyHexMesh, and readers for what OpenFOAM writes about them.

A part's surface goes in as STL files in metres, one per patch: the walls, each capped opening of an internal
passage, or each named face of a fluid volume. A box of cubic cells covers the domain, snappyHexMesh refines it
toward the surfaces, removes the cells on the far side of them from a point known to be in the fluid, and snaps
what is left onto the surfaces and their sharp edges. No boundary layers are added: walls use the blended wall
functions, which hold at any y+, and the y+ check reports which regime the grid landed in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import foam


@dataclass
class Surface:
    name: str                     # becomes the patch name
    tris: np.ndarray              # (n, 3, 3) triangle corners, m
    kind: str = "wall"            # wall, inlet, outlet
    level: tuple[int, int] = (1, 1)   # snappyHexMesh refinement levels, min and max by curvature
    flow: float = 0.0
    scalar: float | None = None


@dataclass
class Region:
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    level: int


@dataclass
class Grid:
    cells: int = 0
    faces: int = 0
    patches: dict = field(default_factory=dict)      # name -> faces in the final mesh
    max_non_ortho: float = 0.0
    max_skewness: float = 0.0
    failed_checks: int = 0
    log: str = ""


def write_stl(path: Path, tris: np.ndarray, name: str):
    t = np.asarray(tris, float)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    lines = [f"solid {name}"]
    for nn, (a, b, c) in zip(n, t):
        lines += [f" facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}", "  outer loop",
                  f"   vertex {a[0]:.9e} {a[1]:.9e} {a[2]:.9e}", f"   vertex {b[0]:.9e} {b[1]:.9e} {b[2]:.9e}",
                  f"   vertex {c[0]:.9e} {c[1]:.9e} {c[2]:.9e}", "  endloop", " endfacet"]
    lines.append(f"endsolid {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


BOX_FACES = {"xmin": (0, 4, 7, 3), "xmax": (1, 2, 6, 5), "ymin": (0, 1, 5, 4), "ymax": (3, 7, 6, 2),
             "zmin": (0, 3, 2, 1), "zmax": (4, 5, 6, 7)}


def box(case: foam.Case, lo, hi, size: float, kinds: dict[str, str]) -> tuple[int, int, int]:
    """A background box of near cubic cells of the given size. kinds maps each face name in BOX_FACES to a patch
    kind (faces left out become slip walls). Returns the cell counts."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(1, np.round((hi - lo) / size)).astype(int)
    v = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
         (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    pats = [foam.Patch(name, kinds.get(name, "slip"), [f]) for name, f in BOX_FACES.items()]
    foam.block_mesh(case, v, [f"hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)"], pats)
    return tuple(int(x) for x in n)


def snappy(case: foam.Case, surfaces: list[Surface], location, regions: list[Region] = (), max_cells: int = 2_000_000,
           feature_level: int | None = None, between: int = 3, gap_level: int = 0):
    """Write the surfaces, extract their sharp edges and write snappyHexMeshDict. location is a point inside the
    fluid, in m, that lies on no cell face."""
    tri_dir = case.root / "constant" / "triSurface"
    for s in surfaces:
        write_stl(tri_dir / f"{s.name}.stl", s.tris, s.name)
    feats = "\n".join(f"{s.name}.stl\n{{\n    extractionMethod extractFromSurface;\n"
                      f"    extractFromSurfaceCoeffs {{ includedAngle 150; }}\n"
                      f"    subsetFeatures {{ nonManifoldEdges yes; openEdges yes; }}\n    writeObj no;\n}}\n"
                      for s in surfaces)
    case.write("system/surfaceFeatureExtractDict", "surfaceFeatureExtractDict", feats)
    geom = "".join(f"    {s.name}.stl {{ type triSurfaceMesh; name {s.name}; }}\n" for s in surfaces)
    geom += "".join(f"    {r.name} {{ type searchableBox; min {foam.val(r.lo)}; max {foam.val(r.hi)}; }}\n"
                    for r in regions)
    ftr = "".join(f"        {{ file \"{s.name}.eMesh\"; "
                  f"level {feature_level if feature_level is not None else s.level[1]}; }}\n" for s in surfaces)
    gap = f" gapLevel ({3} {s.level[1]} {s.level[1] + gap_level});" if gap_level else ""
    refs = "".join(f"        {s.name} {{ level ({s.level[0]} {s.level[1]});{gap if s.kind == 'wall' else ''} "
                   f"patchInfo {{ type {'wall' if s.kind == 'wall' else 'patch'}; }} }}\n" for s in surfaces)
    regs = "".join(f"        {r.name} {{ mode inside; levels ((1e15 {r.level})); }}\n" for r in regions)
    body = f"""castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
{geom}}}

castellatedMeshControls
{{
    maxLocalCells       {max_cells};
    maxGlobalCells      {max_cells};
    minRefinementCells  0;
    maxLoadUnbalance    0.1;
    nCellsBetweenLevels {between};
    features
    (
{ftr}    );
    refinementSurfaces
    {{
{refs}    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
{regs}    }}
    locationInMesh      {foam.val(tuple(location))};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          60;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true; layers {{ }} expansionRatio 1.2; finalLayerThickness 0.5; minThickness 0.1; nGrow 0;
    featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-30;
    minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01;
    minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}

mergeTolerance 1e-6;
"""
    case.write("system/snappyHexMeshDict", "snappyHexMeshDict", body)
    names = {p.name for p in case.patches}
    case.patches = list(case.patches) + [foam.Patch(s.name, s.kind, flow=s.flow, scalar=s.scalar)
                                         for s in surfaces if s.name not in names]


def mesh(case: foam.Case, timeout: float = 3600) -> Grid:
    """blockMesh, the sharp edges, snappyHexMesh and checkMesh. Returns the grid's size and quality."""
    case.run("blockMesh")
    case.run("surfaceFeatureExtract")
    log = case.run("snappyHexMesh", "-overwrite", timeout=timeout)
    chk = case.run("checkMesh", check=False)
    g = Grid(log=log[-3000:])
    m = re.search(r"cells:\s+(\d+)", chk)
    g.cells = int(m.group(1)) if m else 0
    m = re.search(r"faces:\s+(\d+)", chk)
    g.faces = int(m.group(1)) if m else 0
    m = re.search(r"Mesh non-orthogonality Max:\s*([\d.eE+-]+)", chk)
    g.max_non_ortho = float(m.group(1)) if m else 0.0
    m = re.search(r"Max skewness = ([\d.eE+-]+)", chk)
    g.max_skewness = float(m.group(1)) if m else 0.0
    m = re.search(r"Failed (\d+) mesh checks", chk)
    g.failed_checks = int(m.group(1)) if m else 0
    g.patches = {name: n for name, (_, n) in patch_ranges(case).items()}
    return g


# --- reading the mesh and boundary values -------------------------------------------------------------------------
def _list(text: str, start: int = 0) -> tuple[int, int]:
    """The count and the index just after the opening bracket of the first list in text after start."""
    m = re.compile(r"\n(\d+)\s*\n?\(").search(text, start)
    return int(m.group(1)), m.end()


def patch_ranges(case: foam.Case) -> dict[str, tuple[int, int]]:
    """Each patch's first face and face count."""
    text = (case.root / "constant" / "polyMesh" / "boundary").read_text()
    out = {}
    for m in re.finditer(r"\n\s*(\w+)\s*\{([^}]*)\}", text):
        n = re.search(r"nFaces\s+(\d+);", m.group(2))
        st = re.search(r"startFace\s+(\d+);", m.group(2))
        if n and st:
            out[m.group(1)] = (int(st.group(1)), int(n.group(1)))
    return out


def read_boundary_faces(case: foam.Case, patches: list[str]) -> dict[str, dict]:
    """Face centres, outward area vectors (m2) and owner cells of the named patches."""
    pm = case.root / "constant" / "polyMesh"
    ranges = {k: v for k, v in patch_ranges(case).items() if k in patches and v[1] > 0}
    if not ranges:
        return {}
    pts = _vectors((pm / "points").read_text())
    owner = _ints((pm / "owner").read_text())
    lo = min(st for st, _ in ranges.values())
    hi = max(st + n for st, n in ranges.values())
    text = (pm / "faces").read_text()
    _, i = _list(text)
    polys = []
    for k, line in enumerate(text[i:].split("\n")):
        if k >= hi + 1:
            break
        line = line.strip()
        if not line or line == ")":
            continue
        polys.append(line)
    polys = [np.array(t[t.index("(") + 1:t.rindex(")")].split(), int) for t in polys[lo:hi]]
    out = {}
    for name, (st, n) in ranges.items():
        cen, area = np.zeros((n, 3)), np.zeros((n, 3))
        for j in range(n):
            p = pts[polys[st - lo + j]]
            c0 = p.mean(0)
            q = np.roll(p, -1, axis=0)
            t = np.cross(p - c0, q - c0) / 2
            mag = np.linalg.norm(t, axis=1)
            area[j] = t.sum(0)
            cen[j] = (mag[:, None] * (c0 + p + q) / 3).sum(0) / max(mag.sum(), 1e-300)
        out[name] = {"centres": cen, "areas": area, "owner": owner[st:st + n]}
    return out


def _vectors(text: str) -> np.ndarray:
    n, i = _list(text)
    return np.array(text[i:text.index("\n)", i)].replace("(", " ").replace(")", " ").split(), float).reshape(n, 3)


def _ints(text: str) -> np.ndarray:
    n, i = _list(text)
    return np.array(text[i:text.index("\n)", i)].split(), int)


def read_patch_values(path: Path, patch: str) -> np.ndarray | None:
    """The values a field holds on one boundary patch, uniform or not, or None when the patch has none."""
    text = path.read_text()
    m = re.search(rf"\n\s*{patch}\s*\n\s*\{{", text)
    if not m:
        return None
    end = text.index("\n    }", m.end())
    block = text[m.end():end]
    v = re.search(r"\bvalue\s+(uniform|nonuniform)\s+", block)
    if not v:
        return None
    if v.group(1) == "uniform":
        raw = block[v.end():block.index(";", v.end())].replace("(", " ").replace(")", " ").split()
        return np.array(raw, float)
    lm = re.search(r"List<(\w+)>\s*(\d+)\s*\(", block[v.end():])
    if not lm:     # an empty list on a patch with no faces
        return np.zeros(0)
    start = v.end() + lm.end()
    stop = block.index("\n)", start) if "\n)" in block[start:] else block.index(")", start)
    vals = np.array(block[start:stop].replace("(", " ").replace(")", " ").split(), float)
    n = int(lm.group(2))
    return vals.reshape(n, -1) if lm.group(1) == "vector" else vals
