"""Find where a part is most stressed and decide whether each peak is real.

Linear elastic FEA has two kinds of peak. Real ones sit on fillets, holes and
other smooth features and settle to a fixed value as the mesh is refined.
Artificial ones sit on perfectly sharp inside corners, at the edge of an
infinitely rigid fixture, or where a rigid coupling grabs the part; refining
there makes the number climb without limit. Wobbl tracks every peak through
the refinement levels and only reports the governing stress from peaks that
behave like real ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.faces import Corner, FaceInfo
from ..mesh.mesher import Mesh


@dataclass
class Hotspot:
    id: int
    xyz: list[float]
    kind: str                       # feature, corner, fixture, bolt, coupling, wall
    zone: str | None = None         # fixture or coupling name for zone peaks
    detail: str = ""
    face: int | None = None
    radius: float = 0.0             # tracking radius, mm
    values: list[float] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)   # element size at the peak per level
    singular: bool = False
    converged: bool = False
    change: float | None = None     # relative change over the last refinement

    @property
    def value(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def excluded(self) -> bool:
        return self.kind != "feature"

    def trend(self) -> str:
        return " -> ".join(f"{v:.3g}" for v in self.values if v == v)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["xyz"] = [round(v, 3) for v in self.xyz]
        d["values"] = [round(v, 4) for v in self.values]
        d["sizes"] = [round(v, 4) for v in self.sizes]
        d["excluded"] = self.excluded
        return d


class ZoneMap:
    """Distances from every node to the places where a modeling idealization acts."""

    def __init__(self, mesh: Mesh, zones: dict[str, np.ndarray], zone_kind: dict[str, str],
                 corners: list[Corner]):
        self.mesh = mesh
        self.trees: dict[str, tuple[cKDTree, str]] = {}
        for name, nodes in zones.items():
            if len(nodes):
                self.trees[name] = (cKDTree(mesh.X[nodes]), zone_kind.get(name, "fixture"))
        pts = [c.points for c in corners if c.points is not None and len(c.points)]
        self.corner_tree = cKDTree(np.vstack(pts)) if pts else None
        self.corners = corners

    def nearest(self, xyz: np.ndarray) -> tuple[str | None, str | None, float]:
        best = (None, None, np.inf)
        for name, (tree, kind) in self.trees.items():
            d, _ = tree.query(xyz)
            if d < best[2]:
                best = (name, kind, float(d))
        if self.corner_tree is not None:
            d, _ = self.corner_tree.query(xyz)
            if d < best[2]:
                best = ("corner", "corner", float(d))
        return best

    def excluded_mask(self, X: np.ndarray, radius: float) -> np.ndarray:
        """Nodes within ``radius`` of any zone or sharp corner."""
        m = np.zeros(len(X), bool)
        for tree, _ in self.trees.values():
            d, _ = tree.query(X, distance_upper_bound=radius)
            m |= np.isfinite(d)
        if self.corner_tree is not None:
            d, _ = self.corner_tree.query(X, distance_upper_bound=radius)
            m |= np.isfinite(d)
        return m


def nearest_face(mesh: Mesh, xyz: np.ndarray) -> int | None:
    _, _, c = mesh.tri_geometry()
    i = int(np.argmin(np.linalg.norm(c - xyz, axis=1)))
    s = int(mesh.tri_surf[i])
    return s if s >= 0 else None


def find_hotspots(mesh: Mesh, measure: np.ndarray, zmap: ZoneMap, faces: dict[int, FaceInfo],
                  diag: float, max_spots: int = 5, floor: float = 0.45) -> list[Hotspot]:
    """Cluster the highest stressed nodes into distinct peaks and say what each one is."""
    sizes = mesh.node_sizes()
    top = float(np.nanmax(measure))
    if top <= 0:
        return []
    cand = np.nonzero(measure >= floor * top)[0]
    cand = cand[np.argsort(-measure[cand])]
    spots: list[Hotspot] = []
    taken: list[tuple[np.ndarray, float]] = []
    for i in cand:
        x = mesh.X[i]
        if any(np.linalg.norm(x - p) < r for p, r in taken):
            continue
        h = float(sizes[i]) if np.isfinite(sizes[i]) else diag / 50
        r = max(3.0 * h, 0.02 * diag)
        name, kind, d = zmap.nearest(x)
        hs = Hotspot(len(spots) + 1, x.tolist(), "feature", radius=max(1.5 * h, 0.01 * diag))
        if name is not None and d <= max(1.5 * h, 0.004 * diag):
            hs.kind = kind
            hs.zone = None if kind == "corner" else name
        fid = nearest_face(mesh, x)
        hs.face = fid
        hs.detail = describe_spot(hs, faces.get(fid) if fid is not None else None, zmap)
        spots.append(hs)
        taken.append((x, r))
        if len(spots) >= max_spots:
            break
    return spots


def describe_spot(hs: Hotspot, face: FaceInfo | None, zmap: ZoneMap) -> str:
    what = ""
    if face is not None:
        if face.role == "fillet":
            what = f"the {face.radius:.3g} mm inside fillet (face {face.id})"
        elif face.role == "hole":
            what = f"the {2 * face.radius:.3g} mm hole (face {face.id})"
        elif face.role == "boss":
            what = f"the {2 * face.radius:.3g} mm boss (face {face.id})"
        else:
            what = f"face {face.id}"
    if hs.kind == "corner":
        return f"at a sharp inside corner ({what})" if what else "at a sharp inside corner"
    if hs.kind in ("fixture", "bolt", "wall"):
        return f"at the edge of support {hs.zone} ({what})" if what else f"at the edge of support {hs.zone}"
    if hs.kind == "coupling":
        return f"where the rigid attachment {hs.zone} grabs the part ({what})"
    if hs.kind == "load":
        return f"where {hs.zone} is pushed into the holes ({what})"
    if face is not None and face.role in ("fillet", "hole", "boss"):
        return ("on " if face.role == "fillet" else "at the edge of ") + what
    return f"on {what}" if what else "in the body of the part"


def track(spots: list[Hotspot], mesh: Mesh, measure: np.ndarray):
    """Record each hotspot's peak on a new mesh (max within its tracking radius) and the local element size."""
    tree = cKDTree(mesh.X)
    sizes = mesh.node_sizes()
    for hs in spots:
        idx = tree.query_ball_point(hs.xyz, hs.radius)
        if not idx:
            _, j = tree.query(hs.xyz)
            idx = [j]
        idx = np.asarray(idx)
        hs.values.append(float(measure[idx].max()))
        hs.sizes.append(float(np.nanmin(sizes[idx])))


def classify(spots: list[Hotspot], tol: float):
    """Converged or singular, from the values across refinement levels."""
    for hs in spots:
        v = hs.values
        if len(v) >= 2 and v[-2] > 0:
            hs.change = (v[-1] - v[-2]) / v[-2]
            hs.converged = abs(hs.change) <= tol
        if hs.kind == "corner":
            hs.singular = True
        elif hs.kind != "feature" and len(v) >= 2:
            growing = [v[k] / v[k - 1] > 1.03 for k in range(1, len(v)) if v[k - 1] > 0]
            hs.singular = bool(growing) and all(growing)
