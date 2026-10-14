"""Pieces buckling and vibration share: reading the part, the mesh levels, and where a peak stress is trusted."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.faces import find_sharp_corners
from ..materials import resolve_material
from ..mesh.mesher import Sizing, auto_sizing
from ..structural.hotspots import ZoneMap
from ..structural.run import describe_part, exclusion_radius, resolve_selections
from ..structural.spec import StructuralStudy

# levels of global refinement, each about 2.4 times the nodes of the one before, up to a node cap
QUALITY = {"fast": dict(levels=2, nodes=25000, tol=0.03), "normal": dict(levels=3, nodes=60000, tol=0.02),
           "fine": dict(levels=4, nodes=150000, tol=0.01)}
REFINE = 0.75


@dataclass
class Prepared:
    prep: object
    features: object
    loads: list
    fixtures: list
    selections: dict
    material: object


def prepare(base: StructuralStudy, work: Path, progress=None, pre=None) -> Prepared:
    """Read the CAD and resolve the supports and loads the way the strength study does. A vibration study may have
    no loads at all, only supports."""
    mat = resolve_material(base.material)
    if pre is None:
        if progress:
            progress("geometry", f"reading {Path(base.part).name}, finding faces, holes and fillets")
        pre = describe_part(base.part, work, base.scale, base.body, base.params)
    prep, features = pre
    if base.loads:
        loads, fixtures, info = resolve_selections(base, features)
    else:
        loads, fixtures, info = [], *resolve_fixtures(base, features)
    return Prepared(prep, features, loads, fixtures, info, mat)


def resolve_fixtures(base: StructuralStudy, features) -> tuple[list, dict]:
    from ..geometry.select import SelectionError, Selector
    from ..structural.spec import SpecError

    sel = Selector(features.faces, features.bbox)
    fixtures, info = [], {}
    for k, fx in enumerate(base.fixtures):
        try:
            s = sel.select(fx.faces)
        except SelectionError as e:
            raise SpecError(f"fixture {k + 1} ({fx.describe()}): {e}") from None
        fixtures.append((fx, s))
        info[f"fix{k + 1}"] = s.to_dict()
    if not fixtures:
        raise SpecError("no fixtures: say what holds the part, for example --fix 'holes d=4.5' or --fix 'x=min'")
    return fixtures, info


def level_sizes(p: Prepared, quality: str, mesh_size: float | None = None) -> list[Sizing]:
    """Global element sizes for each level. The first level is sized so the last one stays under the node cap,
    but never with elements much thicker than the walls."""
    q = QUALITY.get(quality, QUALITY["normal"])
    first = max(8000, int(q["nodes"] / (1 / REFINE ** 3) ** (q["levels"] - 1)))
    s = auto_sizing(p.prep, thickness=p.features.thickness_typical, max_nodes=first)
    h = float(mesh_size) if mesh_size else min(s.h, 1.5 * p.features.thickness_typical)
    return [Sizing(h=h * REFINE ** k) for k in range(q["levels"])]


def excluded_nodes(mesh, built, features, sizing: Sizing, measure: np.ndarray) -> tuple[np.ndarray, float]:
    """Nodes near supports, rigid attachments, loads on holes and sharp inside corners, where linear stress depends on
    the idealization rather than the part, and how far that reaches."""
    corners = find_sharp_corners(mesh, faces=features.faces, tangent=features.tangent_pairs)
    zmap = ZoneMap(mesh, built.zones, built.zone_kind, corners)
    hot = mesh.X[measure > 0.5 * measure.max()] if measure.max() > 0 else mesh.X[:1]
    r = exclusion_radius(features, sizing, mesh, hot)
    return zmap.excluded_mask(mesh.X, r), r


def governing(measure: np.ndarray, excluded: np.ndarray) -> int:
    valid = ~excluded if (~excluded).any() else np.ones(len(measure), bool)
    return int(np.nonzero(valid)[0][np.argmax(measure[valid])])


def change(values: list[float]) -> float:
    return abs(values[-1] - values[-2]) / max(abs(values[-1]), 1e-300) if len(values) > 1 else float("inf")
