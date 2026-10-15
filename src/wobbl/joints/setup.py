"""Setting up an assembly for a contact solve: faces picked body by body, prescribed moves, materials per body,
springs for bodies held only by contact, and penalty contact pairs with their gaps dealt with."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..geometry.select import Selection, SelectionError, Selector, parse_vector
from ..materials import Material, resolve_material
from ..solvers import calculix_nl as nl
from ..structural.build import _rbm_rows, tri_area_weights
from ..structural.spec import SpecError
from ..units import fmt, parse_quantity
from .assembly import Assembly, FacePair

PENALTY = 20.0       # penalty slope K = PENALTY x E / (element size at the contact)
STICK = 0.01         # stick slope as a fraction of K: shear builds over a hundredth of the overlap scale
SPRING = 1e-6        # stabilizing springs, together, as a fraction of E x body size in each direction
TOUCH = 1e-3         # gaps and overlaps under this fraction of the assembly are CAD tolerance
TOUCH_MM = 0.004     # and never more than this, so a press fit of a hundredth of a millimetre is still an interference

_BODY = re.compile(r"^\s*(?:body|part)\s+(\d+)\b\s*(?:and\b|:)?\s*(.*)$", re.I)


def select(asm: Assembly, text: str) -> Selection:
    """A face selector, optionally restricted to one body: 'body 2 and top' is the top of body 2."""
    m = _BODY.match(text)
    if not m:
        return Selector(asm.features.faces, asm.features.bbox).select(text)
    b = int(m.group(1))
    if b not in asm.body_faces:
        raise SelectionError(f"there is no body {b}; the assembly has bodies 1 to {asm.n_bodies}")
    sub = {i: asm.features.faces[i] for i in asm.body_faces[b] if i in asm.features.faces}
    lo = np.min([f.bbox[0] for f in sub.values()], axis=0)
    hi = np.max([f.bbox[1] for f in sub.values()], axis=0)
    s = Selector(sub, (lo.tolist(), hi.tolist())).select(m.group(2).strip() or "all")
    s.text = text
    return s


def resolve(asm: Assembly, loads, fixtures) -> tuple[list, list, dict]:
    """Faces for each load and support, as the strength study resolves them, with body prefixes understood."""
    out_l, out_f, info = [], [], {}
    for k, ld in enumerate(loads):
        s = None
        if ld.kind != "gravity":
            try:
                s = select(asm, ld.faces)
            except SelectionError as e:
                raise SpecError(f"load {k + 1} ({ld.describe()}): {e}") from None
            info[f"load{k + 1}"] = s.to_dict()
        out_l.append((ld, s))
    for k, fx in enumerate(fixtures):
        try:
            s = select(asm, fx.faces)
        except SelectionError as e:
            raise SpecError(f"fixture {k + 1} ({fx.describe()}): {e}") from None
        out_f.append((fx, s))
        info[f"fix{k + 1}"] = s.to_dict()
    if not out_f:
        raise SpecError("no fixtures: say what holds the assembly, for example --fix 'body 1 and bottom'")
    return out_l, out_f, info


@dataclass
class Move:
    """A prescribed displacement of some faces, applied after the loads: '0.05 mm +x @ body 2 and x=min'."""
    vector: np.ndarray       # mm
    faces: str
    text: str


def parse_move(text: str) -> Move:
    if "@" not in text:
        raise SpecError(f"move '{text}' needs faces: add '@ <selector>', for example '@ body 2 and x=min'")
    head, faces = (x.strip() for x in text.rsplit("@", 1))
    m = re.match(r"^\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*[a-z\"]*)\s+(.+)$", head, re.I)
    if not m:
        raise SpecError(f"could not read a distance and a direction from move '{text}', as in '0.05 mm +x'")
    dist = parse_quantity(m.group(1), "length").value
    try:
        d = parse_vector(m.group(2))
    except SelectionError as e:
        raise SpecError(f"move '{text}': {e}") from None
    return Move(dist * d, faces, text)


def materials(mesh, default: str, per_body: dict[int, str]) -> tuple[list[nl.Material], dict[int, Material]]:
    """CalculiX materials for each body, and the library material behind every body."""
    bodies = sorted(set(int(b) for b in np.unique(mesh.body)))
    mat_of = {b: resolve_material(per_body.get(b, default)) for b in bodies}
    names = {}
    for b in bodies:
        names.setdefault(mat_of[b].key + mat_of[b].name, []).append(b)
    out = []
    for group in names.values():
        m = mat_of[group[0]]
        p = m.internal()
        out.append(nl.Material(m.name, p["E"], p["nu"], p["density"],
                               elements=np.isin(mesh.body, group) if len(names) > 1 else None))
    return out, mat_of


def floating_bodies(mesh, model) -> list[int]:
    """Bodies that their supports alone cannot hold still: they rest on others through contact only."""
    out = []
    origin = mesh.X.mean(0)
    for b in np.unique(mesh.body):
        nodes = np.unique(mesh.tets[mesh.body == b].ravel())
        rows = []
        for c in model.constraints:
            held = np.intersect1d(c.nodes, nodes)
            for d in c.dofs:
                e = np.zeros(3)
                e[d - 1] = 1.0
                rows += _rbm_rows(mesh.X[held] - origin, e) if len(held) else []
        for nc in model.normal_constraints:
            held = np.intersect1d(nc.nodes, nodes)
            rows += _rbm_rows(mesh.X[held] - origin, np.asarray(nc.normal)) if len(held) else []
        if len(rows) < 6 or np.linalg.matrix_rank(np.array(rows), tol=1e-6 * np.abs(rows).max()) < 6:
            out.append(int(b))
    return out


def springs_for(mesh, bodies: list[int], mat_of: dict[int, Material]) -> list[tuple[np.ndarray, float]]:
    """Springs to ground on up to 60 nodes of each floating body, together a millionth of the body's own stiffness,
    so it cannot drift away before its contact closes. Their share of the load is checked afterwards."""
    out = []
    for b in bodies:
        corner = np.unique(mesh.tets[mesh.body == b][:, :4].ravel())
        pick = corner[np.linspace(0, len(corner) - 1, min(60, len(corner))).astype(int)]
        size = float(np.linalg.norm(np.ptp(mesh.X[corner], axis=0)))
        out.append((pick, SPRING * mat_of[b].E * size / len(pick)))
    return out


def contact_pairs(asm: Assembly, mesh, pairs: list[FacePair], mat_of: dict[int, Material], mu: float,
                  notes: list[str]) -> list[nl.ContactPair]:
    """Penalty contact for each pair of faces. The penalty slope is twenty times the softer modulus over the element
    size at the contact, stiff enough that faces overlap by a small fraction of an element and soft enough to
    converge. Gaps and overlaps within CAD tolerance are closed to exact contact."""
    tol = min(TOUCH * asm.prep.diag, TOUCH_MM)
    out = []
    for k, p in enumerate(pairs):
        slave, master = mesh.surface_mask([p.slave]), mesh.surface_mask([p.master])
        if not slave.any() or not master.any():
            continue
        _, a = tri_area_weights(mesh, slave)
        h = float(np.median(np.sqrt(2.0 * a)))
        E = min(mat_of[p.body_slave].E, mat_of[p.body_master].E)
        K = PENALTY * E / h
        clearance = None
        if abs(p.gap) <= tol and (p.gap_max - p.gap_min) <= 2 * tol:
            clearance = 0.0
            if abs(p.gap) > 1e-9 or abs(p.gap_max - p.gap_min) > 1e-9:
                notes.append(f"The {fmt(abs(p.gap))} mm {'gap' if p.gap > 0 else 'overlap'} the CAD leaves between "
                             f"{p.describe()} is within tolerance, so they start in exact contact.")
        elif p.gap > 0:
            notes.append(f"{p.describe().capitalize()} start {fmt(p.gap)} mm apart, a gap the load has to close "
                         "before they press.")
        else:
            notes.append(f"{p.describe().capitalize()} overlap by {fmt(-p.gap)} mm in the CAD, which is solved as "
                         "an interference fit pressed apart in the first increments.")
        out.append(nl.ContactPair(f"C{k + 1}", slave, master, K, mu, STICK * K if mu > 0 else 0.0, 0.0, clearance))
    return out
