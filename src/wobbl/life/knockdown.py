"""Shape classes, imperfection knockdowns and the hand checks of a buckling answer.

Linear buckling assumes a perfect part, so the factor CalculiX finds is cut by a knockdown that depends on how
imperfection sensitive the shape is, from published test data:

    column     the European column curves (EN 1993-1-1 curve b for steel, EN 1999-1-1 class A for aluminum),
               which fold in crookedness, eccentricity and yielding together
    cylinder   NASA SP-8007: 1 - 0.901 (1 - exp(-sqrt(R/t)/16)) in axial compression, 1 - 0.731 (...) in
               bending, 0.75 under external pressure
    dome       0.2, near the low end of spherical cap tests summarized in NASA SP-8032
    plate      1.0: flat plates are not imperfection sensitive and carry load after they buckle
    general    0.5, a judgment between plates and shells, and the result says so

A stocky part yields before it buckles, so the knocked down factor is also capped by yield through the
Johnson-Ostenfeld parabola, which the column curve already contains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..units import fmt
from ..verify.checks import Check

# imperfection factor alpha and plateau slenderness of the column curves
COLUMN_CURVES = {"steel": (0.34, 0.2, "EN 1993-1-1 buckling curve b"),
                 "aluminum": (0.20, 0.1, "EN 1999-1-1 class A alloys"),
                 "other": (0.34, 0.2, "EN 1993-1-1 buckling curve b, as no curve exists for this material")}


@dataclass
class Column:
    axis: np.ndarray
    length: float
    area: float
    i_min: float
    load: float            # compressive force along the axis, N
    k: float | None        # effective length factor for the end conditions, when they are a textbook case
    ends: str = ""

    @property
    def slenderness(self) -> float:
        return self.length / math.sqrt(self.i_min / self.area)


# ------------------------------------------------------------------------------------------ shape classes
def column_info(mesh, built, prep) -> Column | None:
    """The part as a prismatic column, when the loads push along its longest direction from one end toward the
    other and the supports sit at that other end."""
    X = mesh.X
    F, _ = built.model.applied_total()
    if np.linalg.norm(F) <= 0:
        return None
    lo, hi = X.min(0), X.max(0)
    ax = int(np.argmax(hi - lo))
    d = F / np.linalg.norm(F)
    if abs(d[ax]) < 0.95:
        return None
    axis = np.eye(3)[ax]
    s = X[:, ax]
    L = float(hi[ax] - lo[ax])
    tol = 0.03 * L
    load_nodes = np.nonzero(np.abs(built.model.nodal_forces()).sum(1) > 0)[0]
    load_nodes = np.concatenate([load_nodes] + [r.nodes for r in built.model.rigid]) if built.model.rigid \
        else load_nodes
    if not len(load_nodes):
        return None
    held = {}
    for c in built.model.constraints:
        for i in c.nodes:
            held.setdefault(int(i), set()).update(c.dofs)
    if not held:
        return None
    hn = np.array(list(held))
    end_lo, end_hi = s <= lo[ax] + tol, s >= hi[ax] - tol
    at_lo = lambda nodes: bool(end_lo[nodes].all())   # noqa: E731
    at_hi = lambda nodes: bool(end_hi[nodes].all())   # noqa: E731
    if at_hi(load_nodes):
        loaded, base_end = end_hi, end_lo
    elif at_lo(load_nodes):
        loaded, base_end = end_lo, end_hi
    else:
        return None
    # supports at the far end, and perhaps sideways at the loaded end, but none along the length
    if not (end_lo[hn] | end_hi[hn]).all() or not base_end[hn].any():
        return None
    push = float(F @ axis) * (1 if loaded is end_hi else -1)
    if push >= 0:            # the load pulls the column: it cannot buckle as a column
        return None
    area, i_min = _section(mesh, axis, 0.5 * (lo[ax] + hi[ax]))
    if area <= 0 or not 0.75 <= area * L / mesh.volume() <= 1.33:
        return None
    lateral = {1, 2, 3} - {ax + 1}
    # the end faces themselves, for how they are held
    flat = 1e-6 * L
    base_face = np.abs(s - (lo[ax] if base_end is end_lo else hi[ax])) <= flat
    top_face = np.abs(s - (hi[ax] if loaded is end_hi else lo[ax])) <= flat
    face_held = [held.get(int(i), set()) for i in np.nonzero(base_face)[0]]
    share = np.mean([({1, 2, 3} <= h) for h in face_held])
    lat_share = np.mean([(lateral <= h) for h in face_held])
    top_lat = np.mean([(lateral <= held.get(int(i), set())) for i in np.nonzero(top_face)[0]])
    k, ends = None, ""
    if share > 0.9 and top_lat < 0.05:
        k, ends = 2.0, "clamped at the base and free at the loaded end"
    elif share > 0.9 and top_lat > 0.9:
        k, ends = 0.699, "clamped at the base and pinned at the loaded end"
    elif lat_share > 0.9 and top_lat > 0.9:
        k, ends = 1.0, "pinned at both ends"
    return Column(axis, L, area, i_min, abs(push), k, ends)


def _section(mesh, axis: np.ndarray, at: float) -> tuple[float, float]:
    import warnings

    import pyvista as pv

    from ..verify.beam import section_properties

    corners = mesh.tets[:, :4]
    cells = np.hstack([np.full((len(corners), 1), 4), corners]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(corners), pv.CellType.TETRA, dtype=np.uint8), mesh.X)
    origin = mesh.X.mean(0)
    origin[int(np.argmax(np.abs(axis)))] = at
    sl = grid.slice(normal=axis, origin=origin)
    if sl.n_points == 0:
        return 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tri = sl.triangulate().extract_surface().triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    props = section_properties(np.asarray(tri.points)[faces], axis)
    if props is None:
        return 0.0, 0.0
    A, _, I = props
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    B = np.array([u, np.cross(axis, u)])
    return float(A), float(np.linalg.eigvalsh(B @ I @ B.T)[0])


def classify(asked: str, col: Column | None, features, built, loads) -> tuple[str, str]:
    """The shape class for the knockdown, with the reason."""
    if asked != "auto":
        return asked, "chosen with --shape"
    cyl = _cylinder(features)
    total = sum(f.area for f in features.faces.values())
    sphere = sum(f.area for f in features.faces.values() if f.kind == "sphere")
    if cyl is not None:
        R, t, _ = cyl
        return "cylinder", f"a thin cylindrical shell, radius {fmt(R)} mm and wall {fmt(t)} mm (R/t {R / t:.0f})"
    if sphere > 0.5 * total:
        return "dome", "most of its surface is spherical"
    if col is not None and col.slenderness >= 15:
        return "column", (f"a straight member {fmt(col.length)} mm long pushed along its length, slenderness L/r "
                          f"{col.slenderness:.0f}")
    lo, hi = np.asarray(features.bbox[0]), np.asarray(features.bbox[1])
    dims = np.sort(hi - lo)
    biggest = max(features.faces.values(), key=lambda f: f.area)
    F, _ = built.model.applied_total()
    if (dims[0] <= 0.1 * dims[1] and biggest.kind == "plane" and biggest.normal is not None
            and np.linalg.norm(F) > 0 and abs(np.dot(F / np.linalg.norm(F), biggest.normal)) < 0.3):
        return "plate", f"a flat plate {fmt(dims[0])} mm thick loaded in its own plane"
    return "general", "not a column, plate, cylinder or dome"


def _cylinder(features) -> tuple[float, float, np.ndarray] | None:
    """A thin walled tube: coaxial outer and inner cylinders covering most of the surface."""
    faces = list(features.faces.values())
    total = sum(f.area for f in faces)
    cyl = [f for f in faces if f.kind == "cylinder" and f.radius and f.axis is not None]
    if sum(f.area for f in cyl) < 0.6 * total:
        return None
    outer = [f for f in cyl if not f.concave]
    inner = [f for f in cyl if f.concave]
    if not outer or not inner:
        return None
    o = max(outer, key=lambda f: f.area)
    i = max(inner, key=lambda f: f.area)
    if abs(abs(np.dot(o.axis, i.axis)) - 1) > 1e-3:
        return None
    t = o.radius - i.radius
    R = 0.5 * (o.radius + i.radius)
    if t <= 0 or R / t < 10:
        return None
    return R, t, np.asarray(o.axis)


def knockdown(shape: str, features, built, loads) -> tuple[float, str]:
    if shape == "cylinder":
        cyl = _cylinder(features)
        if cyl is None:
            return 0.2, ("a cylinder whose walls Wobbl could not measure, so it takes 0.2, the low end of NASA "
                         "SP-8007 for thin cylinders")
        R, t, axis = cyl
        phi = math.sqrt(R / t) / 16.0
        if any(ld.kind == "pressure" for ld, _ in loads):
            return 0.75, "NASA SP-8007 for cylinders under external pressure"
        F, _ = built.model.applied_total()
        along = np.linalg.norm(F) > 0 and abs(np.dot(F / np.linalg.norm(F), axis)) > 0.9
        if along:
            g = 1 - 0.901 * (1 - math.exp(-phi))
            return g, f"NASA SP-8007 for cylinders in axial compression, 1 - 0.901 (1 - exp(-sqrt(R/t) / 16))"
        g = 1 - 0.731 * (1 - math.exp(-phi))
        return g, "NASA SP-8007 for cylinders in bending, 1 - 0.731 (1 - exp(-sqrt(R/t) / 16))"
    if shape == "dome":
        return 0.2, ("spherical caps under external pressure buckle far below the classical load in tests, and 0.2 is "
                     "near the low end of the data in NASA SP-8032")
    if shape == "plate":
        return 1.0, ("flat plates are not imperfection sensitive: they keep carrying load after they buckle, though "
                     "their deflection grows from the buckling load on")
    if shape == "column":
        return 1.0, "column curve"
    return 0.5, ("no published knockdown fits this shape, so Wobbl takes 0.5, between a plate (about 1) and an axially "
                 "compressed shell (down to 0.2)")


def column_curve(lam: float, col: Column, mat) -> tuple[float, float, str, str]:
    """Design factor of a column on the European buckling curves: chi times the load factor that squashes it.

    lambda_bar = sqrt(N_squash / N_elastic), Phi = 0.5 (1 + alpha (lambda_bar - lambda_0) + lambda_bar^2) and
    chi = 1 / (Phi + sqrt(Phi^2 - lambda_bar^2)), at most one. Returns the design factor, the knockdown on the
    elastic factor, the source and the interaction it covers.
    """
    fam = "steel" if (mat.category == "metal" and mat.E > 150000) else \
        "aluminum" if (mat.category == "metal" and 60000 <= mat.E <= 80000) else "other"
    alpha, lam0, src = COLUMN_CURVES[fam]
    squash = mat.allowable * col.area / col.load
    lb = math.sqrt(squash / lam)
    phi = 0.5 * (1 + alpha * (lb - lam0) + lb * lb)
    chi = min(1.0, 1.0 / (phi + math.sqrt(max(phi * phi - lb * lb, 0.0))))
    design = chi * squash
    why = (f"{src} (imperfection factor {alpha:g}): relative slenderness {lb:.2f}, reduction {chi:.3f} on the "
           f"squash load of {fmt(mat.allowable * col.area)} N")
    return design, design / lam, why, "crookedness, eccentricity and yielding, through the column curve"


def johnson(lam_kd: float, lam_y: float, ductile: bool) -> tuple[float, str]:
    """Johnson-Ostenfeld: a knocked down elastic factor above half the yield factor is cut to
    lam_y (1 - lam_y / (4 lam_kd)). A brittle part simply breaks at its strength."""
    if not math.isfinite(lam_y):
        return lam_kd, ""
    if not ductile:
        return (min(lam_kd, lam_y), "the material breaking first" if lam_y < lam_kd else "")
    if lam_kd > 0.5 * lam_y:
        return lam_y * (1 - lam_y / (4 * lam_kd)), "yielding, through the Johnson-Ostenfeld parabola"
    return lam_kd, ""


def euler_check(lam: float, col: Column | None, mat) -> Check:
    if col is None or col.k is None:
        why = ("the part is not a straight column pushed from one end" if col is None else
               "its end supports are not a textbook case")
        return Check("hand calculation", "skip", f"no Euler column applies ({why})")
    pe = math.pi ** 2 * mat.E * col.i_min / (col.k * col.length) ** 2
    expect = pe / col.load
    err = lam / expect - 1
    s = (f"Euler's column, {col.ends} (K = {col.k:g}), with I = {fmt(col.i_min)} mm4 over {fmt(col.length)} mm gives "
         f"{fmt(pe)} N, a factor of {fmt(expect)}; the finite element factor is {fmt(lam)} ({100 * err:+.1f}%)")
    detail = {"euler_N": pe, "euler_factor": expect, "slenderness": col.slenderness, "k": col.k}
    if abs(err) <= 0.05:
        return Check("hand calculation", "pass", s, detail)
    if abs(err) <= 0.15:
        return Check("hand calculation", "info", s + ". Short or stocky columns sit a little off Euler, from shear and "
                                                     "the way the ends are held", detail)
    return Check("hand calculation", "warn", s + ", far from Euler, so check the supports and the loads", detail)


def separation_check(factors: list[float]) -> Check:
    """How far the next distinct buckling mode sits above the first. A repeated factor is the same buckle in two
    directions of a symmetric section, which is not an interaction; distinct modes within 10% can interact."""
    pos = sorted(f for f in factors if f > 0)
    if len(pos) < 2:
        return Check("mode separation", "skip", "only one buckling mode was found")
    twin = pos[1] <= 1.005 * pos[0]
    nxt = next((f for f in pos[1:] if f > 1.005 * pos[0]), None)
    s = "the first factor repeats, the same buckle in two directions of a symmetric section; " if twin else ""
    if nxt is None:
        return Check("mode separation", "info", s + "no distinct mode was found among the ones solved",
                     {"factors": factors})
    sep = nxt / pos[0]
    s += f"the next distinct mode is at {fmt(nxt)}, {sep:.2f} times the first"
    if sep >= 1.1:
        return Check("mode separation", "pass", s, {"factors": factors})
    return Check("mode separation", "warn", s + ". Distinct modes this close can interact, which makes a real part "
                                                "more sensitive to imperfections than either mode alone",
                 {"factors": factors})
