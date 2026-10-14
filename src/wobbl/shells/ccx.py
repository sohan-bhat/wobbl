"""CalculiX for shell models: S6 elements with a plain or a laminated section, solved and read back per node.

CalculiX expands every shell element into a 15 node wedge through the thickness and writes its results on
those expanded nodes. Each one is matched back to the shell node it grew from by where it sits, half a
thickness or less along that node's normal (or along the normal of one of its elements, where CalculiX
splits a fold), so every shell node gets its midsurface displacement and the strain and stress through its
thickness. A laminate's ply strains come from that strain profile, which stays continuous through the plies
while the stresses jump from ply to ply.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..fe import RigidCoupling
from ..solvers.base import SolverError
from .mesh import ShellMesh


@dataclass
class Layer:
    """One ply of a laminate: CalculiX engineering constants (E1 E2 E3 nu12 nu13 nu23 G12 G13 G23, MPa)."""
    name: str
    constants: tuple[float, ...]
    density: float                 # t/mm3
    thickness: float               # mm
    angle: float                   # degrees from the reference direction, toward the element's second axis


@dataclass
class ShellModel:
    mesh: ShellMesh
    E: float = 0.0                 # MPa, plain section
    nu: float = 0.0
    density: float = 0.0           # t/mm3
    layers: list[Layer] = field(default_factory=list)   # a laminate when not empty, listed from the inner face out
    ref_dir: np.ndarray | None = None                   # the 0 degree direction, laid onto each element
    constraints: list[tuple[str, np.ndarray, tuple[int, ...]]] = field(default_factory=list)  # dofs 1-3, 4-6
    forces: np.ndarray | None = None                    # (n,3) N at the shell nodes
    gravity: np.ndarray | None = None                   # mm/s2
    rigid: list[RigidCoupling] = field(default_factory=list)
    posts: list = field(default_factory=list)           # bosses as beams (posts.Post), plain sections only
    post_h: float = 0.0                                  # beam element length along the posts

    @property
    def n_post_nodes(self) -> int:
        return len(self.place_posts())

    def place_posts(self) -> np.ndarray:
        """Number the posts' new nodes after the shell's and return their coordinates (worked out once).

        Each post is a line of beam nodes from its free end to its joined end, then one midside node for each
        spoke of a stiff spider that joins that end to the shell nodes inside the post's outline. CalculiX will
        not put a beam's node in a rigid body, so the spider stands in for the solid top of a boss, which is far
        stiffer than the wall it meets. A post joined at a single node would punch the wall like a point load.
        """
        if getattr(self, "_post_X", None) is not None:
            return self._post_X
        n, out = self.mesh.n_nodes, []
        for p in self.posts:
            pts = p.points(self.post_h or self.mesh.h)
            mids = 0.5 * (pts[-1][None] + self.mesh.X[p.footprint])
            p.nodes = np.arange(n, n + len(pts))
            p.spokes = np.arange(n + len(pts), n + len(pts) + len(mids))
            n += len(pts) + len(mids)
            out += [pts, mids]
        self._post_X = np.vstack(out) if out else np.zeros((0, 3))
        return self._post_X

    @property
    def laminate(self) -> bool:
        return bool(self.layers)

    def section_thickness(self) -> np.ndarray:
        """Thickness at each node: the laminate's total, or the wall's own."""
        if self.layers:
            return np.full(self.mesh.n_nodes, sum(L.thickness for L in self.layers))
        return self.mesh.thickness

    def mass(self) -> float:
        _, a, _ = self.mesh.element_geometry()
        if self.layers:
            return float(a.sum() * sum(L.density * L.thickness for L in self.layers))
        return float((a * self.mesh.element_thickness()).sum() * self.density)

    def applied_total(self) -> tuple[np.ndarray, np.ndarray]:
        """Total applied force (N) and moment about the origin (N mm), with gravity and rigid loads."""
        X = self.mesh.X
        F = self.forces if self.forces is not None else np.zeros((len(X), 3))
        f, m = F.sum(0), np.cross(X, F).sum(0)
        for r in self.rigid:
            f = f + r.force
            m = m + np.cross(r.ref_point, r.force) + r.moment
        if self.gravity is not None and np.linalg.norm(self.gravity) > 0:
            _, a, c = self.mesh.element_geometry()
            per = self.layers and sum(L.density * L.thickness for L in self.layers)
            w = a * (per if self.layers else self.mesh.element_thickness() * self.density)
            f = f + w.sum() * self.gravity
            m = m + np.cross((c * w[:, None]).sum(0), self.gravity)
        return f, m


@dataclass
class ShellResult:
    U: np.ndarray                  # (n,3) midsurface displacement, mm
    S_bot: np.ndarray              # (n,6) stress on the inner face (xx yy zz xy yz zx), MPa
    S_top: np.ndarray              # (n,6) stress on the outer face
    profiles: list                 # per node: (s positions through the thickness, (k,6) strains), tensor shear
    reaction: np.ndarray           # (3,) total support force, N
    t_solve: float = 0.0
    solver: str = ""
    n_expanded: int = 0
    post_vm: np.ndarray | None = None       # von Mises stress on the posts' surfaces, MPa
    post_xyz: np.ndarray | None = None

    def von_mises(self) -> np.ndarray:
        """Larger of the inner and outer face von Mises stress at each node."""
        return np.maximum(von_mises(self.S_bot), von_mises(self.S_top))


def von_mises(S: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, syz, szx = S.T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def element_frames(mesh: ShellMesh, ref: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each element's first in plane axis (the reference direction laid onto it), second axis and normal."""
    n, _, _ = mesh.element_geometry()
    r = np.asarray(ref if ref is not None else _longest_axis(mesh.X), float)
    r = r / np.linalg.norm(r)
    e1 = r[None] - (n @ r)[:, None] * n
    weak = np.linalg.norm(e1, axis=1) < 0.2     # the reference runs nearly along the normal: use another axis
    if weak.any():
        alt = np.eye(3)[int(np.argmin(np.abs(r)))]
        e1[weak] = alt[None] - (n[weak] @ alt)[:, None] * n[weak]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    return e1, np.cross(n, e1), n


def _longest_axis(X: np.ndarray) -> np.ndarray:
    return np.eye(3)[int(np.argmax(np.ptp(X, axis=0)))]


def write_shell_inp(path: Path, model: ShellModel) -> dict[str, tuple[int, int]]:
    """Write the deck. Returns rigid body name -> (ref node, rot node), 0 based."""
    mesh = model.mesh
    beams = model.place_posts()
    n = mesh.n_nodes + len(beams)
    extra, pts = {}, [beams]
    for k, r in enumerate(model.rigid):
        pts.append(np.asarray([r.ref_point, r.ref_point], float))
        extra[r.name] = (n + 2 * k, n + 2 * k + 1)
    with open(path, "w") as f:
        f.write("*HEADING\nWobbl shell model\n*NODE, NSET=NALL\n")
        for i, p in enumerate(np.vstack([mesh.X, *pts])):
            f.write(f"{i + 1},{p[0]:.10g},{p[1]:.10g},{p[2]:.10g}\n")
        f.write("*ELEMENT, TYPE=S6, ELSET=EALL\n")
        for e, row in enumerate(mesh.tris + 1):
            f.write(f"{e + 1}," + ",".join(map(str, row)) + "\n")
        eid = mesh.n_elems
        for k, p in enumerate(model.posts):
            f.write(f"*ELEMENT, TYPE=B32R, ELSET=POST{k + 1}\n")
            p.elements = []
            for j in range(0, len(p.nodes) - 2, 2):
                eid += 1
                p.elements.append(eid)
                f.write(f"{eid},{p.nodes[j] + 1},{p.nodes[j + 1] + 1},{p.nodes[j + 2] + 1}\n")
            f.write(f"*ELEMENT, TYPE=B32R, ELSET=SPIDER{k + 1}\n")
            for mid, foot in zip(p.spokes, p.footprint):
                eid += 1
                f.write(f"{eid},{p.nodes[-1] + 1},{mid + 1},{foot + 1}\n")
        _fold_normals(f, mesh)
        for name, nodes, _ in model.constraints:
            _nset(f, _set(name), nodes)
        for k, r in enumerate(model.rigid):
            _nset(f, f"RB{k + 1}", r.nodes)
        if model.layers:
            _laminate(f, model)
        else:
            f.write(f"*MATERIAL, NAME=MAT\n*ELASTIC\n{model.E:.10g},{model.nu:.10g}\n*DENSITY\n{model.density:.10g}\n")
            f.write("*SHELL SECTION, ELSET=EALL, MATERIAL=MAT, NODAL THICKNESS\n1.\n*NODAL THICKNESS\n")
            for i, t in enumerate(mesh.thickness):
                f.write(f"{i + 1},{t:.8g}\n")
            for k, p in enumerate(model.posts):
                from .posts import beam_normal

                b = beam_normal(p.end - p.start)
                sec = (f"PIPE\n{p.r_out:.8g},{p.r_out - p.r_in:.8g}" if p.r_in > 0 else f"CIRC\n{p.r_out:.8g}")
                f.write(f"*BEAM SECTION, ELSET=POST{k + 1}, MATERIAL=MAT, SECTION={sec}\n"
                        f"{b[0]:.8f},{b[1]:.8f},{b[2]:.8f}\n")
                # the spokes lie square to the post, so the post's axis gives their section direction
                ax = (p.end - p.start) / max(p.length, 1e-12)
                f.write(f"*BEAM SECTION, ELSET=SPIDER{k + 1}, MATERIAL=STIFF, SECTION=CIRC\n{0.2 * p.r_out:.8g}\n"
                        f"{ax[0]:.8f},{ax[1]:.8f},{ax[2]:.8f}\n")
            if model.posts:
                # thin spokes a thousand times stiffer than the part: a rigid boss top that adds no weight
                f.write(f"*MATERIAL, NAME=STIFF\n*ELASTIC\n{1000 * model.E:.10g},{model.nu:.10g}\n*DENSITY\n1e-30\n")
        for r in model.rigid:
            _rigid_equations(f, r, extra[r.name], mesh.X)
        f.write("*STEP\n*STATIC\n*BOUNDARY\n")
        for name, _, dofs in model.constraints:
            for d in dofs:
                f.write(f"{_set(name)},{d},{d}\n")
        F = model.forces if model.forces is not None else np.zeros((mesh.n_nodes, 3))
        rows = np.flatnonzero(np.abs(F).max(1) > 0)
        if len(rows) or model.rigid:
            f.write("*CLOAD\n")
            for i in rows:
                for d in range(3):
                    if F[i, d] != 0.0:
                        f.write(f"{i + 1},{d + 1},{F[i, d]:.10e}\n")
            for r in model.rigid:
                ref, rot = extra[r.name]
                for d in range(3):
                    if r.force[d] != 0:
                        f.write(f"{ref + 1},{d + 1},{r.force[d]:.10e}\n")
                    if r.moment[d] != 0:
                        f.write(f"{rot + 1},{d + 1},{r.moment[d]:.10e}\n")
        if model.gravity is not None and np.linalg.norm(model.gravity) > 0:
            g = np.asarray(model.gravity, float)
            mag = float(np.linalg.norm(g))
            f.write(f"*DLOAD\nEALL,GRAV,{mag:.10g},{g[0] / mag:.10g},{g[1] / mag:.10g},{g[2] / mag:.10g}\n")
        # the supports' reactions sit on the shell nodes, which the 3D results leave out, so print their totals
        for name, _, _ in model.constraints:
            f.write(f"*NODE PRINT, NSET={_set(name)}, TOTALS=ONLY\nRF\n")
        f.write("*NODE FILE, OUTPUT=3D\nU\n*EL FILE, OUTPUT=3D\nS,E\n*END STEP\n")
    return extra


def _rigid_equations(f, r: RigidCoupling, refrot: tuple[int, int], X: np.ndarray):
    """A rigid attachment as linear equations: every node moves with the reference node plus the small turn the
    rotation node's translations stand for, u = u_ref + theta x (x - x_ref).

    CalculiX refuses a rigid body on shell nodes, whose copies through the thickness it ties together itself;
    linear equations go through, and for the small deflections of a linear study they are the same joint.
    """
    ref, rot = refrot
    f.write("*EQUATION\n")
    c = np.asarray(r.ref_point, float)
    for i in r.nodes:
        x, y, z = X[i] - c
        # (theta x r)_x = ty z - tz y, (theta x r)_y = tz x - tx z, (theta x r)_z = tx y - ty x
        for d, terms in ((1, ((2, z), (3, -y))), (2, ((3, x), (1, -z))), (3, ((1, y), (2, -x)))):
            row = [(i, d, 1.0), (ref, d, -1.0)] + [(rot, a, -v) for a, v in terms if abs(v) > 1e-12]
            f.write(f"{len(row)}\n" + ",".join(f"{n + 1},{dd},{v:.10g}" for n, dd, v in row) + "\n")


FOLD_COS = 0.94   # an element more than about 20 degrees off the mean normal at a node keeps its own there


def element_node_normals(mesh: ShellMesh) -> np.ndarray:
    """(m,6,3) the normal each element uses at each of its nodes.

    The elements around a node fall into groups whose normals lie within about 20 degrees of each other: one
    group along a smooth wall, two across a fold. Every element in a group gets the group's mean normal, so a
    smooth wall shares one normal per node (CalculiX adds nothing there) and a fold splits, where CalculiX
    knots the node's copies together.
    """
    en, a, _ = mesh.element_geometry()
    m = mesh.n_elems
    pairs = np.column_stack([mesh.tris.ravel(), np.repeat(np.arange(m), 6), np.tile(np.arange(6), m)])
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    cut = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    nn = np.zeros((m, 6, 3))
    for grp in np.split(pairs, cut):
        els = grp[:, 1]
        normals = en[els]
        label = np.full(len(els), -1)
        for k in np.argsort(-a[els], kind="stable"):     # the largest element seeds each group
            if label[k] >= 0:
                continue
            free = label < 0
            label[free & (normals @ normals[k] >= FOLD_COS)] = k
        for g in np.unique(label):
            sel = label == g
            v = (normals[sel] * a[els[sel], None]).sum(0)
            nn[els[sel], grp[sel, 2]] = v / max(np.linalg.norm(v), 1e-300)
    # an element whose expanded faces would fold keeps its own normal at all six nodes: a prism of a flat
    # triangle is always valid, and CalculiX knots the node copies it shares with its neighbors
    bad = folded_expansion(mesh, nn, en)
    nn[bad] = en[bad][:, None, :]
    return nn


def folded_expansion(mesh: ShellMesh, nn: np.ndarray, en: np.ndarray) -> np.ndarray:
    """Elements whose inner or outer face, half a thickness along the node normals, folds over: one of its four
    sub triangles through the midside nodes faces against the element."""
    t = mesh.thickness[mesh.tris][..., None]
    X = mesh.X[mesh.tris]
    bad = np.zeros(mesh.n_elems, bool)
    subs = ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5))
    for side in (-0.5, 0.5):
        P = X + side * t * nn
        for i, j, k in subs:
            c = np.cross(P[:, j] - P[:, i], P[:, k] - P[:, i])
            bad |= np.einsum("ij,ij->i", c, en) <= 0
    return bad


def _fold_normals(f, mesh: ShellMesh):
    """Give every element its normal at each node, smooth along a wall and split at a fold.

    CalculiX expands a shell node along the mean normal of all its elements. Across a fold (a box's lid meeting
    its side) that mean runs diagonally, and an element beside the fold, expanded half a thickness along it,
    turns inside out. Here each element averages only the elements that run on with it, so it expands straight
    through its own wall, and CalculiX ties the copies of a fold's nodes together with a rigid knot.
    """
    import io

    nn = element_node_normals(mesh).reshape(-1, 3)
    ids = np.column_stack([np.repeat(np.arange(mesh.n_elems) + 1, 6), mesh.tris.ravel() + 1])
    buf = io.StringIO()
    np.savetxt(buf, np.column_stack([ids, nn]), fmt=["%d", "%d", "%.7f", "%.7f", "%.7f"], delimiter=",")
    f.write("*NORMAL\n" + buf.getvalue())


def _laminate(f, model: ShellModel):
    """Ply materials, one orientation per distinct ply frame and one composite section per element group.

    Each ply's fibers run at its angle from the reference direction laid onto the element, turning toward the
    element's second axis. Elements whose frames agree (all of a flat plate, a row along a tube) share their
    orientations and their section, so a deck stays small.
    """
    mats = {}
    for L in model.layers:
        if L.name not in mats:
            mats[L.name] = f"PLY{len(mats) + 1}"
            c = L.constants
            f.write(f"*MATERIAL, NAME={mats[L.name]}\n*ELASTIC, TYPE=ENGINEERING CONSTANTS\n"
                    + ",".join(f"{v:.8g}" for v in c[:8]) + f"\n{c[8]:.8g}\n*DENSITY\n{L.density:.10g}\n")
    e1, e2, nrm = element_frames(model.mesh, model.ref_dir)
    ang = np.radians([L.angle for L in model.layers])
    a = np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :]   # (m,plies,3)
    b = np.cross(np.broadcast_to(nrm[:, None, :], a.shape), a)
    keys = np.round(np.concatenate([a, b], axis=2), 4).reshape(-1, 6) + 0.0      # + 0.0 folds -0.0 into 0.0
    uniq, which = np.unique(keys, axis=0, return_inverse=True)
    f.write("".join(f"*ORIENTATION, NAME=OR{i + 1}, SYSTEM=RECTANGULAR\n" + ",".join(f"{v:.6f}" for v in row) + "\n"
                    for i, row in enumerate(uniq)))
    per_elem = which.reshape(len(e1), len(model.layers))
    stacks, group = np.unique(per_elem, axis=0, return_inverse=True)
    group = group.ravel()
    for g, names in enumerate(stacks):
        f.write(f"*ELSET, ELSET=LAM{g + 1}\n")
        ids = np.flatnonzero(group == g) + 1
        for i in range(0, len(ids), 12):
            f.write(", ".join(map(str, ids[i:i + 12])) + "\n")
        f.write(f"*SHELL SECTION, ELSET=LAM{g + 1}, COMPOSITE\n")
        for L, o in zip(model.layers, names):
            f.write(f"{L.thickness:.8g},,{mats[L.name]},OR{o + 1}\n")


def _nset(f, name: str, nodes):
    f.write(f"*NSET, NSET={name}\n")
    ids = np.asarray(nodes, dtype=np.int64) + 1
    for i in range(0, len(ids), 12):
        f.write(", ".join(map(str, ids[i:i + 12])) + "\n")


def _set(name: str) -> str:
    import re
    return ("N" + re.sub(r"[^A-Za-z0-9_]", "_", name).upper())[:70]


def solve_shell(model: ShellModel, workdir: str | Path, job: str = "shell", timeout: float = 3600) -> ShellResult:
    from ..solvers import base as solvers
    from ..solvers.calculix import _run

    adapter = solvers.for_domain("structural")
    ccx = adapter.require()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    write_shell_inp(work / f"{job}.inp", model)
    log, _ = _run(ccx, work, job, timeout)
    res = read_shell(work / f"{job}.frd", model)
    res.reaction = support_totals(work / f"{job}.dat")
    res.t_solve = time.time() - t0
    res.solver = f"CalculiX {adapter.version()}"
    size = float(np.linalg.norm(np.ptp(model.mesh.X, axis=0)))
    if not np.isfinite(res.U).all() or np.abs(res.U).max() > 1e3 * size:
        raise SolverError("the part is not held in place (it can move or spin freely)",
                          "Add fixtures that stop sliding and rotation", log[-2000:])
    return res


def read_frd_nodes(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Node ids, coordinates and the last result of each nodal block of an ASCII .frd.

    The records are fixed width, so each block is cut into columns by numpy in one go: a composite shell model
    writes a node for every ply boundary, and a line by line reader spends a minute on its 150,000 nodes.
    """
    raw = Path(path).read_bytes().splitlines()
    n = len(raw)
    ids, xyz, blocks = np.zeros(0, np.int64), np.zeros((0, 3)), {}
    i = 0
    while i < n:
        line = raw[i]
        if line.startswith(b"    2C"):
            j = i + 1
            while j < n and raw[j].startswith(b" -1"):
                j += 1
            ids = _fixed(raw[i + 1:j], 3, 10, 1, np.int64)[:, 0]
            xyz = _fixed(raw[i + 1:j], 13, 12, 3)
            i = j
            continue
        if line.startswith(b" -4"):
            name = line.split()[1].decode()
            j = i + 1
            while j < n and raw[j].startswith(b" -5"):
                j += 1
            k = j
            while k < n and not raw[k].startswith(b" -3"):
                k += 1
            blocks[name] = _result_block(raw[j:k])
            i = k + 1
            continue
        i += 1
    return ids, xyz, blocks


def _fixed(lines: list[bytes], start: int, width: int, count: int, kind=float) -> np.ndarray:
    """(len(lines), count) values from fixed width fields starting at a column; missing fields read 0."""
    if not lines:
        return np.zeros((0, count), kind)
    w = start + width * count
    a = np.array(lines, dtype=f"S{w}")
    cols = np.ascontiguousarray(a.view("S1").reshape(len(lines), w)[:, start:])
    f = cols.view(f"S{width}").reshape(len(lines), count)
    return np.where(f == b"", b"0", f).astype(kind)


def _result_block(body: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Node ids and values of one result block; a record longer than six values carries on in -2 lines."""
    if any(b.startswith(b" -2") for b in body):
        rows, vals = [], []
        for b in body:
            v = [float(b[c:c + 12]) for c in range(13, len(b) - 11, 12)]
            if b.startswith(b" -1"):
                rows.append(int(b[3:13]))
                vals.append(v)
            elif vals:
                vals[-1] = vals[-1] + v
        width = max((len(v) for v in vals), default=0)
        mat = np.zeros((len(vals), width))
        for r, v in enumerate(vals):
            mat[r, :len(v)] = v
        return np.asarray(rows, dtype=np.int64), mat
    width = max(((len(b) - 13) // 12 for b in body), default=0)
    return _fixed(body, 3, 10, 1, np.int64)[:, 0], _fixed(body, 13, 12, width)


def expanded_owner(mesh: ShellMesh, P: np.ndarray, thickness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each expanded node, the shell node it grew from and its signed position along that node's normal
    (negative toward the inner face).

    Most expanded nodes sit right on the normal of their nearest shell node; only those around folds, which
    CalculiX expanded along an element's own normal, need the slower search through every nearby normal.
    """
    from scipy.spatial import cKDTree

    k = min(4, mesh.n_nodes)
    _, near = cKDTree(mesh.X).query(P, k=k)
    near = near.reshape(len(P), k)
    nrm = mesh.normals[near]
    rel = P[:, None, :] - mesh.X[near]
    s = np.einsum("pkj,pkj->pk", rel, nrm)
    perp = np.linalg.norm(rel - s[..., None] * nrm, axis=-1)
    half = 0.5 * thickness[near]
    perp = np.where(np.abs(s) <= 1.05 * half + 1e-6, perp, np.inf)
    two = np.sort(perp, axis=1)[:, :2] if k > 1 else np.column_stack([perp[:, 0], np.full(len(P), np.inf)])
    j = np.argmin(perp, axis=1)
    rows = np.arange(len(P))
    owner, pos = near[rows, j], s[rows, j]
    unsure = ~((two[:, 0] <= 0.2 * half[rows, j]) & (two[:, 0] <= 0.1 * two[:, 1]))
    if unsure.any():
        owner[unsure], pos[unsure] = _owner_search(mesh, P[unsure], thickness)
    return owner, pos


def _owner_search(mesh: ShellMesh, P: np.ndarray, thickness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    en, _, _ = mesh.element_geometry()
    n = mesh.n_nodes
    # up to eight normals per node: its own and those of the elements around it
    cand_n = np.repeat(mesh.normals[:, None, :], 8, axis=1)
    count = np.ones(n, dtype=np.int64)
    for k in range(6):
        for e, i in enumerate(mesh.tris[:, k]):
            if count[i] < 8:
                cand_n[i, count[i]] = en[e]
                count[i] += 1
    k = min(8, n)
    _, near = cKDTree(mesh.X).query(P, k=k)
    near = near.reshape(len(P), k)
    rel = P[:, None, :] - mesh.X[near]                                  # (p,k,3)
    s = np.einsum("pkj,pkmj->pkm", rel, cand_n[near])                   # (p,k,8)
    perp = np.linalg.norm(rel[:, :, None, :] - s[..., None] * cand_n[near], axis=-1)
    half = 0.5 * thickness[near][..., None]
    perp = np.where(np.abs(s) <= 1.05 * half + 1e-6, perp, np.inf)
    flat = perp.reshape(len(P), -1)
    j = np.argmin(flat, axis=1)
    owner = near[np.arange(len(P)), j // 8]
    pos = s.reshape(len(P), -1)[np.arange(len(P)), j]
    return owner, pos


def read_shell(path: str | Path, model: ShellModel) -> ShellResult:
    mesh = model.mesh
    ids, P, blocks = read_frd_nodes(path)
    if "DISP" not in blocks or "STRESS" not in blocks:
        raise SolverError("CalculiX finished without displacement or stress results")
    # the shell, post and rigid body nodes keep their own numbers; the expanded nodes come after them, and the
    # ones the posts' beams grew into are told apart by the expansion CalculiX lists in the .12d file
    keep = ids > mesh.n_nodes + model.n_post_nodes + 2 * len(model.rigid)
    grown = beam_expansion(Path(path).with_suffix(".12d")) if model.posts else {}
    beam = set().union(*grown.values()) if grown else set()
    posts = set().union(*(grown.get(e, set()) for p in model.posts for e in getattr(p, "elements", [])))
    in_beam = np.array([int(i) in beam for i in ids]) & keep
    in_post = np.array([int(i) in posts for i in ids]) & keep
    post_ids, post_P = ids[in_post], P[in_post]
    keep &= ~in_beam
    ids, P = ids[keep], P[keep]
    t = model.section_thickness()
    owner, pos = expanded_owner(mesh, P, t)
    lut = {int(i): k for k, i in enumerate(ids)}

    def field(name, width):
        r, m = blocks[name]
        out = np.zeros((len(ids), width))
        k = np.array([lut.get(int(i), -1) for i in r])
        ok = k >= 0
        out[k[ok], :min(width, m.shape[1])] = m[ok, :width]
        return out

    U, S, E = field("DISP", 3), field("STRESS", 6), field("TOSTRAIN", 6) if "TOSTRAIN" in blocks else None
    n = mesh.n_nodes
    cnt = np.bincount(owner, minlength=n).astype(float)
    Um = np.zeros((n, 3))
    np.add.at(Um, owner, U)
    Um /= np.maximum(cnt, 1)[:, None]
    order = np.lexsort((pos, owner))
    starts = np.searchsorted(owner[order], np.arange(n))
    ends = np.searchsorted(owner[order], np.arange(n), side="right")
    lo, hi = order[np.minimum(starts, len(order) - 1)], order[np.maximum(ends - 1, 0)]
    S_bot, S_top = S[lo], S[hi]
    profiles = []
    if E is not None:
        for i in range(n):
            idx = order[starts[i]:ends[i]]
            profiles.append((pos[idx], E[idx]))
    res = ShellResult(Um, S_bot, S_top, profiles, np.zeros(3), n_expanded=len(ids))
    if len(post_ids):
        r, m = blocks["STRESS"]
        rows = {int(i): k for k, i in enumerate(r)}
        Sp = np.array([m[rows[int(i)], :6] if int(i) in rows else np.zeros(6) for i in post_ids])
        res.post_vm, res.post_xyz = von_mises(Sp), post_P
    return res


def beam_expansion(path: Path) -> dict[int, set[int]]:
    """Expanded node numbers of each beam element, from CalculiX's .12d listing."""
    import re

    out: dict[int, set[int]] = {}
    cur = None
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "ELEMENT" in line and "with label" in line:
            cur = int(line.split()[1]) if '"B3' in line else None
            if cur is not None:
                out[cur] = set()
        elif cur is not None and re.match(r"^\s*\d", line):
            out[cur].update(int(v) for v in line.split())
    return out


def support_totals(dat: str | Path) -> np.ndarray:
    """Sum of the total support forces CalculiX printed for each held node set, N."""
    import re

    total = np.zeros(3)
    lines = Path(dat).read_text(errors="replace").splitlines() if Path(dat).exists() else []
    for i, ln in enumerate(lines):
        if re.search(r"total force \(fx,fy,fz\) for set", ln):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                total += np.array([float(v) for v in lines[j].split()[-3:]])
    return total


__all__ = ["Layer", "ShellModel", "ShellResult", "solve_shell", "write_shell_inp", "read_shell", "element_frames",
           "von_mises"]
