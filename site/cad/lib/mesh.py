"""Triangles from build123d shapes: each face meshed on its own, so edges between faces stay crisp and a curved
face shades smooth with normals taken from the surface itself, then cube projected uvs for the tiling detail maps.
"""
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepLib import BRepLib_ToolTriangulatedShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS


# OCP 7.x names static methods with an _s suffix, OCP 8 without
def _static(cls, name):
    return getattr(cls, name, None) or getattr(cls, name + '_s')


_face = _static(TopoDS, 'Face')
_triangulation = _static(BRep_Tool, 'Triangulation')
_normals = _static(BRepLib_ToolTriangulatedShape, 'ComputeNormals')


def _wrapped(shape):
    return shape.wrapped if hasattr(shape, 'wrapped') else shape


def tessellate(shape, tol=0.02, ang=0.2):
    """(P (n,3), N (n,3), T (m,3)) for a shape, in its own units. tol is the chord deviation and ang the angle
    in radians a facet may turn through, so small round parts stay round and large ones stay light."""
    w = _wrapped(shape)
    BRepMesh_IncrementalMesh(w, tol, False, ang, True)
    Ps, Ns, Ts, base = [], [], [], 0
    ex = TopExp_Explorer(w, TopAbs_FACE)
    while ex.More():
        f = _face(ex.Current())
        ex.Next()
        loc = TopLoc_Location()
        tri = _triangulation(f, loc)
        if tri is None:
            continue
        if not tri.HasNormals():
            _normals(f, tri)
        tr = loc.Transformation()
        n = tri.NbNodes()
        P = np.empty((n, 3))
        N = np.empty((n, 3))
        for k in range(n):
            p = tri.Node(k + 1).Transformed(tr)
            P[k] = (p.X(), p.Y(), p.Z())
            d = tri.Normal(k + 1)
            N[k] = (d.X(), d.Y(), d.Z())
        # normals turn with the face's placement but do not move
        M = np.array([[tr.Value(r, c) for c in range(1, 4)] for r in range(1, 4)])
        N = N @ M.T
        T = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
        for k in range(tri.NbTriangles()):
            a, b, c = tri.Triangle(k + 1).Get()
            T[k] = (a - 1, b - 1, c - 1)
        if f.Orientation() == TopAbs_REVERSED:
            T = T[:, ::-1]
            N = -N
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        Ps.append(P)
        Ns.append(N)
        Ts.append(T + base)
        base += n
    if not Ps:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Ts)


def transform(P, N, M):
    """Points and normals through a 4x4 matrix (rotation and translation, uniform scale at most)."""
    M = np.asarray(M, dtype=np.float64)
    P2 = P @ M[:3, :3].T + M[:3, 3]
    N2 = N @ M[:3, :3].T
    N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
    return P2, N2


def box_uv(P, N, T, density):
    """Cube projected uvs at density tiles per unit: each triangle is laid on the axis plane its normal faces most,
    and a corner shared by triangles laid on different planes is split, so no texture smears across a seam.
    Returns new (P, N, UV, T)."""
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    axis = np.argmax(np.abs(fn), axis=1)
    sign = np.sign(fn[np.arange(len(T)), axis])
    side = axis * 2 + (sign < 0)
    key = T * 6 + side[:, None]
    uniq, inv = np.unique(key.reshape(-1), return_inverse=True)
    src = uniq // 6
    s = uniq % 6
    P2, N2 = P[src], N[src]
    UV = np.empty((len(src), 2))
    for k, (u, v) in enumerate([(1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1)]):
        m = s == k
        flip = -1 if k in (1, 2) else 1
        UV[m, 0] = P2[m, u] * density * flip
        UV[m, 1] = P2[m, v] * density
    return P2, N2, UV, inv.reshape(-1, 3)
