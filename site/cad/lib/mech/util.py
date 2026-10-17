"""Placing parts: build123d Locations and 4x4 matrices back and forth, and moving [(shape, material)] lists whose
shapes may be build123d parts or ready meshes (('mesh', P, N, T))."""
import numpy as np
from build123d import Location
from OCP.gp import gp_Trsf


def matrix(loc):
    """A build123d Location as a 4x4 numpy matrix."""
    t = loc.wrapped.Transformation()
    M = np.eye(4)
    for r in range(3):
        for c in range(4):
            M[r, c] = t.Value(r + 1, c + 1)
    return M


def location(M):
    """A 4x4 rigid matrix as a build123d Location."""
    M = np.asarray(M, dtype=np.float64)
    t = gp_Trsf()
    t.SetValues(*[float(M[r, c]) for r in range(3) for c in range(4)])
    return Location(t)


def frame(origin, x, z):
    """The matrix of a frame at origin whose x and z axes point along x and z (z is kept, x is squared to it)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.asarray(x, dtype=np.float64)
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


def place(parts, M):
    """Move every (shape, material) by M (a 4x4 matrix or a Location)."""
    if not isinstance(M, np.ndarray):
        M = matrix(M)
    loc = location(M)
    out = []
    for s, m in parts:
        if isinstance(s, tuple) and s and s[0] == 'mesh':
            P = s[1] @ M[:3, :3].T + M[:3, 3]
            N = s[2] @ M[:3, :3].T
            out.append((('mesh', P, N, s[3]), m))
        else:
            out.append((loc * s, m))
    return out
