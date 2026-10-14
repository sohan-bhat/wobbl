"""Tetrahedra (CalculiX C3D10 and C3D4 node order): stiffness, lumped mass and stress at the nodes, in numpy.

The drop study integrates the motion itself, so it needs the element matrices in memory. They are the textbook
isoparametric ones, with the same node order and full four point integration as CalculiX, and a test checks the
assembled stiffness against the matrix CalculiX writes for the same mesh.

Corner nodes 1 to 4 sit at natural coordinates (0,0,0), (1,0,0), (0,1,0), (0,0,1); midside nodes 5 to 10 sit on the
edges 1-2, 2-3, 3-1, 1-4, 2-4 and 3-4.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
NODE_XI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                    [.5, 0, .5], [0, .5, .5]], float)
_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS = np.array([[_B, _B, _B], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A]])
WEIGHT = 1.0 / 24.0
# HRZ lumping of the consistent mass: corners 6/216 and midside nodes 32/216 of the element's mass
HRZ = np.array([1 / 36] * 4 + [4 / 27] * 6)


def shape_derivatives(xi: np.ndarray, nodes: int = 10) -> np.ndarray:
    """(p,nodes,3) derivatives of the shape functions with respect to the natural coordinates at points (p,3), for
    the ten node tet or, with nodes=4, the linear one."""
    xi = np.atleast_2d(xi)
    if nodes == 4:
        return np.repeat(np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)[None], len(xi), axis=0)
    L = np.column_stack([1.0 - xi.sum(1), xi])            # (p,4) volume coordinates
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)   # dL_i / dxi_j
    dN_dL = np.zeros((len(xi), 10, 4))
    for i in range(4):
        dN_dL[:, i, i] = 4.0 * L[:, i] - 1.0
    for k, (i, j) in enumerate(EDGES):
        dN_dL[:, 4 + k, i] = 4.0 * L[:, j]
        dN_dL[:, 4 + k, j] = 4.0 * L[:, i]
    return dN_dL @ dL


def _gradients(Xe: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the shape functions in space (m,p,nodes,3) and the Jacobian determinants (m,p)."""
    G = shape_derivatives(xi, Xe.shape[1])                     # (p,nodes,3)
    J = np.einsum("pai,maj->mpij", G, Xe)                     # J_ij = sum_a dN_a/dxi_i x_aj
    det = np.linalg.det(J)
    inv = np.linalg.inv(J)
    return np.einsum("pai,mpji->mpaj", G, inv), det           # dN/dx_j = sum_i dN/dxi_i dxi_i/dx_j


def elasticity(E: float, nu: float) -> np.ndarray:
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def _rule(nodes: int):
    """Integration points and weight, and the natural coordinates of the nodes."""
    if nodes == 4:
        return np.array([[0.25, 0.25, 0.25]]), 1.0 / 6.0, NODE_XI[:4]
    return GAUSS, WEIGHT, NODE_XI


def _bmatrix(dN: np.ndarray) -> np.ndarray:
    """(m,6,3k) strain displacement matrices, engineering shear in the order xy, yz, zx, dofs node by node."""
    m = len(dN)
    B = np.zeros((m, 6, 3 * dN.shape[1]))
    x, y, z = dN[..., 0], dN[..., 1], dN[..., 2]
    B[:, 0, 0::3], B[:, 1, 1::3], B[:, 2, 2::3] = x, y, z
    B[:, 3, 0::3], B[:, 3, 1::3] = y, x
    B[:, 4, 1::3], B[:, 4, 2::3] = z, y
    B[:, 5, 0::3], B[:, 5, 2::3] = z, x
    return B


def stiffness(X: np.ndarray, tets: np.ndarray, E: float, nu: float, chunk: int = 4000) -> sp.csr_matrix:
    """Global stiffness of a C3D10 or C3D4 mesh, dofs 3 i + (0, 1, 2), symmetric CSR."""
    n = len(X)
    D = elasticity(E, nu)
    k = 3 * tets.shape[1]
    pts, wt, _ = _rule(tets.shape[1])
    dofs = (3 * tets[:, :, None] + np.arange(3)[None, None, :]).reshape(len(tets), k)
    K = sp.csr_matrix((3 * n, 3 * n))
    for s in range(0, len(tets), chunk):
        T = tets[s:s + chunk]
        dN, det = _gradients(X[T], pts)
        Ke = np.zeros((len(T), k, k))
        for g in range(len(pts)):
            B = _bmatrix(dN[:, g])
            Ke += np.einsum("mki,kl,mlj->mij", B, D, B) * (wt * det[:, g])[:, None, None]
        d = dofs[s:s + chunk]
        rows = np.repeat(d, k, axis=1).ravel()
        cols = np.tile(d, (1, k)).ravel()
        K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(3 * n, 3 * n))
    return K


def volumes(X: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts, wt, _ = _rule(tets.shape[1])
    _, det = _gradients(X[tets], pts)
    return (det * wt).sum(1)


def lumped_mass(X: np.ndarray, tets: np.ndarray, density: float, scheme: str = "hrz") -> np.ndarray:
    """(n,) lumped nodal masses, t: HRZ for the ten node tet (or a tenth each with scheme='even'), a quarter each
    for the linear one."""
    m = np.zeros(len(X))
    me = density * volumes(X, tets)
    share = (HRZ if scheme == "hrz" else np.full(10, 0.1)) if tets.shape[1] == 10 else np.full(4, 0.25)
    np.add.at(m, tets, me[:, None] * share[None, :])
    return m


class StressRecovery:
    """Stress at the nodes from a displacement field: the strain at each element's own nodes, averaged over the
    elements that share a node, the way CalculiX extrapolates and averages."""

    def __init__(self, X: np.ndarray, tets: np.ndarray, E: float, nu: float):
        self.tets = tets
        self.dN, _ = _gradients(X[tets], _rule(tets.shape[1])[2])  # (m,nodes,shape functions,3)
        self.dN = self.dN.astype(np.float32)
        self.lam, self.mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        count = np.bincount(tets.ravel(), minlength=len(X)).astype(float)
        # averaging over the elements that share a node, as one sparse product
        rows = tets.ravel()
        self.avg = sp.csr_matrix((1.0 / count[rows], (rows, np.arange(rows.size))), shape=(len(X), rows.size))

    def stress(self, U: np.ndarray) -> np.ndarray:
        """(n,6) nodal stress xx yy zz xy yz zx, MPa, for displacements (n,3)."""
        ue = U[self.tets].astype(np.float32)                        # (m,nodes,3)
        grad = np.swapaxes(np.matmul(np.swapaxes(self.dN, 2, 3), ue[:, None]), 2, 3)   # du_i/dx_k at each node
        eps = 0.5 * (grad + np.swapaxes(grad, -1, -2))
        tr = np.trace(eps, axis1=-2, axis2=-1)
        s = 2 * self.mu * eps
        s[..., 0, 0] += self.lam * tr
        s[..., 1, 1] += self.lam * tr
        s[..., 2, 2] += self.lam * tr
        voigt = np.stack([s[..., 0, 0], s[..., 1, 1], s[..., 2, 2], s[..., 0, 1], s[..., 1, 2], s[..., 2, 0]], -1)
        return self.avg @ voigt.reshape(-1, 6).astype(float)


def von_mises(S: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = S.T
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))


def max_principal(S: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = S.T
    T = np.stack([np.stack([xx, xy, zx], -1), np.stack([xy, yy, yz], -1), np.stack([zx, yz, zz], -1)], -2)
    return np.linalg.eigvalsh(T)[:, -1]


__all__ = ["stiffness", "lumped_mass", "volumes", "StressRecovery", "von_mises", "max_principal", "elasticity"]
