"""Modes and what Wobbl computes from them: mass, participation, harmonic and random response.

CalculiX finds the natural frequencies and mode shapes, each scaled to unit modal mass. Everything a vibration
answer needs past that is a sum over modes, which this module does in numpy:

    harmonic   q_i(f) = P_i / (w_i^2 - w^2 + 2 i zeta w_i w)       one complex amplitude per mode
    random     C_ij = integral of q_i q_j* times the input PSD       a modes x modes covariance

with P_i = -Gamma_i a for a base acceleration a (motion relative to the base) and P_i = phi_i . F for a force.
Stress follows the same sums through each mode's stress field. The von Mises stress of a random response is not
Gaussian, so its mean square comes from the covariance through the von Mises quadratic form (Segalman's method),
which is exact for the mean square.

Wobbl also builds the consistent mass matrix of the 10 node tetrahedra itself, which checks that the modes really
have unit modal mass and that the participation factors CalculiX prints add up to the mass of the part.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

G0 = 9806.65   # mm/s2 in one g

# the von Mises stress squared is s^T Q s, with s = (xx, yy, zz, xy, yz, zx)
VM_Q = np.zeros((6, 6))
VM_Q[:3, :3] = np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]])
VM_Q[3:, 3:] = 3.0 * np.eye(3)
_vals, _vecs = np.linalg.eigh(VM_Q)
VM_L = _vecs * np.sqrt(np.clip(_vals, 0.0, None))     # VM_Q = VM_L @ VM_L.T


@dataclass
class Modes:
    """The modes of one mesh, scaled to unit modal mass."""
    freq: np.ndarray                    # (m,) Hz
    shapes: np.ndarray                  # (m, n_all, 3) displacement per unit modal coordinate, mm/sqrt(t)
    stress: np.ndarray | None           # (m, n, 6) stress per unit modal coordinate
    participation: np.ndarray           # (m, 6) from CalculiX: x, y, z and rotations
    effective_mass: np.ndarray          # (m, 6) t
    total_mass: np.ndarray              # (6,) t, CalculiX's total effective mass
    n: int                              # mesh nodes; shapes may carry rigid attachment nodes after them
    mass_nodes: dict = field(default_factory=dict)    # attached mass -> nodes (indices into shapes)
    mass_values: dict = field(default_factory=dict)   # attached mass -> mass at each, t
    solver_total_mass: np.ndarray | None = None       # CalculiX's own totals, kept for comparison

    @property
    def count(self) -> int:
        return len(self.freq)

    def gamma(self, direction) -> np.ndarray:
        """Participation of each mode in a translation along direction, sqrt(t)."""
        d = np.asarray(direction, float)
        return self.participation[:, :3] @ (d / np.linalg.norm(d))

    def mass_fraction(self, direction) -> float:
        """Share of the part's mass the modes found move along direction (one means every mode that matters)."""
        g = self.gamma(direction)
        total = float(self.total_mass[:3].mean())
        return float((g ** 2).sum() / total) if total > 0 else 0.0


# ------------------------------------------------------------------------------------------------ mass
def tet10_mass() -> np.ndarray:
    from ..thermal.study import _tet10_mass

    return _tet10_mass()


def _volumes(mesh) -> np.ndarray:
    c = mesh.X[mesh.tets[:, :4]]
    return np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0], np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0


def _tet_quadrature(n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A collapsed Gauss rule on the unit tetrahedron, n^3 points, exact for polynomials up to degree 2n - 3."""
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = (g + 1) / 2, w / 2
    u, v, t = np.meshgrid(g, g, g, indexing="ij")
    wu, wv, wt = np.meshgrid(w, w, w, indexing="ij")
    pts = np.stack([u, v * (1 - u), t * (1 - u) * (1 - v)], -1).reshape(-1, 3)
    return pts, (wu * wv * wt * (1 - u) ** 2 * (1 - v)).reshape(-1)


def _tet10_shape(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions (q, 10) and their reference derivatives (q, 10, 3) in CalculiX C3D10 node order: corners,
    then the midsides of edges 0-1, 1-2, 2-0, 0-3, 1-3 and 2-3."""
    xi, eta, zeta = pts.T
    L = np.stack([1 - xi - eta - zeta, xi, eta, zeta], 1)                     # (q, 4)
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)    # (4, 3)
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    N = np.concatenate([L * (2 * L - 1), np.stack([4 * L[:, i] * L[:, j] for i, j in edges], 1)], 1)
    dN = np.concatenate([(4 * L - 1)[:, :, None] * dL[None], np.stack(
        [4 * (L[:, i, None] * dL[j] + L[:, j, None] * dL[i]) for i, j in edges], 1)], 1)
    return N, dN


_MASS_CACHE: dict = {}


def element_masses(mesh) -> np.ndarray:
    """Consistent mass matrices (e, 10, 10) of the mesh's 10 node tetrahedra at unit density, integrated over each
    element's real, curved shape. A midside node on a curved face makes an element larger than its corners say, and
    on a coarse mesh of a round part that is several percent of the mass. Cached for the last mesh asked about."""
    key = (id(mesh), mesh.tets.shape, float(mesh.X[mesh.tets[:, 4:]].sum()) if len(mesh.tets) else 0.0)
    if _MASS_CACHE.get("key") == key:
        return _MASS_CACHE["M"]
    pts, w = _tet_quadrature(5)
    N, dN = _tet10_shape(pts)
    NN = np.einsum("q,qa,qb->qab", w, N, N)
    out = np.empty((len(mesh.tets), 10, 10))
    for s in range(0, len(mesh.tets), 4000):
        Xe = mesh.X[mesh.tets[s:s + 4000]]                                     # (e, 10, 3)
        J = np.einsum("eai,qaj->eqij", Xe, dN)
        out[s:s + 4000] = np.einsum("qab,eq->eab", NN, np.abs(np.linalg.det(J)))
    _MASS_CACHE.clear()
    _MASS_CACHE.update(key=key, M=out)
    return out


def mass_times(mesh, density: float, U: np.ndarray, mass_nodes: dict | None = None,
               mass_values: dict | None = None) -> np.ndarray:
    """M U for displacement fields U (n_all, 3) with the consistent mass of 10 node tetrahedra plus point masses."""
    out = np.zeros_like(U, dtype=float)
    if mesh.tets.shape[1] == 10:
        ue = U[mesh.tets]                                                  # (e, 10, 3)
        fe = np.einsum("eab,ebd->ead", element_masses(mesh), ue) * density
    else:
        rho_v = density * _volumes(mesh)
        ue = U[mesh.tets]
        fe = (np.full((4, 4), 1.0) + np.eye(4)) / 20.0
        fe = np.einsum("ab,ebd->ead", fe, ue) * rho_v[:, None, None]
    np.add.at(out, mesh.tets, fe)
    for name, nodes in (mass_nodes or {}).items():
        out[nodes] += np.asarray(mass_values[name])[:, None] * U[nodes]
    return out


def part_mass(mesh, density: float, mass_values: dict | None = None) -> float:
    vol = element_masses(mesh).sum() if mesh.tets.shape[1] == 10 else _volumes(mesh).sum()
    return float(density * vol + sum(float(np.sum(v)) for v in (mass_values or {}).values()))


def modal_masses(mesh, density: float, modes: Modes) -> np.ndarray:
    """phi^T M phi for every mode, computed by Wobbl: one when CalculiX's scaling and Wobbl's mass agree."""
    return np.array([float(np.sum(s * mass_times(mesh, density, s.astype(float), modes.mass_nodes,
                                                  modes.mass_values)))
                     for s in modes.shapes])


def participation(mesh, density: float, modes: Modes, X_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Participation factors phi^T M r for unit translations along x, y and z and unit rotations about the mass
    center, (m, 6), and the totals r^T M r, (6,).

    Only mesh nodes and the nodes that carry an attached mass move with r. The rotation node of a rigid attachment
    holds its rotations, not a position, so it stays out; X_all gives every node's position, the attachment point
    for the extra ones.
    """
    n_all = modes.shapes.shape[1]
    carry = np.zeros(n_all, bool)
    carry[:mesh.n_nodes] = True
    for nodes in modes.mass_nodes.values():
        carry[nodes] = True
    total = part_mass(mesh, density, modes.mass_values)
    ones = np.zeros((n_all, 3))
    ones[carry] = 1.0
    w = mass_times(mesh, density, ones, modes.mass_nodes, modes.mass_values)[:, 0]   # each node's mass share
    c = (X_all[carry] * w[carry, None]).sum(0) / w[carry].sum()
    fields = []
    for k in range(3):
        r = np.zeros((n_all, 3))
        r[carry, k] = 1.0
        fields.append(r)
    for k in range(3):
        r = np.zeros((n_all, 3))
        r[carry] = np.cross(np.eye(3)[k], X_all[carry] - c)
        fields.append(r)
    Mr = [mass_times(mesh, density, r, modes.mass_nodes, modes.mass_values) for r in fields]
    gamma = np.array([[float(np.sum(s * m)) for m in Mr] for s in modes.shapes.astype(float)])
    tot = np.array([total] * 3 + [float(np.sum(r * m)) for r, m in zip(fields[3:], Mr[3:])])
    return gamma, tot


def rayleigh_frequency(mesh, density: float, u: np.ndarray, direction, accel: float = G0,
                       mass_nodes: dict | None = None, mass_values: dict | None = None) -> float:
    """Rayleigh's quotient with a static deflection u under ``accel`` along direction, Hz.

    For the same mesh this is an upper bound on the lowest natural frequency, and close to the frequency of the
    mode that moves most of the mass along that direction: the classic hand check of a vibration answer.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    r = np.zeros_like(u, dtype=float)
    r[:] = d
    mu = mass_times(mesh, density, u, mass_nodes, mass_values)
    mr = mass_times(mesh, density, r, mass_nodes, mass_values)
    work = accel * float(np.sum(u * mr))
    kinetic = float(np.sum(u * mu))
    if kinetic <= 0 or work <= 0:
        return float("nan")
    return float(np.sqrt(work / kinetic) / (2 * np.pi))


# -------------------------------------------------------------------------------------------- harmonic
def transfer(f: np.ndarray, fn: np.ndarray, zeta: float) -> np.ndarray:
    """H_i(f) = 1 / (w_i^2 - w^2 + 2 i zeta w_i w), (nf, m)."""
    w = 2 * np.pi * np.asarray(f, float)[:, None]
    wn = 2 * np.pi * np.asarray(fn, float)[None, :]
    return 1.0 / (wn ** 2 - w ** 2 + 2j * zeta * wn * w)


def harmonic_coords(f: np.ndarray, modes: Modes, zeta: float, modal_force: np.ndarray) -> np.ndarray:
    """Complex modal amplitudes (nf, m) for a modal force per mode (from base_force or point_force)."""
    return transfer(f, modes.freq, zeta) * np.asarray(modal_force)[None, :]


def base_force(modes: Modes, accel: float, direction) -> np.ndarray:
    """Modal forces of a base acceleration (mm/s2) along direction; the response is relative to the base."""
    return -modes.gamma(direction) * accel


def point_force(modes: Modes, F: np.ndarray) -> np.ndarray:
    """Modal forces of nodal forces F (n, 3)."""
    return np.einsum("mnd,nd->m", modes.shapes[:, :len(F)].astype(float), F)


def vm_amplitude(S: np.ndarray) -> np.ndarray:
    """Largest von Mises stress over one cycle of a harmonic stress S (n, 6) complex.

    With s(t) = a cos wt - b sin wt, s^T Q s peaks at (A + B) / 2 + sqrt(((A - B) / 2)^2 + C^2), where A = a^T Q a,
    B = b^T Q b and C = a^T Q b, so phase differences between components are counted exactly.
    """
    a, b = S.real, S.imag
    A = np.einsum("ni,ij,nj->n", a, VM_Q, a)
    B = np.einsum("ni,ij,nj->n", b, VM_Q, b)
    C = np.einsum("ni,ij,nj->n", a, VM_Q, b)
    return np.sqrt(np.clip(0.5 * (A + B) + np.sqrt(0.25 * (A - B) ** 2 + C ** 2), 0.0, None))


def stress_at(q: np.ndarray, modes: Modes) -> np.ndarray:
    """Complex stress field (n, 6) for modal amplitudes q (m,)."""
    return np.einsum("m,mnc->nc", q, modes.stress.astype(float))


# ---------------------------------------------------------------------------------------------- random
@dataclass
class PSD:
    """An acceleration power spectral density, g^2/Hz, straight between breakpoints on log-log axes."""
    f: np.ndarray
    g2: np.ndarray
    name: str = "custom"

    def __call__(self, f) -> np.ndarray:
        f = np.asarray(f, float)
        out = np.zeros_like(f)
        inside = (f >= self.f[0]) & (f <= self.f[-1])
        out[inside] = np.exp(np.interp(np.log(f[inside]), np.log(self.f), np.log(self.g2)))
        return out

    def grms(self) -> float:
        """Overall level, integrating each power law segment exactly."""
        tot = 0.0
        for (f1, p1), (f2, p2) in zip(zip(self.f, self.g2), zip(self.f[1:], self.g2[1:])):
            k = np.log(p2 / p1) / np.log(f2 / f1)
            tot += p1 * f1 / (k + 1) * ((f2 / f1) ** (k + 1) - 1) if abs(k + 1) > 1e-9 else p1 * f1 * np.log(f2 / f1)
        return float(np.sqrt(tot))

    def describe(self) -> str:
        pts = ", ".join(f"{f:g} Hz {p:g}" for f, p in zip(self.f, self.g2))
        return f"{self.name} ({pts} g2/Hz, {self.grms():.3g} grms)"


# NASA GSFC-STD-7000 (GEVS), generalized random vibration for components of 22.7 kg or less: +6 dB per octave from
# 20 to 50 Hz, flat to 800 Hz, -6 dB per octave to 2000 Hz. Qualification is 14.1 grms, acceptance 3 dB lower.
PROFILES = {
    "gevs-qual": [(20.0, 0.026), (50.0, 0.16), (800.0, 0.16), (2000.0, 0.026)],
    "gevs-accept": [(20.0, 0.013), (50.0, 0.08), (800.0, 0.08), (2000.0, 0.013)],
}
PROFILE_NAMES = {"gevs-qual": "NASA GEVS qualification", "gevs-accept": "NASA GEVS acceptance"}


def parse_psd(text: str) -> PSD:
    """A named profile ('gevs-qual', 'gevs-accept') or breakpoints '20:0.01, 80:0.04, 500:0.04, 2000:0.01'
    (Hz:g2/Hz, at least two)."""
    t = text.strip().lower()
    if t in PROFILES:
        f, p = zip(*PROFILES[t])
        return PSD(np.array(f), np.array(p), PROFILE_NAMES[t])
    pts = re.findall(r"([\d.]+(?:e[-+]?\d+)?)\s*(?:hz)?\s*[:=@ ]\s*([\d.]+(?:e[-+]?\d+)?)", t)
    if len(pts) < 2:
        raise ValueError(f"could not read the PSD '{text}'. Use a profile ({', '.join(PROFILES)}) or breakpoints "
                         "like '20:0.01, 80:0.04, 500:0.04, 2000:0.01' in Hz:g2/Hz")
    f = np.array([float(a) for a, _ in pts])
    p = np.array([float(b) for _, b in pts])
    if np.any(np.diff(f) <= 0) or np.any(p <= 0):
        raise ValueError("PSD breakpoints need rising frequencies and positive levels")
    return PSD(f, p)


def random_grid(fn: np.ndarray, zeta: float, lo: float, hi: float, per_mode: int = 81, base: int = 400) -> np.ndarray:
    """Frequencies to integrate a random response on: a log spaced base plus fine points across every resonance
    in the band, about ten half power bandwidths wide each, so each peak is resolved to a fraction of a percent."""
    pts = [np.geomspace(lo, hi, base)]
    for f in fn:
        if lo * 0.5 <= f <= hi * 2:
            pts.append(f * (1 + zeta * np.linspace(-10, 10, per_mode)))
            pts.append(f * (1 + np.linspace(-0.5, 0.5, 41)))
    g = np.unique(np.concatenate(pts))
    return g[(g >= lo) & (g <= hi)]


@dataclass
class RandomResult:
    C0: np.ndarray        # (m, m) covariance of the modal coordinates, mm2 t
    C2: np.ndarray        # (m, m) the same weighted by f^2, for zero crossing rates
    grid: np.ndarray


def random_modal(modes: Modes, zeta: float, psd: PSD, direction) -> RandomResult:
    """Modal covariance of a base acceleration with this PSD along direction (motion relative to the base)."""
    f = random_grid(modes.freq, zeta, float(psd.f[0]), float(psd.f[-1]))
    H = transfer(f, modes.freq, zeta) * (-modes.gamma(direction))[None, :]   # per unit base acceleration
    W = psd(f) * G0 ** 2                                                    # (mm/s2)^2 / Hz
    HH = np.einsum("fi,fj->fij", H, H.conj()) * W[:, None, None]
    return RandomResult(_trapz(HH, f), _trapz(HH * (f ** 2)[:, None, None], f), f)


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    return np.tensordot(dx, 0.5 * (y[1:] + y[:-1]), axes=(0, 0))


def quadratic_rms(T: np.ndarray, C: np.ndarray) -> np.ndarray:
    """sqrt(sum_ij Re C_ij t_i . t_j) per node for per mode vectors T (m, n, k)."""
    m, n, k = T.shape
    flat = T.reshape(m, n * k)
    v = np.sum((C.real @ flat) * flat, axis=0).reshape(n, k).sum(1)
    return np.sqrt(np.clip(v, 0.0, None))


def vm_rms(modes: Modes, C: np.ndarray) -> np.ndarray:
    """Root mean square von Mises stress at every node (Segalman), MPa."""
    T = np.einsum("mnc,cd->mnd", modes.stress.astype(float), VM_L)
    return quadratic_rms(T, C)


def disp_rms(modes: Modes, C: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Root mean square displacement magnitude at some nodes, mm."""
    return quadratic_rms(modes.shapes[:, nodes].astype(float), C)


def zero_crossing_rate(modes: Modes, R: RandomResult, node: int) -> float:
    """Expected rate of upward zero crossings of the von Mises stress process at a node, sqrt(m2 / m0), Hz."""
    T = np.einsum("mc,cd->md", modes.stress[:, node].astype(float), VM_L)[:, None, :]
    m0 = quadratic_rms(T, R.C0)[0] ** 2
    m2 = quadratic_rms(T, R.C2)[0] ** 2
    return float(np.sqrt(m2 / m0)) if m0 > 0 else 0.0


def miles_rms(fn: float, zeta: float, psd_level: float) -> float:
    """Miles' equation, the acceleration response of a single mode to a flat PSD near it: sqrt(pi/2 f Q W), g."""
    return float(np.sqrt(np.pi / 2 * fn * psd_level / (2 * zeta)))


def single_mode_rms(gamma: float, fn: float, zeta: float, psd_level: float) -> float:
    """Root mean square modal coordinate of one mode under a flat PSD near its frequency, from the closed form
    integral of |H|^2: E[q^2] = Gamma^2 W / (8 zeta w^3)."""
    w = 2 * np.pi * fn
    return float(abs(gamma) * np.sqrt(psd_level * G0 ** 2 / (8 * zeta * w ** 3)))
