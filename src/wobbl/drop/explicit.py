"""Explicit time integration of a part striking a floor: central differences, lumped mass, penalty contact.

The part is linear elastic with small strains, which holds through an impact: it lasts a fraction of a
millisecond, too short for the part to turn far. The floor pushes on the surface nodes that go below it with a
force per node that follows the floor's law: elastic, crushing at a plateau with a permanent dent for foams,
and stiff once a soft layer is squeezed flat.

The step is 80% of the stable limit 2 / omega_max, with omega_max from power iteration on the mesh and the
floor springs, so the slivers a tet mesher leaves set it only as far as they really limit it. Every step logs
the contact force, the energy and the acceleration of the watched nodes; displacements are kept at about a
hundred frames for the stresses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Contact:
    """The floor as seen by the candidate nodes: a plane x . normal = level, below which it pushes back."""
    normal: np.ndarray            # unit, from the floor up into the part
    level: float
    nodes: np.ndarray             # candidate node indices
    k: np.ndarray                 # N/mm, elastic stiffness of each node's patch of floor
    cap: np.ndarray | None = None     # N, crush force of each patch (foam plateau) or None
    bottom: np.ndarray | None = None  # mm, depth at which a soft layer is flat on its base, or None
    k_hard: np.ndarray | None = None  # N/mm, stiffness once bottomed out


@dataclass
class History:
    t: np.ndarray
    force: np.ndarray             # N, total floor force along the normal
    accel: np.ndarray             # (s,3) mm/s2, mass weighted acceleration of the watched nodes
    com_v: np.ndarray             # (s,3) mm/s, velocity of the center of mass
    energy: np.ndarray            # N mm, kinetic + strain - work of the floor and gravity, minus the start
    frames_t: np.ndarray
    frames_u: list = field(default_factory=list)
    dt: float = 0.0
    steps: int = 0
    t_contact: tuple[float, float] = (0.0, 0.0)
    max_penetration: float = 0.0
    touched: np.ndarray | None = None       # candidate nodes that met the floor
    reason: str = ""
    t_run: float = 0.0
    kinetic0: float = 0.0


def omega_max(K: sp.csr_matrix, m3: np.ndarray, iters: int = 80, seed: int = 0) -> float:
    """Largest natural frequency (rad/s) of K u = omega^2 M u by power iteration, from below; deterministic."""
    d = 1.0 / np.sqrt(m3)
    x = np.random.default_rng(seed).standard_normal(len(m3))
    lam = 0.0
    for _ in range(iters):
        y = d * (K @ (d * x))
        lam_new = float(x @ y) / float(x @ x)
        x = y / np.linalg.norm(y)
        if abs(lam_new - lam) <= 1e-4 * abs(lam_new):
            lam = lam_new
            break
        lam = lam_new
    return float(np.sqrt(max(lam, 0.0)))


def integrate(K: sp.csr_matrix, mass: np.ndarray, X: np.ndarray, contact: Contact, v0: np.ndarray,
              gravity: np.ndarray, watch: np.ndarray, t_max: float, frames: int = 100, wmax: float | None = None,
              max_steps: int = 60_000, damping: float = 0.3, progress=None) -> History:
    """March from first touch until the part has left the floor and rung for a while, or t_max.

    damping is the damping ratio given to the mesh's highest frequency through stiffness proportional damping,
    beta = 2 damping / omega_max: it calms the ringing that a sudden contact sets off at the scale of the
    elements, and leaves the part's own modes, hundreds of times lower, practically undamped.
    """
    t0 = time.time()
    n = len(mass)
    m3 = np.repeat(mass, 3)
    w = wmax or omega_max(K, m3)
    nc = contact.nodes
    mc = mass[nc]
    w_spring = float(np.sqrt((contact.k / mc).max())) if len(nc) else 0.0
    if contact.k_hard is not None:
        w_spring = max(w_spring, float(np.sqrt((contact.k_hard / mc).max())))
    wt = np.sqrt(1.05 * w ** 2 + w_spring ** 2)
    beta = 2.0 * damping / w
    zeta = 0.5 * beta * wt
    dt = 0.8 * 2.0 / wt * (np.sqrt(1.0 + zeta ** 2) - zeta)
    nrm = contact.normal
    h0 = X[nc] @ nrm - contact.level                          # height of each candidate above the floor
    u = np.zeros(3 * n)
    v = np.tile(v0, n).astype(float)
    g3 = np.tile(gravity, n)
    set_ = np.zeros(len(nc))                                  # permanent dent of a crushing floor
    wm = mass[watch] / mass[watch].sum()
    mtot = mass.sum()
    ke0 = 0.5 * float(m3 @ (v * v))
    # start at the half step: v(-dt/2) from the free fall acceleration
    v = v - 0.5 * dt * g3
    Ku = np.zeros(3 * n)
    diss = 0.0
    log_t, log_f, log_a, log_v, log_e = [], [], [], [], []
    fr_t, fr_u = [0.0], [np.zeros((n, 3), np.float32)]
    stride, work, t = 1, 0.0, 0.0
    f_prev = m3 * g3
    first, last, quiet, steps = None, None, 0, 0
    pen_max, touched = 0.0, np.zeros(len(nc), bool)
    reason = "time limit"
    while steps < max_steps:
        du = dt * v
        u += du
        t += dt
        steps += 1
        uc = u.reshape(n, 3)[nc] @ nrm
        p = -(h0 + uc)                                        # penetration, positive below the floor
        f = np.zeros(len(nc))
        act = p > set_
        if act.any():
            f[act] = contact.k[act] * (p[act] - set_[act])
            if contact.cap is not None:
                over = act & (f > contact.cap)
                set_[over] = p[over] - contact.cap[over] / contact.k[over]
                f[over] = contact.cap[over]
            if contact.bottom is not None:
                flat = p > contact.bottom
                f[flat] += contact.k_hard[flat] * (p[flat] - contact.bottom[flat])
            touched |= act
            pen_max = max(pen_max, float(p.max()))
        fc = np.zeros(3 * n)
        fc.reshape(n, 3)[nc] = f[:, None] * nrm[None, :]
        # one product a step: K v gives the damping force, and K u follows by K u += dt K v
        Kv = K @ v
        Ku += dt * Kv
        fint = Ku + beta * Kv
        diss += beta * dt * float(v @ Kv)
        a = (fc - fint) / m3 + g3
        # work of the floor and gravity over this step, trapezoidal; energy is taken at the whole step
        f_now = fc + m3 * g3
        work += 0.5 * float((f_prev + f_now) @ du)
        f_prev = f_now
        vn = v + 0.5 * dt * a
        v += dt * a
        ftot = float(f.sum())
        if ftot > 0:
            first = t if first is None else first
            last, quiet = t, 0
        elif first is not None:
            quiet += 1
        av = a.reshape(n, 3)[watch]
        log_t.append(t)
        log_f.append(ftot)
        log_a.append(wm @ av)
        log_v.append((mass @ vn.reshape(n, 3)) / mtot)
        ke = 0.5 * float(m3 @ (vn * vn))
        log_e.append(ke + 0.5 * float(u @ Ku) + diss - work - ke0)
        if steps % stride == 0:
            fr_t.append(t)
            fr_u.append(u.reshape(n, 3).astype(np.float32))
            if len(fr_t) > 2 * frames:
                fr_t, fr_u = fr_t[::2], fr_u[::2]
                stride *= 2
        if first is not None and quiet > 0:
            pulse = last - first
            # the part has left the floor: let the stress waves ring for half a pulse more, then stop
            if (t - last) >= max(0.5 * pulse, 20 * dt) and not (p > set_).any():
                reason = "the part left the floor"
                break
        if ke > 50 * ke0 + 1e-12:
            reason = "unstable"
            break
        if t >= t_max:
            break
        if progress and steps % 2000 == 0:
            progress(t, ftot)
    h = History(np.asarray(log_t), np.asarray(log_f), np.asarray(log_a), np.asarray(log_v), np.asarray(log_e),
                np.asarray(fr_t), fr_u, dt, steps, (first or 0.0, last or 0.0), pen_max, nc[touched], reason,
                time.time() - t0, ke0)
    return h


__all__ = ["Contact", "History", "integrate", "omega_max"]
