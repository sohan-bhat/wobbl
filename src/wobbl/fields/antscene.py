"""The 3D scene of an antenna run: the real conductors the solver meshed, the feed, the electric field on a slice
through the feed playing over one radio frequency cycle, and the radiation pattern as a balloon around the antenna.

Everything drawn comes from the run. The conductor surfaces are the triangles the solver used, the slice is the
field it solved (a phasor, played as Re(E e^(j w t))), and the balloon is its far field gain. Only the balloon's
size is chosen for the eye, and the viewer says so.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..viz import Scene as VizScene
from .forms import C0

FRAMES = 32           # frames over one cycle of the radio frequency


def read_triangles(path: str | Path, tags: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) and triangles (m, 3) of the surface elements in a gmsh 2.2 ASCII mesh whose physical group is
    one of tags, with the vertices renumbered from 0."""
    lines = Path(path).read_text().splitlines()
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    ids = np.empty(n, dtype=np.int64)
    xyz = np.empty((n, 3))
    for j in range(n):
        t = lines[i + 2 + j].split()
        ids[j] = int(t[0])
        xyz[j] = (float(t[1]), float(t[2]), float(t[3]))
    k = lines.index("$Elements")
    m = int(lines[k + 1])
    tris = []
    for j in range(m):
        t = lines[k + 2 + j].split()
        if t[1] == "2" and int(t[3]) in tags:
            nt = int(t[2])
            tris.append([int(x) for x in t[3 + nt:6 + nt]])
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    tris = np.array(tris)
    lut = {int(v): j for j, v in enumerate(ids)}
    used = sorted(set(tris.ravel().tolist()))
    remap = {v: j for j, v in enumerate(used)}
    return xyz[[lut[v] for v in used]], np.vectorize(remap.get)(tris)


def balloon(d: np.ndarray, th: np.ndarray, ph: np.ndarray, eff: float, radius: float, center) -> tuple:
    """The pattern as a surface: the radius grows with the field strength (the square root of the gain) so a
    dipole's doughnut and a patch's lobe read at a glance. Returns vertices, triangles and gain in dBi per vertex."""
    g = np.maximum(d * eff, 1e-6)
    ph = np.append(ph, ph[0] + 2 * math.pi)
    g = np.concatenate([g, g[:, :1]], axis=1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    r = radius * np.sqrt(g / g.max())
    X = np.c_[(r * np.sin(T) * np.cos(P)).ravel(), (r * np.sin(T) * np.sin(P)).ravel(), (r * np.cos(T)).ravel()]
    X = X + np.asarray(center, float)
    nt, npp = g.shape
    idx = np.arange(nt * npp).reshape(nt, npp)
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    return X, np.concatenate([np.c_[a, b, c], np.c_[a, c, e]]), 10 * np.log10(g.ravel())


def write(out: Path, *, title: str, question: str, scene, setup, feed: dict, msh, plane: dict, base: dict, gain_db: float,
          eff: float, verdict: tuple[str, str, str], f0: float, dims_note: str = "") -> Path:
    """Build and save scene.wobbl for one antenna run. base is the working frequency's solve (with its slice and
    its pattern on a full sphere)."""
    mm = 1e3
    s = VizScene(title, domain="antenna", question=question, length="mm",
                 source="The conductors are the surfaces the solver meshed. The slice is the solved electric field "
                        f"at {f0 / 1e9:.4g} GHz played over one cycle. The balloon is the far field gain, drawn with "
                        "its radius growing with the field strength and sized to read, not to scale.")
    s.timeline(np.arange(FRAMES) / FRAMES, unit="cycle", label="phase", seconds=5, between="interpolated")
    # conductors, dielectric and the feed
    metal = {setup_name for setup_name in list(setup.metals) + list(setup.sheets)}
    tags = {msh.tags[n] for n in metal if n in msh.tags}
    X, T = read_triangles(msh.path, tags)
    if len(T):
        s.mesh(X * mm, T, name="conductors", color="#d9a441", opacity=1.0, max_vertices=9000, smooth=False)
    for n, v in setup.dielectrics.items():
        b = next((b for b in scene.bodies if b.name == n), None)
        if b is not None:
            lo, hi = np.array(b.bbox[:3]) * mm, np.array(b.bbox[3:]) * mm
            s.body("box", list((hi - lo) / 2), pos=list((hi + lo) / 2), color="#2f7d57", opacity=0.35,
                   name=f"{n} (er {v[0]:g})")
    a, b_ = np.asarray(feed["a"]) * mm, np.asarray(feed["b"]) * mm
    mid = 0.5 * (a + b_)
    size = max(float(np.ptp(np.vstack([X * mm, a, b_]), axis=0).max()) if len(X) else 10.0, 1.0)
    s.arrows([mid - (b_ - a) * 0.0], [(b_ - a) * 1.0], color="#ff4d4f", radius=max(size * 0.006, 0.05),
             name="feed port", labels=["feed, 1 A"], prop=False)
    # the field on the slice, as a phasor played over one cycle
    sl = base.get("slice")
    if sl is not None and np.isfinite(sl).any():
        iu, iv, ik = plane["iu"], plane["iv"], plane["ik"]
        nu, nv = plane["nu"], plane["nv"]
        e = sl[..., :3] + 1j * sl[..., 3:]
        mag = np.abs(e)
        top = float(np.nanpercentile(np.sqrt(np.sum(mag ** 2, axis=-1)), 97)) or 1.0
        ph = np.arange(FRAMES) / FRAMES * 2 * math.pi
        # the field component with the most energy keeps its sign, so the wave reads as a wave
        energy = np.nansum(mag ** 2, axis=(0, 1))
        c = int(np.argmax(energy))
        sig = np.real(e[..., c][None] * np.exp(1j * ph)[:, None, None]) / top
        sig = np.clip(sig, -1.0, 1.0)
        uu = np.linspace(plane["u0"], plane["u1"], nu) * mm
        vv = np.linspace(plane["v0"], plane["v1"], nv) * mm
        gu, gv = np.meshgrid(uu, vv)
        P = np.zeros((nu * nv, 3))
        P[:, iu], P[:, iv], P[:, ik] = gu.ravel(), gv.ravel(), plane["at"] * mm
        ok = np.isfinite(sig[0]).ravel()
        idx = np.arange(nu * nv).reshape(nv, nu)
        q = [idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
        keep = ok[q[0]] & ok[q[1]] & ok[q[2]] & ok[q[3]]
        tri = np.concatenate([np.c_[q[0], q[1], q[2]][keep], np.c_[q[0], q[2], q[3]][keep]])
        if len(tri):
            vals = np.nan_to_num(sig.reshape(FRAMES, -1))
            m = s.mesh(P, tri, name="electric field", opacity=0.82, max_vertices=nu * nv + 1, smooth=True)
            m.color_by(f"E{'xyz'[c]}", vals, unit="of the 97th percentile", colormap="coolwarm", range=[-1, 1], digits=2)
    # the radiation pattern
    d = base.get("d_sphere")
    if d is not None:
        pts = np.array([a, b_, *(X * mm if len(X) else [])])
        center = 0.5 * (pts.min(0) + pts.max(0))
        rad = 0.55 * float(np.linalg.norm(pts.max(0) - pts.min(0))) + 0.15 * C0 / f0 * mm
        Xb, Tb, gdb = balloon(d, base["sphere_theta"], base["sphere_phi"], eff, rad, center)
        m = s.mesh(Xb, Tb, name="radiation pattern", opacity=0.5, max_vertices=len(Xb) + 1, smooth=True)
        m.color_by("gain", np.tile(gdb, (FRAMES, 1)), unit="dBi", colormap="turbo",
                   range=[max(gdb.max() - 30, -20), gdb.max()], digits=3)
        s.label(f"peak gain {gain_db:.1f} dBi", Xb[int(np.argmax(gdb))].tolist(), kind="note")
    s.readout("gain", np.full(FRAMES, gain_db), unit="dBi", digits=3)
    s.readout("efficiency", np.full(FRAMES, 100 * eff), unit="%", digits=3)
    word, line, tone = verdict
    s.verdict(word, line, tone=tone)
    if dims_note:
        s.note(dims_note)
    s.camera(target=center.tolist() if d is not None else mid.tolist(), direction=(0.9, -1.0, 0.8), distance=size * 2.6)
    return s.save(out)
