"""Sound: room and enclosure modes, muffler transmission loss and the sound a vibrating panel makes.

Room modes come from an eigenvalue solve of the Helmholtz equation in the air
volume with rigid walls (GetDP with SLEPc), checked against the exact modes of
a box, Weyl's mode count and a finer mesh. A muffler is an axisymmetric duct
driven by a plane wave with anechoic ends, whose transmission loss Wobbl checks
with an energy balance (incident power equals reflected plus transmitted) and
against plane wave theory below the chamber's first cross mode. A panel is a
baffled piston, checked against the exact on axis pressure and radiation
resistance of a circular piston.
"""
from __future__ import annotations

import itertools
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from . import plots, solve
from .forms import SoundProblem, helmholtz
from .mesh2d import Model2D, Shape, mesh
from .mesh3d import Model3D
from .mesh3d import mesh as mesh3
from .params import ParamError, check_known, text, value

C_AIR = 343.2       # m/s at 20 C
RHO_AIR = 1.204     # kg/m3 at 20 C
P_REF = 20e-6       # Pa
ROOM_KEYS = {"size", "fmax", "t60", "modes", "step"}
MUFFLER_KEYS = {"pipe_d", "chamber_d", "chamber_length", "pipe_length", "fmin", "fmax", "points", "target", "band"}
PANEL_KEYS = {"diameter", "velocity", "acceleration", "displacement", "fmin", "fmax", "points", "distance", "limit"}


def _size3(s: str, unit: float) -> tuple[float, float, float]:
    parts = [p for p in s.lower().replace("mm", "").replace("m", "").replace("*", "x").split("x") if p.strip()]
    if len(parts) != 3:
        raise ParamError(f"size takes three numbers like 5x4x3, got '{s}'")
    return tuple(float(p) * unit for p in parts)


def box_modes(L: tuple, fmax: float, c: float = C_AIR) -> list[tuple[float, tuple[int, int, int]]]:
    """Exact rigid wall modes of a rectangular room up to fmax: f = c/2 sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)."""
    n = [int(2 * fmax * li / c) + 1 for li in L]
    out = []
    for idx in itertools.product(*(range(k + 1) for k in n)):
        if not any(idx):
            continue
        f = c / 2 * math.sqrt(sum((i / li) ** 2 for i, li in zip(idx, L)))
        if f <= fmax:
            out.append((f, idx))
    return sorted(out)


def weyl(V: float, S: float, Ltot: float, f: float, c: float = C_AIR) -> float:
    """Expected number of modes below f in a room of volume V, surface S and total edge length Ltot."""
    return 4 * math.pi * V * f ** 3 / (3 * c ** 3) + math.pi * S * f ** 2 / (4 * c ** 2) + Ltot * f / (8 * c)


def _kind(idx) -> str:
    nz = sum(1 for i in idx if i)
    return {1: "axial", 2: "tangential", 3: "oblique"}[nz]


def bonello(freqs: list[float], f_hi: float) -> tuple[bool, list[int]]:
    """Bonello's criterion: mode counts per third octave band never fall as frequency rises."""
    bands = []
    fc = 20.0
    while fc < f_hi:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        bands.append(sum(1 for f in freqs if lo <= f < hi))
        fc *= 2 ** (1 / 3)
    ok = all(b2 >= b1 for b1, b2 in zip(bands, bands[1:]) if b1)
    return ok, bands


def run_sound(template: str, params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
              log=lambda stage, detail="": None) -> GenericOutcome:
    if template in ("room", "enclosure"):
        return run_room(template, params, out_dir, question, keep, log)
    if template == "muffler":
        return run_muffler(params, out_dir, question, keep, log)
    if template == "panel":
        return run_panel(params, out_dir, question, keep, log)
    raise ParamError(f"unknown sound template '{template}' (use room, enclosure, muffler or panel)")


# --- room and enclosure modes ------------------------------------------------------------
def _eigen(model: Model3D, fmax: float, nev: int, work: Path, tag: str, plane=None, shapes: int = 0):
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    m = mesh3(model, d / "m.msh")
    sp = SoundProblem("3d", modes=nev, shift=1.0, fields=False, plane=plane, shapes=shapes)
    log = solve.run(helmholtz(sp, {"air": 1}), m.path, d, resolution="H", verbosity=4)
    w2 = [z.real for z in solve.eigenvalues(log)]
    f = sorted(math.sqrt(max(v, 0.0)) / (2 * math.pi) for v in w2)
    return m, [x for x in f if x > 0.5], d


def run_room(template: str, params: dict, out_dir, question="", keep=False, log=lambda s, d="": None):
    t0 = time.time()
    p = dict(params)
    check_known(p, ROOM_KEYS, template)
    small = template == "enclosure"
    unit = 1e-3 if small else 1.0
    L = _size3(str(p.get("size", "300x200x150" if small else "5x4x2.7")), unit)
    fmax = value(p, "fmax", 1500 if small else 120, "frequency")
    t60 = value(p, "t60", 0.5)
    V, S = L[0] * L[1] * L[2], 2 * (L[0] * L[1] + L[1] * L[2] + L[0] * L[2])
    Ltot = 4 * sum(L)
    n_exp = weyl(V, S, Ltot, fmax)
    nev = int(value(p, "modes", min(max(int(1.4 * n_exp) + 6, 8), 120)))
    lam = C_AIR / fmax
    base = Model3D(boxes=[(0, 0, 0, *L)], step=str(p.get("step", "")), size=lam / 5)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    zh = min(1.2, 0.45 * L[2]) if not small else 0.5 * L[2]
    plane = (zh, 0.0, L[0], 0.0, L[1], 41, int(41 * L[1] / L[0]) + 1)
    log("solve", f"modes up to {fmt(fmax)} Hz")
    m1, f1, d1 = _eigen(base, fmax, nev, work, "coarse")
    log("verify", "the same modes on a finer mesh")
    m2, f2, d2 = _eigen(base.scaled(0.7), fmax, nev, work, "fine", plane, shapes=6)
    f2 = [f for f in f2 if f <= fmax * 1.02]
    n_cmp = min(len(f1), len(f2))
    top = f2[n_cmp - 1] if n_cmp else 0.0
    checks = [convergence_check([f1[n_cmp - 1], top] if n_cmp else [0.0], 0.005,
                                "highest mode frequency", "Hz", stopped_by="two meshes", min_levels=2)]
    exact = box_modes(L, fmax * 1.05) if not p.get("step") else []
    if exact:
        k = min(len(exact), len(f2))
        errs = [abs(f2[i] - exact[i][0]) / exact[i][0] for i in range(k)]
        worst = max(errs) if errs else 0.0
        checks.append(Check("closed form", "pass" if worst < 0.005 else ("warn" if worst < 0.02 else "fail"),
                            f"the first {k} modes match the exact box modes to {100 * worst:.2g}% or better "
                            f"(lowest {fmt(f2[0])} Hz against {fmt(exact[0][0])} Hz)", {"worst": worst}))
    n_found = len([f for f in f2 if f <= fmax])
    dn = abs(n_found - n_exp) / max(n_exp, 1)
    checks.append(Check("mode count", "pass" if dn < 0.25 or abs(n_found - n_exp) <= 3 else "warn",
                        f"{n_found} modes below {fmt(fmax)} Hz, Weyl's formula for this volume expects about "
                        f"{n_exp:.0f}"))
    fs = 2000 * math.sqrt(t60 / V) if not small else 0.0
    modes = [f for f in f2 if f <= fmax]
    labels = [_kind(e[1]) for e in exact[:len(modes)]] if exact else ["mode"] * len(modes)
    close = [(a, b) for a, b in zip(modes, modes[1:]) if (b - a) / a < 0.02 and a < (fs or fmax)]
    ok_b, bands = bonello(modes, min(fs or fmax, fmax))
    lvl, reasons = confidence(checks)
    if small:
        verdict = "ok"
        head = (f"The {fmt(L[0] * 1e3)} x {fmt(L[1] * 1e3)} x {fmt(L[2] * 1e3)} mm enclosure has {len(modes)} standing "
                f"wave modes below {fmt(fmax)} Hz, the first at {fmt(modes[0])} Hz.")
    else:
        verdict = "ok" if ok_b and not close else "marginal"
        head = (f"The {fmt(L[0])} x {fmt(L[1])} x {fmt(L[2])} m room has {len(modes)} modes below {fmt(fmax)} Hz, the "
                f"first at {fmt(modes[0])} Hz, and modes overlap into a smooth response above about {fmt(fs)} Hz "
                f"(Schroeder, for a {fmt(t60)} s reverberation time).")
        if close:
            head += f" {len(close)} pair{'s' if len(close) > 1 else ''} of modes sit within 2% of each other " \
                    f"(the first near {fmt(close[0][0])} Hz), which makes boomy notes."
        if not ok_b:
            head += " The mode count per third octave falls somewhere below the Schroeder frequency."
    fixes = []
    if verdict != "ok":
        best = _better_ratio(L, fmax, fs)
        if best:
            fixes.append({"kind": "dimensions", "title": f"Change the room to about {fmt(best[0])} x {fmt(best[1])} x "
                                                         f"{fmt(best[2])} m",
                          "summary": "These dimensions keep the volume within 15% and spread the low modes evenly "
                                     "by Bonello's criterion, with no two within 2%.", "auto": False})
        fixes.append({"kind": "treatment", "title": f"Add bass traps tuned near {fmt(close[0][0] if close else modes[0])} Hz",
                      "summary": "Corner traps damp the strongest axial and tangential modes, since every mode has "
                                 "a pressure peak in the corners.", "auto": False})
    images = {}
    shp = []
    for i in range(6):
        fp = d2 / f"mode{i}.txt"
        if fp.exists():
            shp.append(np.loadtxt(fp))
    if shp:
        images["modes"] = _mode_panel(out / "images" / "modes.png", shp, f2, labels, L, zh, unit)
    images["spectrum"] = _spectrum(out / "images" / "spectrum.png", modes, exact, fs)
    values = {"modes_Hz": modes, "exact_Hz": [e[0] for e in exact][:len(modes)], "schroeder_Hz": fs,
              "bonello_ok": ok_b, "bands": bands, "close_pairs": close, "nodes": [m1.n_nodes, m2.n_nodes]}
    metrics = [("lowest mode", f"{fmt(modes[0])} Hz"), ("modes", f"{len(modes)} below {fmt(fmax)} Hz"),
               ("Schroeder", f"{fmt(fs)} Hz" if fs else "not used"), ("close pairs", str(len(close)))]
    assumptions = ["Rigid walls (no absorption), air at 20 C with sound speed 343 m/s.",
                   "Modes are the room's resonances without any source, and real rooms damp them by their absorption."]
    if p.get("step"):
        assumptions.append("The air volume comes from the STEP file, read in millimeters.")
    o = GenericOutcome("sound", template, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["spectrum", "Mode frequencies"], ["modes", "The first mode shapes"]]},
                       images, fixes, solver=f"GetDP {solve.getdp().version()} with SLEPc and gmsh",
                       t_total=time.time() - t0)
    _room_playback(out, shp, f2, labels, L, zh, unit, fs, exact)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _better_ratio(L, fmax, fs):
    """Search box proportions near the current volume for an even low mode spread (closed form modes)."""
    V = L[0] * L[1] * L[2]
    h = L[2]
    best, score = None, None
    for rx in np.linspace(1.1, 2.6, 31):
        for ry in np.linspace(1.0, 2.0, 21):
            if ry >= rx:
                continue
            k = (V / (rx * ry)) ** (1 / 3)
            dims = (rx * k, ry * k, k)
            if abs(dims[2] - h) / h > 0.15:
                continue
            fr = [f for f, _ in box_modes(dims, min(fs or fmax, fmax))]
            ok, bands = bonello(fr, min(fs or fmax, fmax))
            gaps = np.diff(fr) if len(fr) > 1 else np.array([0.0])
            close = int(np.sum(gaps / np.array(fr[:-1]) < 0.02)) if len(fr) > 1 else 0
            sc = (not ok, close, float(np.std(gaps)))
            if score is None or sc < score:
                best, score = dims, sc
    return best


def _mode_panel(path, shapes, freqs, labels, L, z, unit):
    plt = plots._mpl()
    from ..report.render import BG, FG

    n = len(shapes)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4), dpi=110, facecolor=BG)
    for i, ax in enumerate(axes.ravel()):
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        a = shapes[i]
        x, y, v = a[:, 0], a[:, 1], a[:, 3]
        v = v / (np.max(np.abs(v)) or 1)
        ax.tricontourf(x / unit, y / unit, v, levels=21, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(f"{fmt(freqs[i])} Hz {labels[i] if i < len(labels) else ''}", color=FG, fontsize=9)
    fig.suptitle(f"pressure of the first modes on a plane {fmt(z / unit)} {'mm' if unit < 1 else 'm'} up "
                 "(red and blue are opposite phase)", color=FG, fontsize=10)
    fig.tight_layout()
    return plots.save(fig, path)


def _spectrum(path, modes, exact, fs):
    plt = plots._mpl()
    from ..report.render import BG, FG, MARK

    fig, ax = plots._axes(plt, (8, 3.2))
    ax.vlines(modes, 0, 1, color=MARK, lw=1.6, label="field solve")
    if exact:
        ax.plot([e[0] for e in exact[:len(modes)]], [1.05] * min(len(exact), len(modes)), "v", color="#4a90d9", ms=4,
                label="exact box modes")
    if fs:
        ax.axvline(fs, color="#ff4d4f", ls="--", lw=1.2, label="Schroeder frequency")
    ax.set_yticks([])
    ax.set_xlabel("frequency, Hz", color=FG)
    ax.legend(facecolor=BG, edgecolor="#3a3d42", labelcolor=FG, fontsize=8, loc="lower right")
    fig.tight_layout()
    return plots.save(fig, path)


# --- muffler ---------------------------------------------------------------------------------
def muffler_tl(m: float, k: np.ndarray, L: float) -> np.ndarray:
    """Plane wave transmission loss of a simple expansion chamber of area ratio m and length L."""
    return 10 * np.log10(1 + 0.25 * (m - 1 / m) ** 2 * np.sin(k * L) ** 2)


def run_muffler(params: dict, out_dir, question="", keep=False, log=lambda s, d="": None):
    t0 = time.time()
    p = dict(params)
    check_known(p, MUFFLER_KEYS, "muffler")
    a = value(p, "pipe_d", 50, "length") / 2
    R = value(p, "chamber_d", 150, "length") / 2
    Lc = value(p, "chamber_length", 300, "length")
    Lp = value(p, "pipe_length", 4 * a * 1e3, "length")
    if R <= a:
        raise ParamError("the chamber must be wider than the pipe")
    fc = 3.8317 * C_AIR / (2 * math.pi * R)          # first axisymmetric cross mode of the chamber
    fmin = value(p, "fmin", 20, "frequency")
    fmax = value(p, "fmax", min(1.3 * fc, 3.8317 * C_AIR / (2 * math.pi * a) * 0.8), "frequency")
    n = int(value(p, "points", 60))
    freqs = np.linspace(fmin, fmax, n)
    lam = C_AIR / fmax
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    x1, x2, x3 = Lp, Lp + Lc, 2 * Lp + Lc
    pts = ((0, 0), (a, 0), (a, x1), (R, x1), (R, x2), (a, x2), (a, x3), (0, x3))
    span = x3 / 2 + R

    def model(k):
        h = min(lam / 8, a / 2)
        return Model2D("axi", [Shape("duct", "poly", pts)], 0.0, size=h, far=h,
                       lines={"inlet": "y=0", "outlet": f"y={x3!r}", "axis": "x=0"}).scaled(k)

    msh = mesh(model(1.0), work / "m.msh")
    rows = []
    for i, f in enumerate(freqs):
        if i % 10 == 0:
            log("solve", f"{fmt(f)} Hz ({i + 1} of {n})")
        rows.append(_duct_solve(msh, f, work, f"f{i}"))
    tl = np.array([r["tl"] for r in rows])
    bal = np.array([r["balance"] for r in rows])
    k = 2 * math.pi * freqs / C_AIR
    m = (R / a) ** 2
    tl_pw = muffler_tl(m, k, Lc)
    # the first lobe, below the first pass band at c / 2L: its peak and where it ends are what plane wave theory
    # gets right; end corrections shift the curve a little, which a point by point comparison would punish
    f_pass = C_AIR / (2 * Lc)
    lobe = freqs < f_pass
    i_pk = int(np.argmax(np.where(lobe, tl, -1)))
    tl_peak_pw = 10 * math.log10(1 + 0.25 * (m - 1 / m) ** 2)
    near = (freqs > 0.7 * f_pass) & (freqs < 1.3 * f_pass)
    f_zero = float(freqs[near][np.argmin(tl[near])]) if near.any() else 0.0
    dtl = abs(float(tl[i_pk]) - tl_peak_pw)
    log("verify", "a finer mesh at the first peak")
    fine = _duct_solve(mesh(model(0.7), work / "fine.msh"), float(freqs[i_pk]), work, "fine")
    checks = [convergence_check([rows[i_pk]["tl"], fine["tl"]], 0.01, "transmission loss at the first peak", "dB",
                                stopped_by="two meshes", min_levels=2)]
    wb = float(np.max(bal))
    checks.append(Check("energy balance", "pass" if wb < 0.01 else ("warn" if wb < 0.03 else "fail"),
                        f"incident power equals reflected plus transmitted power to {100 * wb:.2g}% at every frequency"))
    dz = abs(f_zero - f_pass) / f_pass if f_zero else 1.0
    st = "pass" if dtl < 1.0 and dz < 0.08 else ("warn" if dtl < 2.0 and dz < 0.15 else "fail")
    checks.append(Check("closed form", st,
                        f"plane wave theory puts the first peak at {tl_peak_pw:.1f} dB and the first pass band at "
                        f"{fmt(f_pass)} Hz, the field solve at {float(tl[i_pk]):.1f} dB and {fmt(f_zero)} Hz "
                        f"(the sampled curve, {fmt(float(freqs[1] - freqs[0]))} Hz apart)",
                        {"peak_dB": float(tl[i_pk]), "peak_theory_dB": tl_peak_pw, "pass_band_Hz": f_zero}))
    lvl, reasons = confidence(checks)
    tl_max = 20 * math.log10((m + 1 / m) / 2)
    target = value(p, "target", 0, "db") if "target" in p else 0.0
    band = [float(x) for x in str(p.get("band", "")).split("-") if x.strip()] if p.get("band") else []
    if target and len(band) == 2:
        sel = (freqs >= band[0]) & (freqs <= band[1])
        worst = float(tl[sel].min()) if sel.any() else 0.0
        verdict = "ok" if worst >= target else ("marginal" if worst >= target - 3 else "fails")
        head = (f"The muffler gives at least {worst:.1f} dB from {fmt(band[0])} to {fmt(band[1])} Hz against the "
                f"{target:g} dB needed. It peaks at {float(tl.max()):.1f} dB and passes sound with no loss where the "
                "chamber length is a whole number of half wavelengths.")
    else:
        verdict = "ok"
        head = (f"The {fmt(Lc * 1e3)} mm expansion chamber (area ratio {m:.3g}) peaks at {float(tl.max()):.1f} dB "
                f"transmission loss near {fmt(float(freqs[int(np.argmax(tl))]))} Hz and drops to zero near "
                f"{fmt(C_AIR / (2 * Lc))} Hz, where the chamber is half a wavelength long.")
    fixes = []
    if verdict != "ok" and len(band) == 2:
        fmid = math.sqrt(band[0] * band[1])
        fixes.append({"kind": "length", "title": f"Make the chamber about {fmt(C_AIR / (4 * fmid) * 1e3)} mm long",
                      "summary": f"An expansion chamber peaks where it is a quarter wavelength long, so this centers the "
                                 f"first peak on {fmt(fmid)} Hz.", "auto": False})
        fixes.append({"kind": "ratio", "title": "Widen the chamber",
                      "summary": f"The peak loss is 20 log((m + 1/m) / 2), {tl_max:.1f} dB now, and grows by about "
                                 "6 dB each time the area ratio doubles.", "auto": False})
    images = {"tl": plots.curve(out / "images" / "tl.png", freqs, {"field solve": tl, "plane wave theory": tl_pw},
                                "frequency, Hz", "transmission loss, dB", hline=target or None,
                                hlabel=f"needed {target:g} dB" if target else "")}
    values = {"freqs_Hz": list(freqs), "tl_dB": list(tl), "tl_plane_wave_dB": list(tl_pw), "area_ratio": m,
              "cross_mode_Hz": fc, "max_balance_error": wb, "nodes": msh.n_nodes}
    metrics = [("peak loss", f"{float(tl.max()):.1f} dB"), ("area ratio", f"{m:.3g}"),
               ("first cross mode", f"{fmt(fc)} Hz"), ("pass band", f"{fmt(C_AIR / (2 * Lc))} Hz")]
    assumptions = ["Axisymmetric duct with rigid walls, no flow and no damping, air at 20 C.",
                   "A plane wave of 1 Pa comes in through the inlet pipe and both pipe ends are anechoic, which is how "
                   "transmission loss is defined (the tailpipe's own reflection is left out).",
                   f"Above {fmt(fc)} Hz the chamber carries cross modes, which plane wave theory does not see."]
    o = GenericOutcome("sound", "muffler", question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["tl", "Transmission loss"]]}, images, fixes,
                       solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t0)
    shots = [_duct_solve(msh, float(f), work, f"shot{j}", fields=True) | {"f": float(f)}
             for j, f in enumerate((freqs[i_pk], f_zero or f_pass))]
    _muffler_playback(out, msh, shots, (a, R, x3), freqs, tl, tl_pw)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _duct_solve(msh, f: float, work: Path, tag: str, fields: bool = False) -> dict:
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = SoundProblem("axi", freq=f, inlet="inlet", outlets=["outlet"], fields=fields)
    tags = {k: v for k, v in msh.tags.items() if k != "air" and k != "shell"}
    solve.run(helmholtz(sp, tags), msh.path, d, resolution="H")
    s_in = solve.read_global(d, "S_inlet.txt")[0]
    p_in = solve.read_complex(d, "P1_inlet.txt") / s_in
    p2_out = solve.read_global(d, "P2_outlet.txt")[0]
    w_inc = s_in                                     # |A|^2 S with A = 1 Pa, all over 2 rho c
    w_ref = abs(p_in - 1.0) ** 2 * s_in
    w_tr = p2_out
    out = {"tl": 10 * math.log10(w_inc / max(w_tr, 1e-30)), "balance": abs(w_inc - w_ref - w_tr) / w_inc}
    if fields:
        pc = solve.read_nodes(d, "p.txt", msh.node_ids, comps=2)
        out["p"] = pc[:, 0] + 1j * pc[:, 1]
    return out


# --- vibrating panel -----------------------------------------------------------------------
def piston_axis(a: float, z: float, k: float, v: float) -> float:
    """Exact pressure amplitude on the axis of a baffled circular piston, Pa."""
    return 2 * RHO_AIR * C_AIR * v * abs(math.sin(k / 2 * (math.sqrt(z * z + a * a) - z)))


def piston_power(a: float, k: float, v: float) -> float:
    """Exact radiated power of a baffled circular piston of velocity amplitude v, W."""
    from scipy.special import j1

    x = 2 * k * a
    return 0.5 * RHO_AIR * C_AIR * math.pi * a * a * v * v * (1 - 2 * j1(x) / x)


def run_panel(params: dict, out_dir, question="", keep=False, log=lambda s, d="": None):
    t0 = time.time()
    p = dict(params)
    check_known(p, PANEL_KEYS, "panel")
    a = value(p, "diameter", 200, "length") / 2
    fmin = value(p, "fmin", 100, "frequency")
    fmax = value(p, "fmax", 2000, "frequency")
    n = int(value(p, "points", 12))
    freqs = np.geomspace(fmin, fmax, n)
    dist = value(p, "distance", 1, "distance")
    if "velocity" in p:
        vel = np.full(n, value(p, "velocity", 0.01))
        src = f"{fmt(vel[0] * 1e3)} mm/s"
    elif "acceleration" in p:
        acc = value(p, "acceleration", 1)
        vel = acc / (2 * math.pi * freqs)
        src = f"{fmt(acc)} m/s2"
    else:
        disp = value(p, "displacement", 0.01, "length")
        vel = disp * 2 * math.pi * freqs
        src = f"{fmt(disp * 1e6)} um"
    lam_min, lam_max = C_AIR / fmax, C_AIR / fmin
    R = max(1.3 * dist, 3 * a, 0.6 * lam_max)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    def model(k):
        h = min(lam_min / 8, a / 4)
        # a patch of air over the piston puts a vertex at its rim, so the piston and the baffle are separate edges
        rim = Shape("air", "rect", (0.0, 0.0, a, a / 2))
        return Model2D("axi", [rim], R, clip="x>=0,y>=0", size=h, far=min(lam_min / 6, R / 8), growth=0.3,
                       lines={"piston": f"y=0&x<{a!r}", "axis": "x=0", "baffle": "y=0", "outer": "outer"},
                       spots=[(a, 0.0, h / 2)]).scaled(k)

    msh = mesh(model(1.0), work / "m.msh")
    rows = []
    for i, f in enumerate(freqs):
        log("solve", f"{fmt(f)} Hz ({i + 1} of {n})")
        rows.append(_panel_solve(msh, f, float(vel[i]), R, (1e-6 * a, dist), work, f"f{i}"))
    k = 2 * math.pi * freqs / C_AIR
    p_fem = np.array([r["p"] for r in rows])
    p_ex = np.array([piston_axis(a, dist, kk, vv) for kk, vv in zip(k, vel)])
    spl = 20 * np.log10(np.maximum(p_fem / math.sqrt(2), 1e-12) / P_REF)
    spl_ex = 20 * np.log10(np.maximum(p_ex / math.sqrt(2), 1e-12) / P_REF)
    w_fem = np.array([r["w"] for r in rows])
    w_ex = np.array([piston_power(a, kk, vv) for kk, vv in zip(k, vel)])
    d_spl = float(np.max(np.abs(spl - spl_ex)))
    d_w = float(np.max(np.abs(w_fem - w_ex) / w_ex))
    log("verify", "a finer mesh at the top frequency")
    fine_m = mesh(model(0.7), work / "fine.msh")
    fine = _panel_solve(fine_m, float(freqs[-1]), float(vel[-1]), R, (1e-6 * a, dist), work, "fine", fields=True)
    checks = [convergence_check([rows[-1]["p"], fine["p"]], 0.02, f"pressure at {fmt(dist)} m on the axis", "Pa",
                                stopped_by="two meshes", min_levels=2),
              Check("closed form", "pass" if d_spl < 1.0 else ("warn" if d_spl < 2.0 else "fail"),
                    f"the exact on axis pressure of a baffled piston matches the field solve within {d_spl:.2g} dB "
                    f"at {fmt(dist)} m over the band"),
              Check("power balance", "pass" if d_w < 0.03 else ("warn" if d_w < 0.08 else "fail"),
                    f"power from the piston matches the exact radiation resistance of a piston within {100 * d_w:.2g}%")]
    lvl, reasons = confidence(checks)
    limit = value(p, "limit", 0, "db") if "limit" in p else 0.0
    i_max = int(np.argmax(spl))
    if limit:
        verdict = "ok" if spl[i_max] <= limit - 3 else ("marginal" if spl[i_max] <= limit else "fails")
        head = (f"The panel reaches {spl[i_max]:.1f} dB SPL at {fmt(dist)} m near {fmt(freqs[i_max])} Hz "
                f"({'within' if verdict != 'fails' else 'over'} the {limit:g} dB limit), driven at {src}.")
    else:
        verdict = "ok"
        head = (f"The panel makes {spl[0]:.1f} dB SPL at {fmt(fmin)} Hz rising to {spl[i_max]:.1f} dB near "
                f"{fmt(freqs[i_max])} Hz, {fmt(dist)} m in front of it, driven at {src}.")
    fixes = []
    if verdict != "ok":
        cut = spl[i_max] - limit + 3
        fixes.append({"kind": "vibration", "title": f"Cut the panel's vibration by about {cut:.0f} dB",
                      "summary": f"That is a velocity {10 ** (cut / 20):.2g}x smaller, from damping layers, stiffer "
                                 "mounts or isolating the source.", "auto": False})
    images = {"spl": plots.curve(out / "images" / "spl.png", freqs, {"field solve": spl, "exact piston": spl_ex},
                                 "frequency, Hz", f"SPL at {fmt(dist)} m, dB", hline=limit or None,
                                 hlabel=f"limit {limit:g} dB" if limit else "", logx=True)}
    values = {"freqs_Hz": list(freqs), "spl_dB": list(spl), "spl_exact_dB": list(spl_ex), "power_W": list(w_fem),
              "distance_m": dist}
    metrics = [("peak SPL", f"{spl[i_max]:.1f} dB"), ("at", f"{fmt(freqs[i_max])} Hz"), ("distance", f"{fmt(dist)} m"),
               ("radiated power", f"{fmt(float(w_fem[i_max]) * 1e3)} mW")]
    assumptions = [f"The panel moves as a rigid {fmt(2 * a * 1e3)} mm disk (a piston) in an infinite rigid baffle; "
                   "a panel vibrating in its bending modes radiates less, so this is an upper estimate.",
                   "Sound pressure level is the RMS pressure against 20 uPa, free field, air at 20 C.",
                   "The open boundary absorbs outgoing spherical waves to first order."]
    o = GenericOutcome("sound", "panel", question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["spl", "Sound pressure level on the axis"]]}, images, fixes,
                       solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t0)
    _panel_playback(out, fine_m, fine["field"], float(freqs[-1]), a, dist, freqs, spl, spl_ex)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _panel_solve(msh, f: float, v: float, R: float, probe, work: Path, tag: str, fields: bool = False) -> dict:
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = SoundProblem("axi", freq=f, radiation=("outer", R), piston=("piston", v), fields=fields,
                      probes=[(probe[0], probe[1], 0.0)])
    solve.run(helmholtz(sp, msh.tags), msh.path, d, resolution="H")
    pp = solve.read_point(d, "probe0.txt")
    pr = abs(complex(pp[0], pp[1] if len(pp) > 1 else 0.0))
    p1 = solve.read_complex(d, "P1_piston.txt")
    w = 0.5 * (p1 * v).real                           # 1/2 Re of p v* over the piston, v real
    out = {"p": pr, "w": w}
    if fields:
        pc = solve.read_nodes(d, "p.txt", msh.node_ids, comps=2)
        out["field"] = pc[:, 0] + 1j * pc[:, 1]
    return out


# --- playback ---------------------------------------------------------------------------------
def _room_playback(out, shapes, freqs, labels, L, z, unit, fs, exact) -> str:
    """Mode shapes on a horizontal plane, each a standing wave the viewer plays as shape times cos(w t)."""
    from . import playback as pb

    modes = []
    for i, a in enumerate(shapes):
        xs, ys = np.unique(np.round(a[:, 0], 9)), np.unique(np.round(a[:, 1], 9))
        g = np.full((len(ys), len(xs)), np.nan)
        g[np.searchsorted(ys, np.round(a[:, 1], 9)), np.searchsorted(xs, np.round(a[:, 0], 9))] = a[:, 3]
        modes.append({"frequency_Hz": round(freqs[i], 4), "kind": labels[i] if i < len(labels) else "mode",
                      "shape": pb.pack_signed(g)})
    manifest = {"modes": "the first mode shapes: pressure on the plane, packed on a range symmetric about zero, "
                         "to be played as shape times cos(2 pi f t)",
                "modes_Hz": "every mode found below fmax, Hz", "room_mm": "length, width and height, mm",
                "plane_mm": "height of the sampled plane, mm", "schroeder_Hz": "where modes merge, Hz (0 for a box)",
                "exact_Hz": "the exact box modes, Hz, for comparison"}
    return pb.write(out, "sound-modes", manifest,
                    {"room_mm": [round(v * 1e3, 3) for v in L], "plane_mm": round(z * 1e3, 3), "unit_is_mm": unit < 1,
                     "modes": modes, "modes_Hz": [round(f, 4) for f in freqs], "schroeder_Hz": round(fs, 3),
                     "exact_Hz": [round(e[0], 4) for e in exact][:len(freqs)]})


def _muffler_playback(out, msh, shots, dims, freqs, tl, tl_pw) -> str:
    """The pressure phasor through the duct at the first peak and the first pass band, and the loss curve."""
    from . import playback as pb

    a, R, x3 = dims
    win = (0.0, R, 0.0, x3)
    n = (20, int(min(160, max(40, round(20 * x3 / R)))))
    frames = []
    for sh in shots:
        re = pb.grid(msh.X, msh.tris, win, n, node_vals=sh["p"].real)
        im = pb.grid(msh.X, msh.tris, win, n, node_vals=sh["p"].imag)
        m = float(np.nanmax(np.abs(np.concatenate([re.ravel(), im.ravel()]))))
        frames.append({"frequency_Hz": round(sh["f"], 3), "tl_dB": round(sh["tl"], 3),
                       "p_re": pb.pack(re, -m, m), "p_im": pb.pack(im, -m, m)})
    manifest = {"frames": "pressure phasor in the duct for a 1 Pa wave coming in at y = 0, Pa, packed; play "
                          "Re(p e^(j w t)) = p_re cos(w t) - p_im sin(w t)",
                "grid": "r from x0 to x1 and the duct axis from y0 to y1 in mm, axisymmetric about x = 0; cells "
                        "outside the duct are -1",
                "duct_mm": "pipe radius, chamber radius and overall length, mm",
                "tl": "transmission loss over frequency (x in Hz, dB) from the field solve and plane wave theory"}
    return pb.write(out, "sound-muffler", manifest,
                    {"duct_mm": [round(a * 1e3, 3), round(R * 1e3, 3), round(x3 * 1e3, 3)],
                     "grid": {"x0": 0.0, "x1": round(R * 1e3, 3), "y0": 0.0, "y1": round(x3 * 1e3, 3),
                              "nx": n[0], "ny": n[1], "mirror": True},
                     "frames": frames, "tl": pb.series(freqs, {"field_dB": tl, "plane_wave_dB": tl_pw})})


def _panel_playback(out, msh, p, f, a, dist, freqs, spl, spl_ex) -> str:
    """The radiated pressure phasor in front of the panel at the top frequency, and the SPL curve."""
    from . import playback as pb

    w = min(msh.X[:, 0].max(), max(1.3 * dist, 4 * a)) / 1.0
    win = (0.0, w, 0.0, w)
    re = pb.grid(msh.X, msh.tris, win, (64, 64), node_vals=p.real)
    im = pb.grid(msh.X, msh.tris, win, (64, 64), node_vals=p.imag)
    m = float(np.nanpercentile(np.abs(np.concatenate([re.ravel(), im.ravel()])), 99))
    manifest = {"p_re, p_im": "pressure phasor in front of the baffle at frequency_Hz, Pa, packed on one range; "
                              "play Re(p e^(j w t))",
                "grid": "r from x0 to x1 and the distance from the baffle from y0 to y1 in mm, axisymmetric about "
                        "x = 0", "panel_mm": "panel radius and the listening distance, mm",
                "spl": "sound pressure level on the axis at the listening distance (x in Hz, dB re 20 uPa)"}
    return pb.write(out, "sound-panel", manifest,
                    {"frequency_Hz": round(f, 3), "panel_mm": [round(a * 1e3, 3), round(dist * 1e3, 3)],
                     "grid": {"x0": 0.0, "x1": round(w * 1e3, 3), "y0": 0.0, "y1": round(w * 1e3, 3), "nx": 64,
                              "ny": 64, "mirror": True},
                     "p_re": pb.pack(re, -m, m), "p_im": pb.pack(im, -m, m),
                     "spl": pb.series(freqs, {"field_dB": spl, "exact_dB": spl_ex})})
