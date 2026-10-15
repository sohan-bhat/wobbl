"""Antennas and wireless range: resonance, impedance, S11, pattern and gain, and a link budget.

A dipole or a monopole on a ground plane is rotationally symmetric, so Wobbl
solves Maxwell's equations for the azimuthal magnetic field in the r-z plane
with GetDP, frequency by frequency: the wire is a hole whose surface and the
ground plane are perfect conductors, a short gap at the feed carries the
source voltage (a delta gap), and the outer boundary absorbs outgoing waves.
The feed current gives the input impedance, the current along the wire gives
the radiation pattern and directivity, and Wobbl checks that the power fed in
equals the power leaving through the outer boundary and the power in the
pattern, refines the mesh, moves the boundary out, and compares with the
induced EMF theory of thin dipoles.

A patch is not rotationally symmetric. Wobbl sizes it with the transmission
line model (Balanis, Antenna Theory, chapter 14) on an effective permittivity
from a field solve of the patch's cross section, and says plainly that no
full wave solve of the patch was made.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..materials import get_dielectric
from ..outcome_generic import GenericOutcome
from ..units import fmt
from ..verify.checks import Check, confidence, convergence_check
from . import plots, solve
from .forms import C0, MU0, WaveProblem, fullwave_axi
from .mesh2d import Model2D, Shape, mesh
from .params import ParamError, check_known, text, value

ETA0 = MU0 * C0
SIGMA_CU = 5.8e7
EULER = 0.5772156649015329
WIRE_KEYS = {"frequency", "length", "diameter", "gap", "impedance", "span", "points", "tolerance", "tx_power",
             "sensitivity", "rx_gain", "losses", "environment", "distance", "reliability", "heights", "floors",
             "budget", "quality"}
LINK_KEYS = {"frequency", "tx_power", "sensitivity", "tx_gain", "rx_gain", "losses", "environment", "distance",
             "reliability", "heights", "floors", "sigma", "exponent"}
PATCH_KEYS = {"frequency", "width", "length", "height", "dielectric", "er", "impedance", "tx_power", "sensitivity",
              "rx_gain", "losses", "environment", "distance", "reliability", "heights", "floors", "budget", "quality"}


@dataclass
class WireSpec:
    kind: str                     # dipole or monopole
    f0: float
    h: float                      # monopole height, or half the dipole
    a: float                      # wire radius
    g: float                      # gap on the half model
    z0: float
    freqs: list[float]
    tol: float
    link: dict = field(default_factory=dict)


def wire_spec(kind: str, p: dict) -> WireSpec:
    check_known(p, WIRE_KEYS, kind)
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    full = value(p, "length", (0.47 if kind == "dipole" else 0.235) * lam * 1e3, "length")
    h = full / 2 if kind == "dipole" else full
    a = value(p, "diameter", max(lam / 300, 0.2e-3) * 1e3, "length") / 2
    g = value(p, "gap", max(2 * a, lam / 200) * 1e3, "length") / (2 if kind == "dipole" else 1)
    if g >= 0.5 * h:
        raise ParamError("the feed gap is too long for the antenna")
    if a > 0.05 * h:
        raise ParamError("the wire is too thick next to its length for this model. Keep the diameter under a "
                         "tenth of the half length")
    span = value(p, "span", 0.25)
    n = int(value(p, "points", 11))
    freqs = [float(f) for f in np.linspace(f0 * (1 - span), f0 * (1 + span), max(n, 5))]
    s = WireSpec(kind, f0, h, a, g, value(p, "impedance", 50), freqs, value(p, "tolerance", 0.02))
    s.link = {k: p[k] for k in LINK_KEYS if k in p and k != "frequency"}
    return s


def wire_model(s: WireSpec, k: float = 1.0, radius_factor: float = 1.0) -> Model2D:
    lam_min = C0 / max(s.freqs)
    lam_max = C0 / min(s.freqs)
    R = max(2.5 * lam_max, s.h + 1.5 * lam_max) * radius_factor
    wire = Shape("wire", "poly", ((0, 0), (s.a, 0), (s.a, s.g), (s.a, s.h), (0, s.h)), hole=True, size=0.7 * s.a)
    return Model2D("axi", [wire], R, clip="x>=0,y>=0", size=lam_min / 10, far=lam_min / 10, growth=0.25,
                   lines={"gap": f"x={s.a!r}&y<{s.g!r}", "axis": "x=0", "ground": "y=0", "outer": "outer"},
                   spots=[(s.a, s.g, 0.5 * min(s.a, s.g)), (s.a, 0.0, 0.5 * min(s.a, s.g))]).scaled(k)


def _solve_f(msh, s: WireSpec, f: float, work: Path, tag: str, current: bool = False, R: float = 0.0) -> dict:
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    rp = 1.05 * s.a
    p = WaveProblem(f, gap_length=s.g, volts=1.0, probe_line=(rp, s.g, s.h * 0.999, 160) if current else None,
                    fields=current)
    solve.run(fullwave_axi(p, msh.tags), msh.path, d, resolution="F")
    if current:
        hh = solve.read_nodes(d, "h.txt", msh.node_ids, comps=2)
        field = {"mesh": msh, "h": hh[:, 0] + 1j * hh[:, 1]}
    else:
        field = {}
    ig = solve.read_complex(d, "ig.txt")
    I = -ig                                   # current flowing up the wire for a positive feed voltage
    out = {"Z": 1.0 / I, "I": I, "prad_abc": float(solve.read_global(d, "prad.txt")[0]), **field}
    if current:
        rows = [ln.split() for ln in (d / "hwire.txt").read_text().splitlines() if ln.split()]
        z = np.array([float(r[3]) for r in rows])
        hv = np.array([complex(float(r[-2]), float(r[-1])) for r in rows])
        out["z"], out["Iz"] = z, -2 * math.pi * rp * hv
    return out


def pattern(z: np.ndarray, Iz: np.ndarray, f: float, gap: float, I0: complex, n: int = 361):
    """Far field of the symmetric line current I(z) on |z| < h (the wire and its image): theta from the axis,
    the power density U(theta) per unit input current scale, and the total radiated power of the full dipole."""
    k = 2 * math.pi * f / C0
    zz = np.concatenate([[0.0], [gap * 0.5], z])
    ii = np.concatenate([[I0], [I0], Iz])
    th = np.linspace(1e-6, math.pi - 1e-6, n)
    F = np.array([np.trapezoid(ii * 2 * np.cos(k * zz * math.cos(t)), zz) for t in th])
    U = ETA0 * k ** 2 / (32 * math.pi ** 2) * np.abs(np.sin(th) * F) ** 2
    P = 2 * math.pi * np.trapezoid(U * np.sin(th), th)
    return th, U, P


def induced_emf(length: float, a: float, f: float) -> complex:
    """Input impedance of a thin center fed dipole of the given total length with a sinusoidal current (induced EMF
    method, Balanis, Antenna Theory, eqs 4-70, 4-79 and 8-60). 73.1 + j42.5 ohm at half a wavelength."""
    from scipy.special import sici

    k = 2 * math.pi * f / C0
    kl = k * length
    si, ci = sici(kl)
    si2, ci2 = sici(2 * kl)
    _, cia = sici(2 * k * a * a / length)
    rr = ETA0 / (2 * math.pi) * (EULER + math.log(kl) - ci + 0.5 * math.sin(kl) * (si2 - 2 * si) +
                                 0.5 * math.cos(kl) * (EULER + math.log(kl / 2) + ci2 - 2 * ci))
    xm = ETA0 / (4 * math.pi) * (2 * si + math.cos(kl) * (2 * si - si2) -
                                 math.sin(kl) * (2 * ci - ci2 - cia))
    sn = math.sin(kl / 2) ** 2
    return complex(rr / sn, xm / sn)


def _crossing(freqs, Z) -> tuple[float, complex] | None:
    x = np.array([z.imag for z in Z])
    for i in range(len(x) - 1):
        if x[i] <= 0 < x[i + 1] or x[i] >= 0 > x[i + 1]:
            t = x[i] / (x[i] - x[i + 1])
            f = freqs[i] + t * (freqs[i + 1] - freqs[i])
            return f, Z[i] + t * (Z[i + 1] - Z[i])
    return None


def s11_db(Z, z0: float):
    g = np.abs((np.asarray(Z) - z0) / (np.asarray(Z) + z0))
    return 20 * np.log10(np.maximum(g, 1e-6))


def run_antenna(template: str, params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
                log=lambda stage, detail="": None) -> GenericOutcome:
    if template == "link":
        return run_link(params, out_dir, question)
    if template not in ("dipole", "monopole", "patch", "loop", "invf"):
        raise ParamError(f"unknown antenna template '{template}' (use dipole, monopole, patch, loop, invf or link)")
    from .radiate import route, run_radiate

    way = route(template, params, question)
    if way.full_wave:
        return run_radiate(template, params, out_dir, question, keep, log, why=way.reasons)
    if template == "patch":
        return run_patch(params, out_dir, question, keep, log, way.reasons)
    t_start = time.time()
    s = wire_spec(template, dict(params))
    dip = template == "dipole"
    mult = 2.0 if dip else 1.0                 # the half model's impedance doubles for the full dipole
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    msh = mesh(wire_model(s), work / "m.msh")
    sweep = []
    for i, f in enumerate(s.freqs):
        log("solve", f"{fmt(f / 1e6)} MHz ({i + 1} of {len(s.freqs)})")
        sweep.append(_solve_f(msh, s, f, work, f"f{i}"))
    Z = [mult * r["Z"] for r in sweep]
    cross = _crossing(s.freqs, Z)
    f_eval = cross[0] if cross else s.f0
    log("verify", "a finer mesh and a farther boundary at the working frequency")
    base = _solve_f(msh, s, s.f0, work, "base", current=True)
    fine_m = mesh(wire_model(s, 0.7), work / "fine.msh")
    fine = _solve_f(fine_m, s, s.f0, work, "fine")
    far_m = mesh(wire_model(s, 1.0, 1.5), work / "far.msh")
    far = _solve_f(far_m, s, s.f0, work, "far")
    z_base, z_fine, z_far = mult * base["Z"], mult * fine["Z"], mult * far["Z"]
    checks = [convergence_check([abs(z_base), abs(z_fine)], 0.01, "input impedance magnitude", "ohm",
                                stopped_by="two meshes", min_levels=2)]
    bd = abs(z_far - z_base) / abs(z_base)
    checks.append(Check("boundary distance", "pass" if bd < 0.01 else ("warn" if bd < 0.03 else "fail"),
                        f"moving the absorbing boundary half again as far out changes the impedance from "
                        f"{_zs(z_base)} to {_zs(z_far)} ({100 * bd:.2g}%)"))
    p_in = 0.5 * (1.0 / base["Z"]).real          # half model, 1 V across its gap
    p_abc = base["prad_abc"]
    eb = abs(p_abc - p_in) / p_in
    checks.append(Check("power balance", "pass" if eb < 0.02 else ("warn" if eb < 0.05 else "fail"),
                        f"power fed in at the gap {fmt(p_in * 1e3)} mW, power leaving through the outer boundary "
                        f"{fmt(p_abc * 1e3)} mW ({100 * eb:.2g}% apart)"))
    th, U, P = pattern(base["z"], base["Iz"], s.f0, s.g, base["I"])
    p_full_in = 2 * p_in                          # the image half radiates as much again
    pb = abs(P - p_full_in) / p_full_in
    checks.append(Check("pattern power", "pass" if pb < 0.03 else ("warn" if pb < 0.08 else "fail"),
                        f"the radiation integral of the current along the wire gives {fmt(P * 1e3)} mW against "
                        f"{fmt(p_full_in * 1e3)} mW fed to the full dipole ({100 * pb:.2g}% apart)"))
    D_dip = 4 * math.pi * float(U.max()) / P
    D = D_dip * (2.0 if not dip else 1.0)
    full_len = 2 * s.h
    zt = induced_emf(full_len, s.a, s.f0) * (1.0 if dip else 0.5)
    rs = math.sqrt(2 * math.pi * s.f0 * MU0 / (2 * SIGMA_CU))
    r_loss = mult * (s.h / (2 * math.pi * s.a)) * rs * 0.5     # sinusoidal current along the wire
    eff = z_base.real / (z_base.real + r_loss)
    G = D * eff
    gdb = 10 * math.log10(G)
    if cross:
        r_ref = 73.1 * (1.0 if dip else 0.5)
        dr = abs(cross[1].real - r_ref) / r_ref
        checks.append(Check("closed form", "pass" if dr < 0.08 else ("warn" if dr < 0.2 else "fail"),
                            f"resonance at {fmt(cross[0] / 1e6)} MHz with {fmt(cross[1].real)} ohm, against "
                            f"{fmt(r_ref)} ohm for a thin half wave {'dipole' if dip else 'monopole'} "
                            f"({100 * dr:.2g}% apart). Induced EMF theory puts this length at {_zs(zt)} at "
                            f"{fmt(s.f0 / 1e6)} MHz, the field solve {_zs(z_base)}",
                            {"induced_emf_ohm": [zt.real, zt.imag]}))
    else:
        checks.append(Check("closed form", "info", f"no resonance inside the sweep. Induced EMF theory puts this "
                                                   f"length at {_zs(zt)} at {fmt(s.f0 / 1e6)} MHz, the field solve "
                                                   f"{_zs(z_base)}"))
    lvl, reasons = confidence(checks)
    s11 = s11_db(Z, s.z0)
    s11_0 = float(s11_db([z_fine], s.z0)[0])
    band = [f for f, v in zip(s.freqs, s11) if v <= -10]
    bw = (max(band) - min(band)) if len(band) > 1 else 0.0
    if cross:
        off = (cross[0] - s.f0) / s.f0
        word = f"resonates at {fmt(cross[0] / 1e6)} MHz ({100 * off:+.1f}% from {fmt(s.f0 / 1e6)} MHz)"
    else:
        off = 1.0
        word = "does not resonate inside the sweep"
    if cross and abs(off) <= s.tol and s11_0 <= -10:
        verdict = "ok"
    elif s11_0 <= -6 and cross and abs(off) <= 2 * s.tol:
        verdict = "marginal"
    else:
        verdict = "fails"
    name = "dipole" if dip else "monopole"
    head = (f"The {fmt(full_len * 1e3 if dip else s.h * 1e3)} mm {name} {word}, with {_zs(z_fine)} and S11 of "
            f"{s11_0:.1f} dB against {fmt(s.z0)} ohm at {fmt(s.f0 / 1e6)} MHz. Peak gain {gdb:.2f} dBi.")
    fixes = []
    if verdict != "ok" and cross:
        new_len = (full_len if dip else s.h) * cross[0] / s.f0
        fixes.append({"kind": "length", "title": f"Trim the {name} to about {fmt(new_len * 1e3)} mm",
                      "summary": "Resonance scales inversely with length, so this moves it onto the working frequency.",
                      "auto": False})
    if cross and abs(cross[1].real - s.z0) / s.z0 > 0.5:
        fixes.append({"kind": "match", "title": "Add a matching network or a balun",
                      "summary": f"At resonance the antenna is {fmt(cross[1].real)} ohm, far from {fmt(s.z0)} ohm.",
                      "auto": False})
    images = {}
    fz = np.array(s.freqs) / 1e6
    images["s11"] = plots.curve(out / "images" / "s11.png", fz, {"S11, dB": s11}, "frequency, MHz", "S11, dB",
                                hline=-10, hlabel="-10 dB")
    images["impedance"] = plots.curve(out / "images" / "impedance.png", fz,
                                      {"resistance, ohm": [z.real for z in Z], "reactance, ohm": [z.imag for z in Z]},
                                      "frequency, MHz", "ohm", hline=0.0)
    gain_db = 10 * np.log10(np.maximum(4 * math.pi * U / P * (2.0 if not dip else 1.0) * eff, 1e-6))
    if not dip:
        gain_db = np.where(th <= math.pi / 2, gain_db, -60)
    full_th = np.concatenate([th, th + math.pi])
    images["pattern"] = plots.polar(out / "images" / "pattern.png", full_th, {"gain": np.concatenate([gain_db,
                                                                                                    gain_db[::-1]])},
                                    "elevation pattern, gain in dBi (0 is along the wire)")
    values = {"freqs_Hz": s.freqs, "z_ohm": [[z.real, z.imag] for z in Z], "s11_dB": [float(v) for v in s11],
              "resonance_Hz": cross[0] if cross else None, "z_resonance_ohm": [cross[1].real, cross[1].imag]
              if cross else None, "z_at_f0_ohm": [z_fine.real, z_fine.imag], "s11_at_f0_dB": s11_0,
              "bandwidth_Hz": bw, "directivity_dBi": 10 * math.log10(D), "gain_dBi": gdb, "efficiency": eff,
              "induced_emf_ohm": [zt.real, zt.imag], "nodes": [msh.n_nodes, fine_m.n_nodes]}
    metrics = [("resonance", f"{fmt(cross[0] / 1e6)} MHz" if cross else "none in sweep"), ("S11", f"{s11_0:.1f} dB"),
               ("impedance", _zs(z_fine)), ("gain", f"{gdb:.2f} dBi")]
    assumptions = [f"Solved with the axisymmetric model, no 3D solve was needed. {way.reasons[0][:1].upper()}"
                   f"{way.reasons[0][1:]}. Give a finite ground plate or put the antenna near something and it is "
                   "solved in 3D.",
                   f"{'Center fed dipole' if dip else 'Monopole on an infinite, perfect ground plane'} in free space, "
                   f"wire diameter {fmt(2 * s.a * 1e3)} mm, fed across a {fmt(s.g * mult * 1e3)} mm gap (delta gap).",
                   "Axisymmetric full wave solve, and the outer boundary absorbs outgoing waves to first order.",
                   f"Copper conductor loss from the skin effect, {100 * (1 - eff):.2g}% of the power. No feed line, "
                   "balun or nearby objects."]
    sections, link_vals = [], {}
    if s.link.get("tx_power") or s.link.get("distance"):
        lk = link_budget({**s.link, "frequency": s.f0, "tx_gain": s.link.get("tx_gain", gdb),
                          "rx_gain": s.link.get("rx_gain", gdb)})
        link_vals = lk["values"]
        checks += lk["checks"]
        head += " " + lk["headline"]
        metrics.append(("range, 90%", f"{fmt(lk['values']['range_reliable_m'])} m"))
        if lk["verdict"] == "fails" or (lk["verdict"] == "marginal" and verdict == "ok"):
            verdict = lk["verdict"]
        assumptions += lk["assumptions"]
        fixes += lk["fixes"]
        images["range"] = lk["image"](out / "images" / "range.png")
        lvl, reasons = confidence(checks, [("medium", "the range comes from an empirical path loss model with "
                                                      f"{fmt(lk['values']['sigma_dB'])} dB of shadowing")])
    o = GenericOutcome("antenna", template, question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**values, **link_vals, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["s11", "S11 over the sweep"], ["impedance", "Input impedance"],
                                     ["pattern", "Elevation pattern"], ["range", "Received power over distance"]]},
                       images, fixes, solver=f"GetDP {solve.getdp().version()} with gmsh",
                       t_total=time.time() - t_start)
    _wire_playback(s, dip, base, Z, s11, th, gain_db, lk if link_vals else None, out)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out, sections)


def _wire_playback(s: WireSpec, dip: bool, base: dict, Z, s11, th, gain_db, lk, out: Path) -> str:
    """The field around the wire at the working frequency as a complex phasor on the r-z plane, so a viewer can
    play the radiated wave as Re(H e^(j w t)), with the pattern, the sweep and the range."""
    from . import playback as pb

    lam = C0 / s.f0
    msh = base["mesh"]
    w = min(msh.X[:, 0].max(), max(1.25 * lam, 1.4 * s.h))
    win = (0.0, w, 0.0, w)
    n = 72
    re = pb.grid(msh.X, msh.tris, win, (n, n), node_vals=base["h"].real)
    im = pb.grid(msh.X, msh.tris, win, (n, n), node_vals=base["h"].imag)
    m = float(np.nanpercentile(np.abs(np.concatenate([re.ravel(), im.ravel()])), 98))
    keep = np.arange(0, len(th), max(1, len(th) // 90))
    data = {"antenna": "dipole" if dip else "monopole", "frequency_Hz": s.f0,
            "wire": {"radius_mm": round(s.a * 1e3, 4), "height_mm": round(s.h * 1e3, 4),
                     "gap_mm": round(s.g * 1e3, 4), "mirror_z": dip},
            "grid": {"x0": 0.0, "x1": round(w * 1e3, 3), "y0": 0.0, "y1": round(w * 1e3, 3), "nx": n, "ny": n,
                     "mirror": True},
            "h_re": pb.pack(re, -m, m), "h_im": pb.pack(im, -m, m),
            "pattern": {"theta_deg": [round(float(np.degrees(t)), 2) for t in th[keep]],
                        "gain_dBi": [round(float(g), 3) for g in gain_db[keep]]},
            "sweep": pb.series(np.array(s.freqs) / 1e6, {"s11_dB": s11, "r_ohm": [z.real for z in Z],
                                                         "x_ohm": [z.imag for z in Z]})}
    manifest = {"h_re, h_im": "real and imaginary parts of the azimuthal magnetic field phasor on the grid, A/m for "
                              "1 V at the feed, packed on one range clipped at the 98th percentile (the field "
                              "grows without bound at the wire); play Re(H e^(j w t)) = h_re cos(w t) - h_im sin(w t)",
                "grid": "r from x0 to x1 and z from y0 to y1 in mm; mirror about the axis, and for a dipole about "
                        "z = 0 too (wire.mirror_z)",
                "wire": "the wire's radius and height above z = 0 (half the dipole) and the feed gap, mm",
                "pattern": "gain in dBi against the angle from the wire's axis, degrees",
                "sweep": "x is frequency in MHz, with S11 in dB and the input resistance and reactance in ohm"}
    if lk:
        data["range"] = _range_playback(lk)
        manifest.update(RANGE_MANIFEST)
    return pb.write(out, "antenna-wire", manifest, data)


def _zs(z: complex) -> str:
    return f"{fmt(z.real)} {'+' if z.imag >= 0 else '-'} j{fmt(abs(z.imag))} ohm"


# --- link budget ------------------------------------------------------------------------------
ENVIRONMENTS = {
    # ITU-R P.1238 site general model: power loss coefficient N and shadow fading sigma, near 2.4 GHz
    "office": {"model": "itu", "N": 30.0, "sigma": 10.0, "floor": 15.0},
    "indoor": {"model": "itu", "N": 30.0, "sigma": 10.0, "floor": 15.0},
    "home": {"model": "itu", "N": 28.0, "sigma": 8.0, "floor": 10.0},
    "residential": {"model": "itu", "N": 28.0, "sigma": 8.0, "floor": 10.0},
    "factory": {"model": "itu", "N": 22.0, "sigma": 9.0, "floor": 15.0},
    "free": {"model": "free", "sigma": 0.0},
    "line-of-sight": {"model": "free", "sigma": 4.0},
    "outdoor": {"model": "tworay", "sigma": 6.0},
    "urban": {"model": "log", "n": 3.5, "sigma": 8.0},
    "suburban": {"model": "log", "n": 3.0, "sigma": 7.0},
}
Z_RELIABILITY = {0.5: 0.0, 0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def path_loss(d, f: float, env: dict, heights=(1.5, 1.5), floors: int = 0):
    """Median path loss in dB at distance d (m)."""
    d = np.maximum(np.asarray(d, float), 0.1)
    lam = C0 / f
    fs = 20 * np.log10(4 * math.pi * d / lam)
    m = env["model"]
    if m == "free":
        return fs
    if m == "itu":
        return 20 * math.log10(f / 1e6) + env["N"] * np.log10(d) - 28 + env.get("floor", 15) * floors
    if m == "log":
        d0 = 1.0
        return 20 * math.log10(4 * math.pi * d0 / lam) + 10 * env["n"] * np.log10(d / d0)
    ht, hr = heights
    dc = 4 * math.pi * ht * hr / lam
    two = 40 * np.log10(d) - 20 * math.log10(ht * hr)
    return np.where(d < dc, fs, np.maximum(two, fs))


def _range(budget: float, f: float, env: dict, heights, floors) -> float:
    """Distance where the path loss uses up the budget, by bisection on a log scale."""
    lo, hi = 0.1, 1e6
    if float(path_loss(lo, f, env, heights, floors)) > budget:
        return 0.0
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if float(path_loss(mid, f, env, heights, floors)) <= budget else (lo, mid)
    return math.sqrt(lo * hi)


def link_budget(p: dict) -> dict:
    f = value(p, "frequency", 2.4e9, "frequency")
    pt = value(p, "tx_power", 0, "dbm")
    sens = value(p, "sensitivity", -90, "dbm")
    gt = value(p, "tx_gain", 2.15, "db")
    gr = value(p, "rx_gain", 2.15, "db")
    loss = value(p, "losses", 0, "db")
    env_name = text(p, "environment", "office")
    if env_name not in ENVIRONMENTS:
        raise ParamError(f"unknown environment '{env_name}' (use {', '.join(ENVIRONMENTS)})")
    env = dict(ENVIRONMENTS[env_name])
    if "exponent" in p:
        env = {"model": "log", "n": value(p, "exponent", 3.0), "sigma": env.get("sigma", 8.0)}
    if "sigma" in p:
        env["sigma"] = value(p, "sigma", 8, "db")
    rel = value(p, "reliability", 0.9)
    rel = rel / 100 if rel > 1 else rel
    zr = Z_RELIABILITY.get(round(rel, 2))
    if zr is None:
        from scipy.stats import norm
        zr = float(norm.ppf(rel))
    hs = str(p.get("heights", "1.5,1.5")).split(",")
    heights = (float(hs[0]), float(hs[-1]))
    floors = int(value(p, "floors", 0))
    budget = pt + gt + gr - loss - sens
    sigma = env["sigma"]
    r50 = _range(budget, f, env, heights, floors)
    rrel = _range(budget - zr * sigma, f, env, heights, floors)
    rfs = _range(budget, f, ENVIRONMENTS["free"], heights, 0)
    lo_n = _range(budget, f, {**env, "N": env["N"] - 3} if env["model"] == "itu" else env, heights, floors)
    hi_n = _range(budget, f, {**env, "N": env["N"] + 3} if env["model"] == "itu" else env, heights, floors)
    dist = value(p, "distance", 0, "distance") if "distance" in p else 0.0
    checks = [Check("free space bound", "pass" if r50 <= rfs * 1.0001 else "warn",
                    f"the {env_name} model's median range {fmt(r50)} m is within the free space limit {fmt(rfs)} m"),
              Check("link uncertainty", "info",
                    f"shadow fading of {fmt(sigma)} dB moves the range from {fmt(r50)} m (half the spots) to "
                    f"{fmt(rrel)} m ({100 * rel:.0f}% of spots), and a power loss coefficient 3 lower or higher gives "
                    f"{fmt(hi_n)} to {fmt(lo_n)} m")]
    if dist:
        rx = pt + gt + gr - loss - float(path_loss(dist, f, env, heights, floors))
        margin = rx - sens
        verdict = "ok" if rrel >= dist else ("marginal" if r50 >= dist else "fails")
        head = (f"At {fmt(dist)} m the median received power is {rx:.1f} dBm, {margin:.1f} dB above the "
                f"{sens:.0f} dBm sensitivity, and the link holds out to {fmt(rrel)} m in {100 * rel:.0f}% of spots and "
                f"{fmt(r50)} m in half of them ({env_name}).")
    else:
        verdict = "ok"
        head = (f"The link reaches about {fmt(r50)} m in half of spots and {fmt(rrel)} m in {100 * rel:.0f}% of them "
                f"({env_name}, {pt:.0f} dBm out, {sens:.0f} dBm sensitivity), against {fmt(rfs)} m in free space.")
    fixes = []
    if verdict != "ok" and dist:
        need = float(path_loss(dist, f, env, heights, floors)) + zr * sigma - (pt + gt + gr - loss - sens)
        fixes.append({"kind": "power", "title": f"Find about {need:.1f} dB more link budget",
                      "summary": "From transmit power, receiver sensitivity (a lower data rate helps), higher gain "
                                 "antennas or fewer walls in the path.", "auto": False})
    assumptions = [f"Path loss from the {_model_text(env, env_name)} at {fmt(f / 1e6)} MHz.",
                   f"Link budget {pt:.1f} dBm out, {gt:.2f} dBi and {gr:.2f} dBi antennas, {loss:.1f} dB of other "
                   f"losses, {sens:.1f} dBm sensitivity, and a fading margin of {zr * sigma:.1f} dB for {100 * rel:.0f}% of "
                   "spots."]

    d = np.logspace(0, math.log10(max(r50, rfs, dist or 1) * 3), 120)
    prx = pt + gt + gr - loss - path_loss(d, f, env, heights, floors)
    pfs = pt + gt + gr - loss - path_loss(d, f, ENVIRONMENTS["free"], heights, 0)
    curves = {"median_dBm": prx, "reliable_dBm": prx - zr * sigma, "free_space_dBm": pfs}

    def image(path):
        ys = {"median": prx, f"{100 * rel:.0f}% of spots": prx - zr * sigma, "free space": pfs}
        return plots.curve(path, d, ys, "distance, m", "received power, dBm", hline=sens, hlabel="sensitivity",
                           logx=True)

    return {"verdict": verdict, "headline": head, "checks": checks, "fixes": fixes, "assumptions": assumptions,
            "image": image, "curves": (d, curves, sens, dist),
            "values": {"range_median_m": r50, "range_reliable_m": rrel, "range_free_space_m": rfs,
                       "link_budget_dB": budget, "sigma_dB": sigma, "reliability": rel, "environment": env_name}}


def _range_playback(lk: dict) -> dict:
    from .playback import series

    d, curves, sens, dist = lk["curves"]
    return {"curves": series(d, curves, 120), "sensitivity_dBm": sens, "distance_m": dist or None,
            **{k: round(float(v), 4) if isinstance(v, float) else v for k, v in lk["values"].items()}}


RANGE_MANIFEST = {"range.curves": "received power against distance (x in m, dBm) for the median, the reliable "
                                  "fraction of spots and free space",
                  "range.range_median_m": "distance reached in half of spots, m",
                  "range.range_reliable_m": "distance reached in the reliability fraction of spots, m"}


def _model_text(env: dict, name: str) -> str:
    m = env["model"]
    if m == "itu":
        return (f"ITU-R P.1238 site general indoor model (power loss coefficient {env['N']:g}, "
                f"{env['sigma']:g} dB shadow fading, for a typical {name})")
    if m == "free":
        return "Friis free space equation"
    if m == "log":
        return f"log distance model with exponent {env['n']:g} and {env['sigma']:g} dB shadow fading"
    return "two ray ground reflection model (free space before the breakpoint distance, 40 dB per decade after)"


def run_link(params: dict, out_dir: str | Path, question: str = "") -> GenericOutcome:
    t0 = time.time()
    check_known(params, LINK_KEYS, "link")
    lk = link_budget(dict(params))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    images = {"range": lk["image"](out / "images" / "range.png")}
    v = lk["values"]
    lvl, reasons = confidence(lk["checks"], [("medium", "the range comes from an empirical path loss model with "
                                                        f"{fmt(v['sigma_dB'])} dB of shadowing")])
    metrics = [("range, median", f"{fmt(v['range_median_m'])} m"),
               (f"range, {100 * v['reliability']:.0f}%", f"{fmt(v['range_reliable_m'])} m"),
               ("free space", f"{fmt(v['range_free_space_m'])} m"), ("budget", f"{v['link_budget_dB']:.1f} dB")]
    o = GenericOutcome("antenna", "link", question, lk["verdict"], lk["headline"], metrics, lvl, reasons, lk["checks"],
                       lk["assumptions"], {**v, "params": {k: str(x) for k, x in params.items()},
                                           "_figures": [["range", "Received power over distance"]]}, images,
                       lk["fixes"], solver="closed form link budget", t_total=time.time() - t0)
    from .playback import write
    write(out, "antenna-link", dict(RANGE_MANIFEST), {"range": _range_playback(lk)})
    return o.write(out)


# --- patch ------------------------------------------------------------------------------------------
def run_patch(params: dict, out_dir: str | Path, question: str = "", keep: bool = False,
              log=lambda stage, detail="": None, why: list | None = None) -> GenericOutcome:
    from .board import hammerstad_jensen, run_board

    t0 = time.time()
    p = dict(params)
    check_known(p, PATCH_KEYS, "patch")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    h = value(p, "height", 1.6, "length")
    diel = text(p, "dielectric", "fr4")
    er = value(p, "er", 0) or get_dielectric(diel).epsr
    lam0 = C0 / f0
    W = value(p, "width", C0 / (2 * f0) * math.sqrt(2 / (er + 1)) * 1e3, "length")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log("solve", "effective permittivity of the patch's cross section")
    bo = run_board("microstrip", {"width": f"{W * 1e3!r} mm", "height": f"{h * 1e3!r} mm", "thickness": "0.035 mm",
                                  "er": er, "target": 0}, out / "cross-section", keep=keep)
    ee = float(bo.values["eps_eff"])
    _, ee_hj = hammerstad_jensen(W, h, 35e-6, er)
    dl = 0.412 * h * (ee + 0.3) * (W / h + 0.264) / ((ee - 0.258) * (W / h + 0.8))
    L = value(p, "length", (C0 / (2 * f0 * math.sqrt(ee)) - 2 * dl) * 1e3, "length")
    fr = C0 / (2 * (L + 2 * dl) * math.sqrt(ee))
    k0 = 2 * math.pi * fr / C0
    from scipy.special import j0

    th = np.linspace(1e-6, math.pi - 1e-6, 2001)
    base = np.sin(k0 * W * np.cos(th) / 2) ** 2 / np.cos(th) ** 2 * np.sin(th) ** 3
    I1 = np.trapezoid(base, th)
    G1 = I1 / (120 * math.pi ** 2)
    G12 = np.trapezoid(base * j0(k0 * L * np.sin(th)), th) / (120 * math.pi ** 2)
    rin = 1 / (2 * (G1 + G12))
    z0 = value(p, "impedance", 50)
    y0 = L / math.pi * math.acos(math.sqrt(min(z0 / rin, 1.0)))
    # directivity of the two radiating slots over the ground plane, by numerical integration
    tt, pp = np.meshgrid(np.linspace(1e-4, math.pi / 2, 181), np.linspace(1e-4, 2 * math.pi, 361), indexing="ij")
    X = k0 * W / 2 * np.sin(tt) * np.sin(pp)
    Zt = k0 * L / 2 * np.sin(tt) * np.cos(pp)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = np.where(np.abs(X) < 1e-9, 1.0, np.sin(X) / X)
    Up = (sx * np.cos(Zt)) ** 2 * (np.cos(pp) ** 2 * np.cos(tt) ** 2 + np.sin(pp) ** 2)
    Prad = np.trapezoid(np.trapezoid(Up * np.sin(tt), pp[0], axis=1), tt[:, 0])
    D = 4 * math.pi * Up.max() / Prad
    off = (fr - f0) / f0
    ee_err = abs(ee - ee_hj) / ee_hj
    checks = [c for c in bo.checks if c.name in ("mesh convergence", "charge balance")]
    checks.append(Check("closed form", "pass" if ee_err < 0.01 else "warn",
                        f"the field solved effective permittivity {ee:.4g} under a {fmt(W * 1e3)} mm wide patch matches "
                        f"Hammerstad and Jensen's {ee_hj:.4g} ({100 * ee_err:.2g}% apart)"))
    checks.append(Check("model limits", "info",
                        "the transmission line model predicts patch resonance to about 2 to 4% on thin substrates, "
                        "and no full wave solve of the patch was made"))
    lvl, reasons = confidence(checks, [("medium", "the patch is sized with the transmission line model, not a full "
                                                  "wave solve")])
    verdict = "ok" if abs(off) <= 0.02 else ("marginal" if abs(off) <= 0.04 else "fails")
    head = (f"A {fmt(W * 1e3)} x {fmt(L * 1e3)} mm patch on {fmt(h * 1e3)} mm of er {er:g} resonates near "
            f"{fmt(fr / 1e6)} MHz ({100 * off:+.1f}% from {fmt(f0 / 1e6)} MHz), with a {fmt(rin)} ohm edge resistance, "
            f"an inset of {fmt(y0 * 1e3)} mm for {fmt(z0)} ohm and about {10 * math.log10(D):.1f} dBi directivity.")
    fixes = []
    if verdict != "ok":
        Ln = C0 / (2 * f0 * math.sqrt(ee)) - 2 * dl
        fixes.append({"kind": "length", "title": f"Make the patch {fmt(Ln * 1e3)} mm long",
                      "summary": "The resonant length is half a guided wavelength less the fringing at both edges.",
                      "auto": False})
    images = {"field": bo.images.get("field", "")}
    assumptions = [f"Rectangular patch over an infinite ground plane, substrate {fmt(h * 1e3)} mm thick with er {er:g}.",
                   "Transmission line model (Balanis, Antenna Theory, chapter 14): two radiating slots joined by a "
                   "line, fringing length from Hammerstad's formula, edge conductance with mutual coupling.",
                   "Dielectric and conductor loss, surface waves and the feed's own reactance are not modeled."]
    assumptions += [f"No 3D solve was made. {r[:1].upper() + r[1:]}. Give the patch's length, width or feed, or ask "
                    "for its S11, gain or pattern, and it is solved in 3D." for r in (why or [])]
    metrics = [("resonance", f"{fmt(fr / 1e6)} MHz"), ("edge resistance", f"{fmt(rin)} ohm"),
               ("inset for match", f"{fmt(y0 * 1e3)} mm"), ("directivity", f"{10 * math.log10(D):.1f} dBi")]
    vals = {"width_m": W, "length_m": L, "resonance_Hz": fr, "eps_eff": ee, "fringe_m": dl, "edge_resistance_ohm": rin,
            "inset_m": y0, "directivity_dBi": 10 * math.log10(D)}
    if p.get("tx_power") or p.get("distance"):
        lk = link_budget({**{k: p[k] for k in LINK_KEYS if k in p}, "frequency": f0,
                          "tx_gain": p.get("tx_gain", 10 * math.log10(D)), "rx_gain": p.get("rx_gain", 2.15)})
        checks += lk["checks"]
        head += " " + lk["headline"]
        vals.update(lk["values"])
        images["range"] = lk["image"](out / "images" / "range.png")
        fixes += lk["fixes"]
        assumptions += lk["assumptions"]
        if lk["verdict"] != "ok":
            verdict = lk["verdict"] if lk["verdict"] == "fails" or verdict == "ok" else verdict
    if not keep:
        shutil.rmtree(out / "cross-section", ignore_errors=True)
        images.pop("field", None)
    o = GenericOutcome("antenna", "patch", question, verdict, head, metrics, lvl, reasons, checks, assumptions,
                       {**vals, "params": {k: str(v) for k, v in params.items()},
                        "_figures": [["range", "Received power over distance"]]}, images, fixes,
                       solver=f"GetDP {solve.getdp().version()} (cross section) and the transmission line model",
                       t_total=time.time() - t0)
    from .playback import write
    pdata = {"patch": {"width_mm": round(W * 1e3, 4), "length_mm": round(L * 1e3, 4), "height_mm": round(h * 1e3, 4),
                       "inset_mm": round(y0 * 1e3, 4), "er": er, "eps_eff": round(ee, 5)},
             "numbers": {"resonance_Hz": round(fr, 1), "edge_resistance_ohm": round(rin, 3),
                         "directivity_dBi": round(10 * math.log10(D), 3)}}
    pman = {"patch": "the patch's size, substrate height and the inset feed for the match, mm",
            "numbers": "resonance, edge resistance and directivity from the transmission line model"}
    if p.get("tx_power") or p.get("distance"):
        pdata["range"] = _range_playback(lk)
        pman.update(RANGE_MANIFEST)
    write(out, "antenna-patch", pman, pdata)
    return o.write(out)
