"""Antennas described by their dimensions, built as 3D geometry for the full wave solver.

A wire (dipole, monopole, loop) is a conductor body, a printed antenna (patch, inverted F) is copper sheets on a
dielectric slab, and a ground plane is a copper sheet. Every builder returns the scene and the wave3d setup with
its feed port, so one run path serves all of them, and a STEP file or a KiCad board goes through the same solver
by building the same two objects from its own geometry.

The defaults are textbook first guesses from the frequency alone (a dipole of 0.47 wavelength, a patch from the
transmission line model, a quarter wave inverted F arm shortened for the substrate). They are starting points the
solver then checks, not fits to the solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..materials import get_dielectric
from .forms import C0
from .params import ParamError, check_known, text, value
from .wave3d import Port, WaveSetup

KINDS = ("dipole", "monopole", "loop", "patch", "invf")
COMMON = {"frequency", "span", "points", "tolerance", "impedance", "tx_power", "sensitivity", "rx_gain", "losses",
          "environment", "distance", "reliability", "heights", "floors", "budget", "quality"}
KEYS = {
    "dipole": COMMON | {"length", "diameter", "gap"},
    "monopole": COMMON | {"length", "diameter", "gap", "ground"},
    "loop": COMMON | {"radius", "circumference", "diameter", "gap"},
    "patch": COMMON | {"width", "length", "height", "dielectric", "er", "loss_tangent", "feed", "ground"},
    "invf": COMMON | {"length", "height", "width", "dielectric", "er", "loss_tangent", "ground", "feed", "board"},
}


@dataclass
class Built:
    kind: str
    scene: object
    setup: WaveSetup
    title: str
    notes: list[str] = field(default_factory=list)
    dims: dict = field(default_factory=dict)         # the antenna's own sizes in m, for the report and the scene
    ref: dict = field(default_factory=dict)          # the simple model's prediction, for the report


# --- solids written as a STEP file -------------------------------------------------------------------
def _write_solids(specs: list, path: str) -> str:
    """Each spec is (kind, name, ...) in mm: ('cyl', name, p0, p1, r), ('box', name, lo, hi) or
    ('arc', name, R, r, sweep_deg) for a circular wire of radius r about the z axis, centered on +x and open at
    the end that the sweep leaves, written to path."""
    from .scene import _session

    gmsh = _session()
    try:
        occ = gmsh.model.occ
        for sp in specs:
            kind, name = sp[0], sp[1]
            if kind == "cyl":
                p0, p1, r = sp[2], sp[3], sp[4]
                tag = occ.addCylinder(p0[0], p0[1], p0[2], p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2], r)
            elif kind == "box":
                lo, hi = sp[2], sp[3]
                tag = occ.addBox(lo[0], lo[1], lo[2], hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            else:
                R, r, sweep = sp[2], sp[3], sp[4]
                tag = occ.addTorus(0, 0, 0, R, r, angle=math.radians(sweep))
                occ.rotate([(3, tag)], 0, 0, 0, 0, 0, 1, math.radians((360 - sweep) / 2))
            occ.synchronize()
            gmsh.model.setEntityName(3, tag, name)
        gmsh.write(path)
        return path
    finally:
        gmsh.finalize()


def _scene(specs: list, work: Path, stem: str = "antenna"):
    from ..mesh.isolate import run_isolated
    from .scene import load

    work.mkdir(parents=True, exist_ok=True)
    step = work / f"{stem}.step"
    run_isolated(_write_solids, specs, str(step), timeout=120)
    return load([str(step)], work / "scene")


def _rect(x0, y0, x1, y1, z) -> np.ndarray:
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)


def _disc(radius: float, z: float, n: int = 48) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[radius * np.cos(a), radius * np.sin(a), np.full(n, z)]


def _setup(p: dict, f0: float, **kw) -> WaveSetup:
    span = value(p, "span", 0.25)
    kw.setdefault("points", 7)
    return WaveSetup(f0=f0, span=span, points=int(value(p, "points", kw.pop("points"))),
                     tolerance=value(p, "tolerance", 0.02), port=kw.pop("port"), link={k: p[k] for k in ("tx_power", "sensitivity", "rx_gain", "losses",
                                                                  "environment", "distance", "reliability",
                                                                  "heights", "floors") if k in p}, **kw)


def _substrate(p: dict) -> tuple[float, float, float, str]:
    name = text(p, "dielectric", "fr4")
    d = get_dielectric(name if not p.get("er") else f"er={value(p, 'er', 0)}")
    er = float(d.epsr)
    tand = value(p, "loss_tangent", d.loss_tangent)
    return er, tand, value(p, "height", 1.6, "length"), d.name


# --- wires -----------------------------------------------------------------------------------------------
def dipole(p: dict, work: Path) -> Built:
    check_known(p, KEYS["dipole"], "dipole")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    length = value(p, "length", 0.47 * lam * 1e3, "length")
    dia = value(p, "diameter", max(lam / 300, 0.2e-3) * 1e3, "length")
    gap = value(p, "gap", max(2 * dia / 2, lam / 200) * 1e3, "length")
    arm = (length - gap) / 2
    # value() gives meters and the solids are written in mm
    L, D, G = length * 1e3, dia * 1e3, gap * 1e3
    sc = _scene([("cyl", "arm 1", (0, 0, -L / 2), (0, 0, -G / 2), D / 2),
                 ("cyl", "arm 2", (0, 0, G / 2), (0, 0, L / 2), D / 2)], work)
    s = _setup(p, f0, port=Port((0, 0, -gap / 2), (0, 0, gap / 2)), metals=[b.name for b in sc.bodies])
    return Built("dipole", sc, s, "dipole", dims={"length": length, "diameter": dia, "gap": gap, "arm": arm},
                 ref={"length_wavelengths": length / lam})


def monopole(p: dict, work: Path) -> Built:
    check_known(p, KEYS["monopole"], "monopole")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    length = value(p, "length", 0.24 * lam * 1e3, "length")
    dia = value(p, "diameter", max(lam / 300, 0.2e-3) * 1e3, "length")
    gap = value(p, "gap", max(dia, lam / 200) * 1e3, "length")
    gnd = value(p, "ground", 1.0 * lam * 1e3, "length")            # plate diameter
    sc = _scene([("cyl", "rod", (0, 0, gap * 1e3), (0, 0, (gap + length) * 1e3), dia * 1e3 / 2)], work, "rod")
    s = _setup(p, f0, port=Port((0, 0, 0), (0, 0, gap)), metals=["rod"], sheets={"ground": [_disc(gnd / 2, 0.0, 64)]})
    return Built("monopole", sc, s, "monopole over a ground plate",
                 dims={"length": length, "diameter": dia, "gap": gap, "ground": gnd},
                 ref={"length_wavelengths": length / lam, "ground_wavelengths": gnd / lam})


def loop(p: dict, work: Path) -> Built:
    check_known(p, KEYS["loop"], "loop")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    if p.get("circumference"):
        R = value(p, "circumference", 0, "length") / (2 * math.pi)
    else:
        R = value(p, "radius", lam / (2 * math.pi) * 1e3, "length")
    dia = value(p, "diameter", max(lam / 300, 0.2e-3) * 1e3, "length")
    gap = value(p, "gap", max(dia, lam / 200) * 1e3, "length")
    if 2 * math.pi * R < 3 * dia or gap > 0.5 * R:
        raise ParamError("the loop is too small for its wire and feed gap. Make the radius larger or the wire thinner")
    sweep = 360 - math.degrees(gap / R)
    sc = _scene([("arc", "ring", R * 1e3, dia * 1e3 / 2, sweep)], work, "ring")
    half = gap / R / 2
    a = (R * math.cos(half), -R * math.sin(half) * 1.0, 0.0)
    b = (R * math.cos(half), R * math.sin(half), 0.0)
    s = _setup(p, f0, port=Port(a, b), metals=["ring"])
    return Built("loop", sc, s, "loop", dims={"radius": R, "diameter": dia, "gap": gap},
                 ref={"circumference_wavelengths": 2 * math.pi * R / lam})


# --- printed antennas ----------------------------------------------------------------------------------
def patch_guess(f0: float, er: float, h: float) -> tuple[float, float, float, float]:
    """Width, length, effective permittivity and fringing length of a patch for f0 (Balanis, Antenna Theory,
    equations 14-6 to 14-2): the width of an efficient radiator, the effective permittivity of the line under it,
    the extension of each edge, and the length that is half a guided wavelength less both extensions."""
    W = C0 / (2 * f0) * math.sqrt(2 / (er + 1))
    ee = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * h / W)
    dl = 0.412 * h * (ee + 0.3) * (W / h + 0.264) / ((ee - 0.258) * (W / h + 0.8))
    L = C0 / (2 * f0 * math.sqrt(ee)) - 2 * dl
    return W, L, ee, dl


def patch_resonance(W: float, L: float, er: float, h: float) -> float:
    """The transmission line model's resonant frequency of a W by L patch (Balanis equations 14-1 to 14-7)."""
    ee = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * h / W)
    dl = 0.412 * h * (ee + 0.3) * (W / h + 0.264) / ((ee - 0.258) * (W / h + 0.8))
    return C0 / (2 * (L + 2 * dl) * math.sqrt(ee))


def patch_edge_resistance(W: float, L: float, f: float, z0: float = 50.0) -> float:
    """Resistance at the radiating edge from the slot conductance and the mutual term (Balanis 14-12 to 14-18)."""
    from scipy.special import j0

    k0 = 2 * math.pi * f / C0
    th = np.linspace(1e-6, math.pi - 1e-6, 2001)
    base = np.sin(k0 * W * np.cos(th) / 2) ** 2 / np.cos(th) ** 2 * np.sin(th) ** 3
    g1 = np.trapezoid(base, th) / (120 * math.pi ** 2)
    g12 = np.trapezoid(base * j0(k0 * L * np.sin(th)), th) / (120 * math.pi ** 2)
    return 1 / (2 * (g1 + g12))


def patch(p: dict, work: Path) -> Built:
    check_known(p, KEYS["patch"], "patch")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    er, tand, h, name = _substrate(p)
    W0, L0, ee, dl = patch_guess(f0, er, h)
    W = value(p, "width", W0 * 1e3, "length")
    L = value(p, "length", L0 * 1e3, "length")
    margin = 2 * max(6 * h, lam / 12)
    gw = gl = value(p, "ground", 0, "length") if p.get("ground") else None
    gw, gl = (gw, gl) if gw else (L + margin, W + margin)       # ground plane size along x (length) and y (width)
    rin = patch_edge_resistance(W, L, patch_resonance(W, L, er, h), 50.0)
    z0 = value(p, "impedance", 50)
    inset = L / math.pi * math.acos(math.sqrt(min(z0 / rin, 1.0)))
    feed = value(p, "feed", inset * 1e3, "length")           # distance of the probe from the radiating edge
    if not 0 <= feed <= L / 2:
        raise ParamError("the feed point lies between the patch edge and its center. Give a feed distance between "
                         "0 and half the patch length")
    G, T = [x * 1e3 for x in (gw, gl)], h * 1e3
    sc = _scene([("box", "substrate", (-G[0] / 2, -G[1] / 2, 0), (G[0] / 2, G[1] / 2, T))], work, "substrate")
    x_f = -L / 2 + feed                  # the length runs along x, the radiating edges are at x = +-L/2
    fr = patch_resonance(W, L, er, h)
    s = _setup(p, f0, port=Port((x_f, 0.0, 0.0), (x_f, 0.0, h)), dielectrics={"substrate": (er, tand, name)},
               hints=[0.96 * fr, 1.04 * fr], points=5,
               sheets={"ground": [_rect(-gw / 2, -gl / 2, gw / 2, gl / 2, 0.0)],
                       "patch": [_rect(-L / 2, -W / 2, L / 2, W / 2, h)]})
    return Built("patch", sc, s, "rectangular patch",
                 dims={"width": W, "length": L, "height": h, "er": er, "ground_x": gw, "ground_y": gl, "feed": feed},
                 ref={"transmission_line_Hz": fr, "eps_eff": ee, "edge_resistance_ohm": rin, "inset_m": inset,
                      "loss_tangent": tand})


def invf(p: dict, work: Path) -> Built:
    """An inverted F trace on a board: a ground plane on part of the top copper, a shorted arm running parallel to
    its edge, and a feed stub between the short and the open end. The board's dielectric lies under all of it."""
    check_known(p, KEYS["invf"], "inverted F")
    f0 = value(p, "frequency", 2.4e9, "frequency")
    lam = C0 / f0
    er, tand, h, name = _substrate(p)
    ee = (er + 1) / 2 * 0.85
    arm = value(p, "length", lam / 4 / math.sqrt(ee) * 1e3, "length")
    w = value(p, "width", max(h * 1e3, 1.0) * 0.6, "length")                      # trace width
    hi = 0.12 * lam / math.sqrt(ee) if not p.get("height") else value(p, "height", 0, "length")
    hi = max(hi, 3 * w)                                                              # arm height over the ground
    gx = value(p, "ground", 0.6 * lam * 1e3, "length") if p.get("ground") else 0.5 * lam
    gy = 0.45 * lam
    sx = 0.1 * arm
    fx = value(p, "feed", 0.0, "length") if p.get("feed") else 0.2 * arm        # feed stub distance from the short
    gap = max(w / 2, 0.4e-3)
    board_x = max(gx, arm + 2 * w) + 0.0
    board_y = gy + hi + w + 3e-3
    x0 = -board_x / 2                       # the short is at the left end of the arm
    ax0 = x0 + 0.5 * (board_x - arm) if gx >= arm else x0
    T = h * 1e3
    sc = _scene([("box", "board", (x0 * 1e3, -gy * 1e3, 0), ((x0 + board_x) * 1e3, (board_y - gy) * 1e3, T))], work, "board")
    z = h
    # radiator: short stub (x from ax0), arm along the top, feed stub, drawn as one F shaped polygon
    y_top, y_arm = hi + w, hi
    xs0, xs1 = ax0, ax0 + w
    xf0, xf1 = ax0 + fx, ax0 + fx + w
    xe = ax0 + arm
    ov = 0.5e-3                                          # the short stub reaches into the ground plane
    poly = np.array([(xs0, -ov, z), (xs1, -ov, z), (xs1, y_arm, z), (xf0, y_arm, z), (xf0, gap, z), (xf1, gap, z),
                     (xf1, y_arm, z), (xe, y_arm, z), (xe, y_top, z), (xs0, y_top, z)], float)
    gnd = _rect(x0, -gy, x0 + board_x, 0.0, z)
    s = _setup(p, f0, port=Port(((xf0 + xf1) / 2, 0.0, z), ((xf0 + xf1) / 2, gap, z), width=w),
               dielectrics={"board": (er, tand, name)}, sheets={"ground": [gnd], "radiator": [poly]})
    return Built("invf", sc, s, "inverted F trace antenna",
                 dims={"arm": arm, "width": w, "height": hi, "feed": fx, "board_x": board_x, "board_y": board_y,
                       "thickness": h, "er": er},
                 ref={"quarter_wave_estimate_m": arm})


BUILD = {"dipole": dipole, "monopole": monopole, "loop": loop, "patch": patch, "invf": invf}


def build(kind: str, p: dict, work: Path) -> Built:
    if kind not in BUILD:
        raise ParamError(f"unknown antenna '{kind}'. Use {', '.join(BUILD)}")
    return BUILD[kind](dict(p), Path(work))
