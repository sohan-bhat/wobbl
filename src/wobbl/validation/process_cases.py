"""Validation cases for the manufacturing studies, each against an exact answer or a published measurement."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import geometry as G


def stefan_front(out: Path):
    """An aluminum like melt at 700 C filling a 100 mm bar, its end held at 600 C from t = 0: the freezing front
    stands at 2 lambda sqrt(alpha t) (Neumann's solution), read 4 s in. The latent heat is released over 4 C around
    the 660 C melting point."""
    from ..geometry.cad import prepare
    from ..mesh.mesher import Sizing, build_mesh
    from ..process import solidify as S
    from ..process.analytic import neumann_front, neumann_lambda

    k, rho, c, L, tm, ti, tw, w, when = 200.0, 2700.0, 900.0, 397e3, 660.0, 700.0, 600.0, 4.0, 4.0
    mesh = build_mesh(prepare(G.box(out / "bar.step", 100, 6, 6), out / "geo"), Sizing(h=1.0, order=1))
    mat = S.Material("metal", rho, k, c, L, lambda T: np.clip((tm + w / 2 - np.asarray(T)) / w, 0, 1), tm + w / 2,
                     tm - w / 2)
    held = np.nonzero(mesh.X[:, 0] < 1e-6)[0]
    r = S.solve(S.Model(mesh.X, mesh.tets, np.zeros(len(mesh.tets), int), [mat], np.full(len(mesh.X), ti),
                        fixed=[(held, tw)], duration=when, save_times=[when]))
    x, T = mesh.X[:, 0], r.saved[when]
    bins = np.linspace(0, 100, 401)
    idx = np.digitize(x, bins)
    keep = [i for i in range(1, len(bins)) if (idx == i).any()]
    xs = np.array([x[idx == i].mean() for i in keep])
    ts = np.array([T[idx == i].mean() for i in keep])
    j = int(np.nonzero(ts >= tm)[0][0])
    front = float(np.interp(tm, [ts[j - 1], ts[j]], [xs[j - 1], xs[j]]))
    a = k / (rho * c) * 1e6
    lam = neumann_lambda(k, k, a, a, c, L, tm, ti, tw)
    return front, neumann_front(lam, a, when), {"lambda": round(lam, 5), "nodes": len(mesh.X), "steps": r.steps,
                                               "energy_error": r.energy["error"]}


def chvorinov_slab(out: Path):
    """A 20 mm slab of pure aluminum poured at its melting point between two 40 mm slabs of sand, with the sides
    insulated so heat flows one way: Chvorinov's rule with its mold constant from the semi infinite mold is exact
    for that, with the metal's own resistance and the latent heat's 5 C release band as the only differences."""
    from ..geometry.cad import prepare
    from ..process import casting as C
    from ..process.data import get_alloy, get_mold
    from ..process.regions import build

    al, sand = get_alloy("pure-al"), get_mold("sand")
    prep = prepare(G.box(out / "slab.step", 20, 20, 20), out / "geo")
    rm = build(prep, [], [], (0, 0, 40), 2.0, 5.0)
    sv = C.solidify(C.CastingStudy("slab", film=0.0, contact=1e5), rm, al, sand, al.liquidus, 1500.0)
    t = float(np.nanmax(sv.result.events["solidus"][sv.casting_nodes]))
    B = C.chvorinov_constant(al, sand, al.liquidus, 25.0)
    return t, B * 10.0 ** 2, {"mold_constant_s_per_cm2": round(B * 100, 2), "nodes": len(sv.rm.X),
                             "energy_error": sv.result.energy["error"]}


def disk_fill(out: Path):
    """A power law melt (n 0.35, m 20 kPa s^n) filling a disk 100 mm across and 2 mm thick from a 4 mm gate ring
    in 1 s: the gate pressure when the cavity is 90% full against the radial Hele Shaw solution for the front at
    that point."""
    from ..geometry.cad import prepare
    from ..mesh.mesher import Sizing, build_mesh
    from ..process import hele_shaw as H
    from ..process.analytic import disk_gate_pressure

    R, gap, rg, m, n = 50.0, 2.0, 4.0, 2e4, 0.35

    def annulus(o):
        a = o.addCylinder(0, 0, 0, 0, 0, gap, R)
        b = o.addCylinder(0, 0, -1, 0, 0, gap + 2, rg)
        o.cut([(3, a)], [(3, b)])

    mesh = build_mesh(prepare(G._session(out / "disk.step", annulus), out / "geo"), Sizing(h=4.0, order=1))
    V = math.pi * (R * R - rg * rg) * gap
    gate = np.nonzero(np.linalg.norm(mesh.X[:, :2], axis=1) <= rg + 1e-6)[0]
    res = H.fill(H.Mesh(mesh.X, mesh.tets, np.full(len(mesh.tets), gap)), H.Viscosity(m=m, n=n),
                 H.FillSettings([gate], V / 1.0))
    i = int(np.searchsorted(res.share, 0.9))
    front = math.sqrt(rg ** 2 + res.share[i] * V / (math.pi * gap))
    return float(res.gate_pressure[i]), disk_gate_pressure(m, n, V, gap / 2, front, rg) / 1e6, {
        "front_mm": round(front, 3), "nodes": len(mesh.X), "volume_error": abs(res.injected - res.in_cavity) / V}


def bimetal_strip(out: Path):
    """Two 2 mm layers bonded in a 100 x 10 mm strip, the top one 2200 MPa and locking in 0.1% more shrinkage than
    the 1100 MPa bottom one, solved by the print warp model's CalculiX deck: the curvature against Timoshenko's
    bimetal formula (J. Opt. Soc. Am. 11, 1925)."""
    from ..geometry.cad import prepare
    from ..mesh.mesher import Sizing, build_mesh
    from ..process import warp as W

    L, B, t1, t2, E1, E2, de = 100.0, 10.0, 2.0, 2.0, 2200.0, 1100.0, 1e-3

    def two(o):
        a = o.addBox(0, 0, 0, L, B, t2)
        b = o.addBox(0, 0, t2, L, B, t1)
        o.fragment([(3, a)], [(3, b)])

    mesh = build_mesh(prepare(G._session(out / "strip.step", two), out / "geo"), Sizing(h=2.5))
    top = mesh.X[mesh.tets[:, :4]].mean(axis=1)[:, 2] > t2
    ws = W.solve(mesh.X, mesh.tets, np.where(top, E1, E2), 0.35, np.full(len(mesh.X), de), out / "work",
                 alpha=np.where(top, 1.0, 0.0))
    kxx, kyy, kxy = W.bottom_curvature(mesh.X, ws)
    return kxx * 1000.0, W.timoshenko_bimetal(t1, t2, E1, E2, de) * 1000.0, {
        "nodes": len(mesh.X), "flatness_mm": round(W.measure(mesh.X, ws)["flatness_mm"], 5),
        "net_bed_force_N": float(np.abs(ws.RF_bed[ws.bed].sum(0)).max())}


def abs_plate_warp(out: Path):
    """A solid ABS plate 100 x 20 x 5.5 mm printed in 0.254 mm layers in a 75 C chamber, through the whole print
    warp study, against the 0.675 mm flatness error Armillotta et al. measured on two such plates (RCIM 50, 2018,
    Table 5). This plate is one of the 48 the warp ramp was fitted to, so it shows how well the fit carries
    through the 3D solve rather than an independent prediction. Their bead groove and disturbance allowance,
    0.1 mm + 0.7 x the layer height, is added to the solved warp, as they did."""
    from ..process.data import FLATNESS_OFFSET, FLATNESS_PER_LAYER
    from ..process.printing import run_print_warp
    from ..process.warp import WarpStudy

    part = G.box(out / "plate.step", 100, 20, 5.5)
    o = run_print_warp(WarpStudy(str(part), "abs", chamber_C=75.0, bed_C=75.0, layer_height=0.254), out / "run")
    got = o.values["flatness_mm"] + FLATNESS_OFFSET + FLATNESS_PER_LAYER * 0.254
    return got, 0.675, {"confidence": o.confidence, "solved_warp_mm": round(o.values["flatness_mm"], 4),
                        "time_s": round(o.t_total, 1)}


def _case(*a):
    from .cases import Case
    return Case(*a)


PROCESS_CASES = [
    _case("stefan-front", "Freezing front in a bar held cold at one end", "manufacturing",
          "Neumann's two phase solution (Carslaw and Jaeger 11.2)", "front position after 4 s", "mm", 0.03, True,
          stefan_front),
    _case("chvorinov-slab", "Pure aluminum slab freezing in sand", "manufacturing",
          "Chvorinov's rule with the semi infinite mold constant", "time to freeze", "s", 0.03, True,
          chvorinov_slab),
    _case("disk-fill", "Center gated disk filling with a power law melt", "manufacturing",
          "radial Hele Shaw flow of a power law fluid", "gate pressure at 90% full", "MPa", 0.04, True, disk_fill),
    _case("bimetal-strip", "Two layer strip curling as it shrinks", "manufacturing",
          "Timoshenko's bimetal formula", "curvature", "1/m", 0.02, True, bimetal_strip),
    _case("abs-plate-warp", "ABS plate warping in a 75 C chamber", "manufacturing",
          "Armillotta et al. 2018, measured (fitted data)", "flatness error", "mm", 0.25, False, abs_plate_warp),
]
