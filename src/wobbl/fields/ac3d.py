"""Sound on any CAD: the standing wave modes of an air volume, and the sound a vibrating body sends out.

The air is either a body of the scene (a room or a cavity drawn as a solid) or the cavity a rigid body encloses
(a speaker box, a duct, a housing), which Wobbl cuts out of the solid. Its modes come from an eigenvalue solve of
the Helmholtz equation with sound hard walls (GetDP with SLEPc), checked against Weyl's mode count, a finer mesh
and, when the air is a box, its exact modes.

A body marked as vibrating moves its whole surface outward and back with the given speed, in free air that ends
in an absorbing sphere. The run gives the sound pressure level at listening points over a sweep, the sound power,
and checks the power leaving through the sphere against the power the surface puts in, and repeats the top
frequency on a finer mesh. A vibrating sphere is the exact pulsating sphere of acoustics texts, which the
validation uses.
"""
from __future__ import annotations

import itertools
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from . import plots, pro3d, solve
from .assign import SoundRoles
from .forms import rname
from .pro3d import num
from .scene import Body, Domain, Scene, SceneError, mesh

C_AIR = 343.2
RHO_AIR = 1.204
P_REF = 20e-6


# --- cavities --------------------------------------------------------------------------------------
def cavities(scene: Scene, walls: list[str], path: str | Path) -> Scene:
    """The air a set of rigid bodies encloses, as a new scene of cavity bodies."""
    from ..mesh.isolate import run_isolated

    d = run_isolated(_cavities, scene.brep, [b.index for b in scene.bodies if b.name in walls], str(path), timeout=300)
    if not d["bodies"]:
        raise SceneError(f"'{', '.join(walls)}' encloses no air. Give the air volume itself as a body marked 'air', "
                         "or close the enclosure")
    return Scene([Body(**b) for b in d["bodies"]], d["brep"], list(scene.notes))


def _cavities(brep: str, indices: list[int], out: str) -> dict:
    from .scene import _session

    gmsh = _session()
    try:
        occ = gmsh.model.occ
        ents = occ.importShapes(brep, highestDimOnly=True)
        vols = [t for d, t in ents if d == 3]
        keep = [vols[i - 1] for i in indices]
        occ.remove([(3, v) for v in vols if v not in keep], recursive=True)
        occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        pad = 0.05 * np.linalg.norm(bb[3:] - bb[:3])
        lo, hi = bb[:3] - pad, bb[3:] + pad
        box = occ.addBox(*lo, *(hi - lo))
        pieces, _ = occ.cut([(3, box)], [(3, v) for v in keep])
        occ.synchronize()
        meta = []
        for d, t in pieces:
            pb = np.array(gmsh.model.getBoundingBox(3, t))
            if np.any(np.abs(pb[:3] - lo) < 1e-3 * pad) or np.any(np.abs(pb[3:] - hi) < 1e-3 * pad):
                occ.remove([(3, t)], recursive=True)
                continue
            surf = [abs(s) for _, s in gmsh.model.getBoundary([(3, t)], oriented=False)]
            meta.append((t, {"name": f"cavity {len(meta) + 1}", "index": len(meta) + 1, "file": "cavity",
                             "volume": occ.getMass(3, t), "area": sum(occ.getMass(2, s) for s in surf),
                             "bbox": tuple(pb), "center": tuple(occ.getCenterOfMass(3, t)), "hole_axis": None}))
        occ.synchronize()
        gmsh.write(out)
        # the written file holds only the cavities, in this order
        return {"bodies": [m for _, m in meta], "brep": out}
    finally:
        gmsh.finalize()


def box_modes(L: tuple, fmax: float, c: float = C_AIR) -> list[float]:
    n = [int(2 * fmax * li / c) + 1 for li in L]
    out = []
    for idx in itertools.product(*(range(k + 1) for k in n)):
        if any(idx):
            f = c / 2 * math.sqrt(sum((i / li) ** 2 for i, li in zip(idx, L)))
            if f <= fmax:
                out.append(f)
    return sorted(out)


def weyl(V: float, S: float, Ltot: float, f: float, c: float = C_AIR) -> float:
    return 4 * math.pi * V * f ** 3 / (3 * c ** 3) + math.pi * S * f ** 2 / (4 * c ** 2) + Ltot * f / (8 * c)


def _is_box(b: Body) -> tuple | None:
    ext = tuple(b.bbox[i + 3] - b.bbox[i] for i in range(3))
    return ext if abs(b.volume - ext[0] * ext[1] * ext[2]) < 2e-3 * b.volume else None


# --- problems ----------------------------------------------------------------------------------------
def eigen_problem(msh, nev: int, shift_hz: float, pl: dict, shapes: int) -> str:
    vols = [n for n, t in msh.tags.items() if t < 100]
    out = [pro3d.groups(msh.tags, {"Dom": vols})]
    out.append(f"Function {{\n  c = {num(C_AIR)};\n}}\n")
    out.append("Jacobian {\n  { Name Vol; Case { { Region All; Jacobian Vol; } } }\n}\n")
    out.append(pro3d.INTEGRATION)
    out.append("FunctionSpace {\n  { Name Hp; Type Form0; BasisFunction {\n"
               "      { Name sn; NameOfCoef pn; Function BF_Node; Support Dom; Entity NodesOf[All]; }\n"
               "      { Name s2; NameOfCoef p2; Function BF_Node_2E; Support Dom; Entity EdgesOf[All]; } }\n  }\n}\n")
    out.append("Formulation {\n  { Name H; Type FemEquation;\n    Quantity { { Name p; Type Local; NameOfSpace Hp; } }\n"
               "    Equation {\n"
               "      Integral { [ Dof{d p}, {d p} ]; In Dom; Jacobian Vol; Integration I2; }\n"
               "      Integral { DtDtDof [ 1 / c^2 * Dof{p}, {p} ]; In Dom; Jacobian Vol; Integration I2; }\n"
               "    }\n  }\n}\n")
    out.append("Resolution {\n  { Name H; System { { Name A; NameOfFormulation H; Type Complex; } }\n"
               f"    Operation {{ GenerateSeparate[A]; EigenSolve[A, {nev}, {num((2 * math.pi * shift_hz) ** 2)}, 0]; "
               "SaveSolutions[A]; }\n  }\n}\n")
    out.append("PostProcessing {\n  { Name H; NameOfFormulation H; Quantity {\n"
               "      { Name p; Value { Local { [ {p} ]; In Dom; Jacobian Vol; } } } } }\n}\n")
    po = [pro3d.print_plane("p", pl, f"mode{i}.txt", f" TimeStep {i},") for i in range(shapes)]
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing H; Operation {\n    " + "\n    ".join(po or [""]) +
               "\n  } }\n}\n")
    return "".join(out)


def radiation_problem(msh, f: float, sources: dict[str, float], listen: list, pl: dict | None) -> str:
    vols = [n for n, t in msh.tags.items() if t < 100]
    srcs = [n for n in sources if n in msh.tags]
    out = [pro3d.groups(msh.tags, {"Dom": vols, "Src": srcs, "Out": ["outer"], "All_": vols + srcs + ["outer"]})]
    k = 2 * math.pi * f / C_AIR
    w = 2 * math.pi * f
    fn = [f"  k = {num(k)};", f"  w = {num(w)};", f"  rho = {num(RHO_AIR)};", f"  c = {num(C_AIR)};",
          f"  Rs = {num(msh.radius)};"]
    fn += [f"  vn[{rname(n)}] = {num(sources[n])};" for n in srcs]
    out.append("Function {\n" + "\n".join(fn) + "\n}\n")
    out.append("Jacobian {\n  { Name Vol; Case { { Region All; Jacobian Vol; } } }\n"
               "  { Name Sur; Case { { Region All; Jacobian Sur; } } }\n}\n")
    out.append(pro3d.INTEGRATION)
    out.append("FunctionSpace {\n  { Name Hp; Type Form0; BasisFunction {\n"
               "      { Name sn; NameOfCoef pn; Function BF_Node; Support All_; Entity NodesOf[All]; }\n"
               "      { Name s2; NameOfCoef p2; Function BF_Node_2E; Support All_; Entity EdgesOf[All]; } }\n  }\n}\n")
    # e^(j w t), outgoing waves e^(-j k r); the surface moves outward into the air with speed vn
    out.append("Formulation {\n  { Name H; Type FemEquation;\n    Quantity { { Name p; Type Local; NameOfSpace Hp; } }\n"
               "    Equation {\n"
               "      Integral { [ Dof{d p}, {d p} ]; In Dom; Jacobian Vol; Integration I2; }\n"
               "      Integral { [ -k^2 * Dof{p}, {p} ]; In Dom; Jacobian Vol; Integration I2; }\n"
               "      Integral { [ Complex[1 / Rs, k] * Dof{p}, {p} ]; In Out; Jacobian Sur; Integration I2; }\n"
               "      Integral { [ -Complex[0, w * rho] * vn[], {p} ]; In Src; Jacobian Sur; Integration I2; }\n"
               "    }\n  }\n}\n")
    out.append(f"Resolution {{\n  {{ Name H; System {{ {{ Name A; NameOfFormulation H; Type Complex; Frequency {num(f)}; }} }}\n"
               "    Operation { Generate[A]; Solve[A]; SaveSolution[A]; }\n  }\n}\n")
    out.append("PostProcessing {\n  { Name H; NameOfFormulation H; Quantity {\n"
               "      { Name p; Value { Local { [ {p} ]; In All_; Jacobian Vol; } } }\n"
               "      { Name Pin; Value { Integral { [ 0.5 * Re[ {p} * Conj[vn[]] ] ]; In Src; Jacobian Sur; "
               "Integration I2; } } }\n"
               "      { Name Pout; Value { Integral { [ 0.5 * SquNorm[{p}] / (rho * c) ]; In Out; Jacobian Sur; "
               "Integration I2; } } }\n  } }\n}\n")
    po = ['Print[ Pin[Src], OnGlobal, Format Table, File "Pin.txt" ];',
          'Print[ Pout[Out], OnGlobal, Format Table, File "Pout.txt" ];']
    for i, x in enumerate(listen):
        po.append(f'Print[ p, OnPoint {{{num(x[0])}, {num(x[1])}, {num(x[2])}}}, Format Table, File "lp{i}.txt" ];')
    if pl:
        po.append(pro3d.print_plane("p", pl, "slice_p.txt"))
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing H; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)


# --- runs --------------------------------------------------------------------------------------------
def run(scene: Scene, roles: SoundRoles, out_dir, question: str = "", quality: str = "normal", fmax: float = 0.0,
        log=lambda s, d="": None, title: str = "", listen: list | None = None, keep: bool = False):
    if roles.sources:
        return run_radiation(scene, roles, out_dir, question, quality, fmax, log, title, listen or [], keep)
    return run_modes(scene, roles, out_dir, question, quality, fmax, log, title, keep)


def run_modes(scene: Scene, roles: SoundRoles, out_dir, question="", quality="normal", fmax=0.0,
              log=lambda s, d="": None, title="", keep=False):
    from ..outcome_generic import GenericOutcome

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work" / "solve"      # the scene the caller loaded may sit beside it in out/work
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if roles.walls and not roles.air:
        log("mesh", "finding the air the walls enclose")
        air_scene = cavities(scene, roles.walls, work / "cavity.brep")
    else:
        air_scene = Scene([b for b in scene.bodies if b.name in roles.air] or scene.bodies[:1], scene.brep,
                          list(scene.notes))
        air_scene = _subset(scene, [b.name for b in air_scene.bodies], work / "air.brep")
    air = air_scene.bodies
    V = sum(b.volume for b in air)
    S = sum(b.area for b in air)
    Ltot = sum(4 * sum(b.bbox[i + 3] - b.bbox[i] for i in range(3)) for b in air)
    if not fmax:
        fmax = (3 * 30 * C_AIR ** 3 / (4 * math.pi * V)) ** (1 / 3)      # about 30 modes
    n_exp = weyl(V, S, Ltot, fmax)
    nev = int(min(max(1.4 * n_exp + 6, 10), 150))
    h = C_AIR / fmax / 5.0
    ks = {"fast": (1.3,), "normal": (1.0, 0.75), "fine": (0.8, 0.6)}[quality]
    pl = pro3d.slice_plane(air_scene.bbox, margin=0.0)
    bb = air_scene.bbox
    # off the middle, where half of all modes have a nodal plane and would show nothing
    pl["at"] = bb[pl["ik"]] + 0.3 * (bb[pl["ik"] + 3] - bb[pl["ik"]])
    levels = []
    for j, k in enumerate(ks):
        log("mesh", f"mesh {j + 1} of {len(ks)}")
        msh = mesh(air_scene, [], [b.name for b in air], work / f"l{j}" / "m.msh", k=k,
                   domain=Domain("none", max_size=h), sizes={b.name: h for b in air})
        log("solve", f"modes up to {fmt(fmax)} Hz ({msh.n_nodes:,} nodes)")
        lg = solve.run(eigen_problem(msh, nev, 1.0, pl, 7 if j == len(ks) - 1 else 0), msh.path, work / f"l{j}",
                       resolution="H", verbosity=4, timeout=1800)
        w2 = [z.real for z in solve.eigenvalues(lg)]
        fr = sorted(math.sqrt(max(v, 0.0)) / (2 * math.pi) for v in w2)
        levels.append({"msh": msh, "f": fr, "dir": work / f"l{j}"})
    fine = levels[-1]
    rigid = [x for x in fine["f"] if x < 1e-3 * fmax]
    modes = [x for x in fine["f"] if 1e-3 * fmax <= x <= fmax]
    if not modes:
        raise ValueError(f"no modes below {fmt(fmax)} Hz; raise --fmax")
    checks = []
    if len(levels) > 1:
        c_ = [x for x in levels[0]["f"] if 1e-3 * fmax <= x <= fmax]
        n = min(len(c_), len(modes))
        d = max(abs(a - b) / b for a, b in zip(c_[:n], modes[:n])) if n else 0.0
        checks.append(Check("mesh convergence", "pass" if d < 0.005 else ("warn" if d < 0.02 else "fail"),
                            f"the first {n} modes moved at most {100 * d:.2g}% on the finer mesh"))
    dn = abs(len(modes) - n_exp) / max(n_exp, 1)
    checks.append(Check("mode count", "pass" if dn < 0.25 or abs(len(modes) - n_exp) <= 3 else "warn",
                        f"{len(modes)} modes below {fmt(fmax)} Hz, where Weyl's formula for this volume and surface "
                        f"expects about {n_exp:.0f}"))
    exact = []
    if len(air) == 1 and _is_box(air[0]):
        exact = box_modes(_is_box(air[0]), fmax * 1.02)
        n = min(len(exact), len(modes))
        worst = max(abs(modes[i] - exact[i]) / exact[i] for i in range(n)) if n else 0.0
        checks.append(Check("closed form", "pass" if worst < 0.005 else ("warn" if worst < 0.02 else "fail"),
                            f"the air is a box, and its first {n} modes match the exact box modes to "
                            f"{100 * worst:.2g}%"))
    lvl, reasons = confidence(checks)
    # crowded modes matter where they are still few and far between, among the lowest ones
    low = modes[:15]
    close = [(a, b) for a, b in zip(low, low[1:]) if (b - a) / a < 0.02]
    head = (f"The air has {len(modes)} standing wave modes below {fmt(fmax)} Hz, the lowest at {fmt(modes[0])} Hz"
            + (f" and the next at {fmt(modes[1])} Hz" if len(modes) > 1 else "") + ".")
    if close:
        head += (f" Among the lowest 15, {len(close)} pair{'s' if len(close) > 1 else ''} sit within 2% of each other "
                 f"(first near {fmt(close[0][0])} Hz), where a tone gets reinforced.")
    shapes = []
    for i in range(1, 7):
        g = pro3d.read_plane(fine["dir"] / f"mode{i}.txt", pl, comps=1)
        if np.isfinite(g).any():
            shapes.append((fine["f"][i] if i < len(fine["f"]) else 0.0, g))
    images = {}
    if shapes:
        images["modes"] = _panel(out / "images" / "modes.png", shapes, pl)
    images["spectrum"] = plots.curve(out / "images" / "spectrum.png", list(range(1, len(modes) + 1)),
                                     {"field solve": modes, **({"exact box": exact[:len(modes)]} if exact else {})},
                                     "mode number", "frequency, Hz")
    values = {"modes_Hz": modes, "exact_Hz": exact[:len(modes)], "weyl_count": n_exp, "volume_m3": V,
              "close_pairs": close, "rigid_body_modes": len(rigid), "nodes": [lv["msh"].n_nodes for lv in levels],
              "_figures": [["modes", "The first mode shapes on a slice"], ["spectrum", "Mode frequencies"]]}
    metrics = [("lowest mode", f"{fmt(modes[0])} Hz"), ("modes", f"{len(modes)} below {fmt(fmax)} Hz"),
               ("air volume", f"{fmt(V * 1e3)} L"), ("close pairs", str(len(close)))]
    assumptions = ["Sound hard walls with no absorption, air at 20 C (343 m/s). Real walls damp the modes, and the "
                   "frequencies hold to about a percent."]
    if roles.walls and not roles.air:
        assumptions.append(f"The air is the cavity inside '{', '.join(roles.walls)}', taken from the CAD.")
    o = GenericOutcome("sound", title or "air", question, "ok", head, metrics, lvl, reasons, checks,
                       assumptions + scene.notes, values, images, [], solver=f"GetDP {solve.getdp().version()} with "
                       "SLEPc and gmsh", t_total=time.time() - t0)
    _modes_playback(out, shapes, pl, air_scene, modes)
    _modes_view(out, o, fine["msh"], [b.name for b in air], shapes, pl, title or "air", question)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _subset(scene: Scene, names: list[str], path) -> Scene:
    """A scene of only some bodies, saved on its own, so meshing sees nothing else."""
    from ..mesh.isolate import run_isolated

    idx = [b.index for b in scene.bodies if b.name in names]
    run_isolated(_keep, scene.brep, idx, str(path), timeout=120)
    bodies = [b for b in scene.bodies if b.name in names]
    from dataclasses import replace
    return Scene([replace(b, index=i + 1) for i, b in enumerate(bodies)], str(path), list(scene.notes))


def _keep(brep: str, idx: list[int], out: str) -> str:
    from .scene import _session

    gmsh = _session()
    try:
        ents = gmsh.model.occ.importShapes(brep, highestDimOnly=True)
        vols = [t for d, t in ents if d == 3]
        gmsh.model.occ.remove([(3, v) for i, v in enumerate(vols) if i + 1 not in idx], recursive=True)
        gmsh.model.occ.synchronize()
        gmsh.write(out)
        return out
    finally:
        gmsh.finalize()


def run_radiation(scene: Scene, roles: SoundRoles, out_dir, question="", quality="normal", fmax=0.0,
                  log=lambda s, d="": None, title="", listen=(), keep=False):
    from ..outcome_generic import GenericOutcome

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work" / "solve"      # the scene the caller loaded may sit beside it in out/work
    shutil.rmtree(work, ignore_errors=True)
    fmax = fmax or 1000.0
    fmin = fmax / 10
    freqs = list(np.geomspace(fmin, fmax, 8 if quality != "fast" else 5))
    srcs = list(roles.sources)
    holes = [b.name for b in scene.bodies if b.name in srcs or b.name in roles.walls]
    c = scene.center
    pts = []
    for t in listen:
        pts.append(np.array([float(x) for x in t.split(",")]) * 1e-3)
    if not pts:
        pts = [c + np.array([0, 0, scene.radius + 1.0])]
    # the absorbing sphere: a wavelength of the top frequency beyond the source, at least two and a half times its
    # size. Its first order condition (jk + 1/r) is exact for the breathing part of any source.
    R = max(2.5 * scene.radius, scene.radius + C_AIR / fmax)
    probe, scale = [], []
    for p in pts:
        r = float(np.linalg.norm(p - c))
        if r <= 0.85 * R:
            probe.append(p)
            scale.append(1.0)
        else:          # farther out the sound falls off as 1/r along the same direction
            probe.append(c + (p - c) * 0.85 * R / r)
            scale.append(0.85 * R / r)
    h_air = C_AIR / fmax / 5
    pl = pro3d.slice_plane(tuple(c - R / 1.6) + tuple(c + R / 1.6), margin=0.0)
    rows, msh = [], None
    log("mesh", "air around the source")
    msh = mesh(scene, holes, [], work / "m.msh", k=1.0 if quality != "fast" else 1.3,
               domain=Domain("sphere", radius=R, max_size=h_air))
    for i, f in enumerate(freqs):
        log("solve", f"{fmt(f)} Hz ({i + 1} of {len(freqs)})")
        d = work / f"f{i}"
        solve.run(radiation_problem(msh, f, roles.sources, probe, pl if i == len(freqs) - 1 else None), msh.path, d,
                  resolution="H")
        lp = [solve.read_point(d, f"lp{j}.txt") for j in range(len(pts))]
        rows.append({"Pin": float(solve.read_global(d, "Pin.txt")[-1]), "Pout": float(solve.read_global(d, "Pout.txt")[-1]),
                     "p": [sc * abs(complex(x[0], x[1] if len(x) > 1 else 0)) for x, sc in zip(lp, scale)], "dir": d})
    checks = []
    bal = max(abs(r["Pout"] - r["Pin"]) / max(abs(r["Pin"]), 1e-30) for r in rows)
    checks.append(Check("power balance", "pass" if bal < 0.05 else ("warn" if bal < 0.12 else "fail"),
                        f"the sound power leaving through the outer sphere matches what the surface puts in to "
                        f"{100 * bal:.2g}% at every frequency"))
    if quality != "fast":
        log("verify", "the top frequency on a finer mesh")
        m2 = mesh(scene, holes, [], work / "m2.msh", k=0.75, domain=Domain("sphere", radius=R, max_size=h_air))
        solve.run(radiation_problem(m2, freqs[-1], roles.sources, probe, None), m2.path, work / "fine", resolution="H")
        p2 = scale[0] * abs(complex(*solve.read_point(work / "fine", "lp0.txt")[:2]))
        d = abs(p2 - rows[-1]["p"][0]) / max(p2, 1e-30)
        checks.append(Check("mesh convergence", "pass" if d < 0.03 else ("warn" if d < 0.08 else "fail"),
                            f"the pressure at the first listening point moved {100 * d:.2g}% on a finer mesh"))
    lvl, reasons = confidence(checks)
    spl = np.array([[20 * math.log10(max(p / math.sqrt(2), 1e-12) / P_REF) for p in r["p"]] for r in rows])
    imax = int(np.argmax(spl[:, 0]))
    head = (f"At {fmt(np.linalg.norm(pts[0] - c))} m the sound reaches {spl[imax, 0]:.1f} dB SPL near "
            f"{fmt(freqs[imax])} Hz, with {fmt(rows[imax]['Pin'] * 1e3)} mW of sound power, for "
            + ", ".join(f"'{n}' moving at {fmt(v * 1e3)} mm/s" for n, v in roles.sources.items()) + ".")
    images = {"spl": plots.curve(out / "images" / "spl.png", freqs, {f"point {j + 1}": spl[:, j] for j in range(len(pts))},
                                 "frequency, Hz", "SPL, dB", logx=True)}
    g = pro3d.read_plane(rows[-1]["dir"] / "slice_p.txt", pl, comps=2)
    if np.isfinite(g).any():
        images["field"] = plots.grid_map(out / "images" / "field.png", g[..., 0], pl, "pressure, Pa",
                                         f"sound pressure at {fmt(freqs[-1])} Hz", cmap="RdBu_r", signed=True)
    values = {"freqs_Hz": freqs, "spl_dB": spl.tolist(), "power_W": [r["Pin"] for r in rows],
              "listen_m": [p.tolist() for p in pts], "outer_radius_m": R,
              "_figures": [["spl", "Sound pressure level"], ["field", "Pressure on a slice"]]}
    metrics = [("peak SPL", f"{spl[imax, 0]:.1f} dB"), ("at", f"{fmt(freqs[imax])} Hz"),
               ("sound power", f"{fmt(rows[imax]['Pin'] * 1e3)} mW"), ("mesh", f"{msh.n_nodes:,} nodes")]
    assumptions = ["Free field in air at 20 C. The surface of each vibrating body moves outward and back with the "
                   "same speed everywhere (a breathing source), and the outer sphere absorbs outgoing sound to "
                   "first order.", "Sound pressure level is the RMS pressure against 20 uPa."]
    if any(x < 1 for x in scale):
        assumptions.append(f"Listening points beyond {fmt(0.85 * R)} m are reached by the 1/r fall off of sound from "
                           "the solved field along the same direction.")
    o = GenericOutcome("sound", title or "source", question, "ok", head, metrics, lvl, reasons, checks,
                       assumptions + scene.notes, values, images, [], solver=f"GetDP {solve.getdp().version()} with gmsh",
                       t_total=time.time() - t0)
    if np.isfinite(g).any():
        from . import playback as pb
        m = float(np.nanpercentile(np.abs(g), 99))
        pb.write(out, "sound-radiation", {"p_re, p_im": "pressure phasor on the slice at the top frequency, Pa, packed; "
                                                        "play Re(p e^(j w t))",
                                          "slice": "the plane and its ranges in mm", "spl": "SPL at each listening "
                                          "point over frequency (x in Hz, dB)"},
                 {"frequency_Hz": freqs[-1], "slice": _pl_mm(pl), "p_re": pb.pack(g[..., 0], -m, m),
                  "p_im": pb.pack(g[..., 1], -m, m), "spl": pb.series(freqs, {f"p{j}": spl[:, j] for j in range(len(pts))})})
        _wave_view(out, o, msh, holes, roles, g, pl, freqs[-1], m, title or "source", question)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _modes_view(out, o, msh, air: list[str], shapes, pl, title, question):
    """Each of the first modes breathing on the slice for one period, inside the air drawn from its mesh."""
    try:
        from ..viz import Scene

        from . import view as V
    except ImportError:
        return None
    if not shapes:
        return None
    per = 12
    frames, t, freq = [], [], []
    for k, (f, g) in enumerate(shapes):
        m = np.nanmax(np.abs(g)) or 1.0
        for j in range(per):
            frames.append(g / m * math.cos(2 * math.pi * j / per))
            t.append(k + j / per)
            freq.append(f)
    sc = Scene(f"{Path(title).stem} standing waves", domain="sound", question=question, length="mm")
    sc.timeline(t, unit="", label="mode", seconds=3.0 * len(shapes), between="exact",
                note="each mode plays one period of its standing wave, its shape times cos(2 pi f t)")
    V.draw_bodies(sc, msh, {n: "air" for n in air}, {n: 0.18 for n in air})
    V.slice_mesh(sc, pl, np.array(frames), "pressure", "relative", cmap="RdBu", frames=True, rng=(-1, 1))
    sc.readout("mode frequency", freq, unit="Hz")
    word, tn = V.tone(o.verdict)
    sc.verdict(word, o.headline, tn)
    sc.note("Mode shapes have no absolute size; red and blue are opposite phase.")
    return sc.save(out)


def _wave_view(out, o, msh, holes, roles, g, pl, f, m, title, question):
    """The sound leaving the vibrating body over one period, from the solved pressure phasor on the slice."""
    try:
        from ..viz import Scene

        from . import view as V
    except ImportError:
        return None
    n = 16
    ph = 2 * math.pi * np.arange(n) / n
    frames = np.array([g[..., 0] * math.cos(a) - g[..., 1] * math.sin(a) for a in ph])
    sc = Scene(f"{Path(title).stem} sound", domain="sound", question=question, length="mm")
    sc.timeline(ph / (2 * math.pi * f) * 1e3, unit="ms", label="time", seconds=4.0, between="exact",
                note=f"one period at {f:.4g} Hz, the real part of the solved pressure phasor")
    V.draw_bodies(sc, msh, {h: ("source" if h in roles.sources else "wall") for h in holes})
    V.slice_mesh(sc, pl, frames, "sound pressure", "Pa", cmap="RdBu", frames=True, rng=(-m, m))
    word, tn = V.tone(o.verdict)
    sc.verdict(word, o.headline, tn)
    return sc.save(out)


def _pl_mm(pl: dict) -> dict:
    return {k: (round(v * 1e3, 4) if k in ("at", "u0", "u1", "v0", "v1") else v) for k, v in pl.items()
            if k not in ("iu", "iv", "ik")}


def _panel(path, shapes, pl):
    plt = plots._mpl()
    from ..report.render import BG, FG

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4), dpi=110, facecolor=BG)
    ext = np.array([pl["u0"], pl["u1"], pl["v0"], pl["v1"]]) * 1e3
    for i, ax in enumerate(axes.ravel()):
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(shapes):
            ax.axis("off")
            continue
        f, g = shapes[i]
        m = np.nanmax(np.abs(g)) or 1.0
        ax.imshow(np.ma.masked_invalid(g / m), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1,
                  aspect="equal", interpolation="bilinear")
        ax.set_title(f"{fmt(f)} Hz", color=FG, fontsize=9)
    fig.suptitle(f"pressure of the first modes on the plane {pl['normal']} = {fmt(pl['at'] * 1e3)} mm "
                 "(red and blue are opposite phase)", color=FG, fontsize=10)
    fig.tight_layout()
    return plots.save(fig, path)


def _modes_playback(out, shapes, pl, air_scene, modes):
    from . import playback as pb

    if not shapes:
        return None
    return pb.write(out, "sound-modes-3d", {
        "modes": "mode shapes on the slice, pressure packed on a range symmetric about zero; play shape times "
                 "cos(2 pi f t)", "modes_Hz": "every mode found, Hz", "slice": "the plane and its ranges in mm",
        "air": "each air volume's box in mm"},
        {"slice": _pl_mm(pl), "modes": [{"frequency_Hz": round(f, 4), "shape": pb.pack_signed(g)} for f, g in shapes],
         "modes_Hz": [round(f, 4) for f in modes],
         "air": [[round(1e3 * x, 3) for x in b.bbox] for b in air_scene.bodies]})
