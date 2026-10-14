"""Electrostatics on any CAD: capacitance matrix, peak field and breakdown margin.

Conductors are cut out of the mesh and held at their voltages (or left floating with no net charge), every
other body is a dielectric, and the air ends in a shell mapped to infinity, which is the zero of potential.
One solve at the operating voltages gives the field. One more per conductor, at 1 V with the others at 0,
gives the Maxwell capacitance matrix. The run checks that the charges balance (with the charge at infinity),
that the matrix is symmetric, that the energy of the field matches half V C V, and that the capacitance and
the peak field hold still on a finer mesh. A peak that keeps growing as the mesh refines sits on a sharp
edge, where the real field depends on the edge radius, and the run says so.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from . import plots, pro3d, solve
from .forms import EPS0, rname
from .scene import Domain, Scene, mesh

AIR_STRENGTH = 3.0e6        # V/m, uniform field breakdown of air at sea level across gaps of several mm


@dataclass
class ElecSetup:
    conductors: dict[str, float | None]                  # body: volts, None for floating
    dielectrics: dict[str, tuple[float, float, str]] = field(default_factory=dict)   # body: (er, strength, name)
    target: float = 2.0
    air_strength: float = AIR_STRENGTH


def problem(msh, s: ElecSetup, volts: dict[str, float], fields: bool, pl: dict | None, order: int = 1) -> str:
    conds = [c for c in s.conductors if c in msh.tags]
    fixed = [c for c in conds if volts.get(c) is not None]
    floating = [c for c in conds if volts.get(c) is None]
    vols = [n for n, t in msh.tags.items() if t < 100]
    near = {f"Near_{rname(n)[2:]}": n for n in vols if n != "shell"}
    out = [pro3d.groups(msh.tags, {"Dom": vols, "Cond": conds + ["outer"], "Fix": fixed + ["outer"],
                                   "Flo": floating})]
    out[0] = out[0].replace("}\n", "".join(f"  {g} = ElementsOf[{rname(n)}, OnOneSideOf Cond];\n"
                                           for g, n in near.items()) + "}\n", 1)
    fn = ["Function {", f"  eps0 = {EPS0!r};"]
    for n in vols:
        er = s.dielectrics.get(n, (1.0,))[0]
        fn.append(f"  epsr[{rname(n)}] = {pro3d.num(er)};")
    out.append("\n".join(fn) + "\n}\n")
    cons = "".join(f"      {{ Region {rname(c)}; Value {pro3d.num(volts[c])}; }}\n" for c in fixed)
    cons += f"      {{ Region {rname('outer')}; Value 0; }}\n"
    chg = "".join(f"      {{ Region {rname(c)}; Value 0; }}\n" for c in floating)
    out.append("Constraint {\n  { Name Volt; Case {\n" + cons + "  } }\n  { Name Charge; Case {\n" + chg + "  } }\n}\n")
    out.append(pro3d.jacobian(msh, True))
    out.append(pro3d.INTEGRATION)
    out.append("FunctionSpace {\n  { Name Hv; Type Form0; BasisFunction {\n"
               "      { Name sn; NameOfCoef vn; Function BF_Node; Support Dom; Entity NodesOf[All, Not Cond]; }\n"
               "      { Name sf; NameOfCoef vf; Function BF_GroupOfNodes; Support Dom; Entity GroupsOfNodesOf[Cond]; }\n"
               + ("      { Name s2; NameOfCoef v2; Function BF_Node_2E; Support Dom; Entity EdgesOf[All, Not Cond]; }\n"
                  if order == 2 else "") +
               "    }\n    GlobalQuantity { { Name U; Type AliasOf; NameOfCoef vf; } "
               "{ Name Q; Type AssociatedWith; NameOfCoef vf; } }\n"
               "    Constraint { { NameOfCoef U; EntityType GroupsOfNodesOf; NameOfConstraint Volt; }\n"
               "      { NameOfCoef Q; EntityType GroupsOfNodesOf; NameOfConstraint Charge; } }\n  }\n}\n")
    out.append("Formulation {\n  { Name E; Type FemEquation;\n    Quantity {\n"
               "      { Name v; Type Local; NameOfSpace Hv; }\n"
               "      { Name U; Type Global; NameOfSpace Hv [U]; }\n"
               "      { Name Q; Type Global; NameOfSpace Hv [Q]; }\n    }\n    Equation {\n"
               "      Integral { [ eps0 * epsr[] * Dof{d v}, {d v} ]; In Dom; Jacobian Vol; Integration I2; }\n"
               "      GlobalTerm { [ -Dof{Q}, {U} ]; In Cond; }\n    }\n  }\n}\n")
    out.append("Resolution {\n  { Name E; System { { Name A; NameOfFormulation E; } }\n"
               "    Operation { Generate[A]; Solve[A]; SaveSolution[A]; }\n  }\n}\n")
    out.append("PostProcessing {\n  { Name E; NameOfFormulation E; Quantity {\n"
               "      { Name v; Value { Local { [ {v} ]; In Dom; Jacobian Vol; } } }\n"
               "      { Name en; Value { Local { [ Norm[{d v}] ]; In Dom; Jacobian Vol; } } }\n"
               "      { Name W; Value { Integral { [ 0.5 * eps0 * epsr[] * SquNorm[{d v}] ]; In Dom; Jacobian Vol; "
               "Integration I1; } } }\n"
               "      { Name Q; Value { Term { [ {Q} ]; In Cond; } } }\n"
               "      { Name U; Value { Term { [ {U} ]; In Cond; } } }\n  } }\n}\n")
    po = ['Print[ W[Dom], OnGlobal, Format Table, File "W.txt" ];']
    for c in conds + ["outer"]:
        po.append(f'Print[ Q, OnRegion {rname(c)}, Format Table, File "Q_{rname(c)}.txt" ];')
        po.append(f'Print[ U, OnRegion {rname(c)}, Format Table, File "U_{rname(c)}.txt" ];')
    if fields:
        for g, n in near.items():
            po.append(f'Print[ en, OnElementsOf {g}, Format SimpleTable, File "peak_{rname(n)}.txt" ];')
        if pl:
            po.append(pro3d.print_plane("en", pl, "slice_e.txt"))
            po.append(pro3d.print_plane("v", pl, "slice_v.txt"))
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing E; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)


def peak_nodes(path) -> tuple[float, tuple] | None:
    """The largest value in a GetDP element table (each row lists its corners' x y z, then a value per corner)
    and where it is."""
    try:
        a = np.loadtxt(path, ndmin=2)
    except (OSError, ValueError):
        return None
    if not a.size:
        return None
    nv = a.shape[1] // 4
    vals = a[:, 3 * nv:4 * nv]
    i, j = np.unravel_index(int(np.argmax(vals)), vals.shape)
    return float(vals[i, j]), tuple(float(x) for x in a[i, 3 * j:3 * j + 3])


def _solve(msh, s: ElecSetup, volts: dict, d: Path, fields: bool = False, pl: dict | None = None) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    solve.run(problem(msh, s, volts, fields, pl), msh.path, d, resolution="E")
    conds = [c for c in s.conductors if c in msh.tags]
    out = {"W": float(solve.read_global(d, "W.txt")[0]),
           "Q": {c: float(solve.read_global(d, f"Q_{rname(c)}.txt")[-1]) for c in conds + ["outer"]},
           "U": {c: float(solve.read_global(d, f"U_{rname(c)}.txt")[-1]) for c in conds}}
    if fields:
        peaks = {}
        for n, t in msh.tags.items():
            f = d / f"peak_{rname(n)}.txt"
            if t >= 100 or n == "shell" or not f.exists():
                continue
            pk = peak_nodes(f)
            if pk:
                peaks[n] = pk
        out["peaks"] = peaks
        if pl:
            out["slice_e"] = pro3d.read_plane(d / "slice_e.txt", pl)
            out["slice_v"] = pro3d.read_plane(d / "slice_v.txt", pl)
    return out


def run(scene: Scene, s: ElecSetup, out_dir: str | Path, question: str = "", quality: str = "normal",
        keep: bool = False, log=lambda stage, detail="": None, title: str = ""):
    from ..outcome_generic import GenericOutcome

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    conds = list(s.conductors)
    if not conds:
        raise ValueError("an electric field study needs at least one conductor, for example "
                         "--body 'electrode: conductor 5 kV' --body 'base: conductor 0 V'")
    fixed = [c for c in conds if s.conductors[c] is not None]
    diel = [n for n in s.dielectrics if n not in conds]
    ks = {"fast": (1.4, 1.0), "normal": (1.0, 0.72), "fine": (0.75, 0.55)}[quality]
    pl = None
    levels = []
    for j, k in enumerate(ks):
        log("mesh", f"mesh {j + 1} of {len(ks)}")
        msh = mesh(scene, conds, diel, work / f"l{j}" / "m.msh", k=k, domain=Domain("shell"), order=2)
        if pl is None:
            pl = pro3d.slice_plane(scene.bbox)
        log("solve", f"operating voltages on mesh {j + 1} ({msh.n_nodes:,} nodes)")
        op = _solve(msh, s, dict(s.conductors), work / f"l{j}" / "op", fields=True, pl=pl if j == len(ks) - 1 else None)
        C = np.zeros((len(fixed), len(fixed)))
        for a, ca in enumerate(fixed):
            log("solve", f"capacitance, {ca} at 1 V")
            unit = {c: (1.0 if c == ca else (0.0 if s.conductors[c] is not None else None)) for c in conds}
            r = _solve(msh, s, unit, work / f"l{j}" / f"c{a}")
            for b, cb in enumerate(fixed):
                C[b, a] = r["Q"][cb]
            if j == len(ks) - 1 and a == 0:
                unit0 = r
        levels.append({"msh": msh, "op": op, "C": C})
    fine, coarse = levels[-1], levels[0]
    op, C, msh = fine["op"], fine["C"], fine["msh"]
    checks = []
    # charge balance: what leaves the conductors arrives at infinity
    qs = op["Q"]
    qmax = max(abs(q) for q in qs.values()) or 1e-30
    bal = abs(sum(qs.values())) / qmax
    checks.append(Check("charge balance", "pass" if bal < 1e-3 else ("warn" if bal < 1e-2 else "fail"),
                        f"the charges on the conductors and at infinity add up to {100 * bal:.2g}% of the largest"))
    if len(fixed) > 1:
        asym = float(np.max(np.abs(C - C.T)) / np.max(np.abs(C)))
        checks.append(Check("reciprocity", "pass" if asym < 0.01 else ("warn" if asym < 0.03 else "fail"),
                            f"the capacitance matrix is symmetric to {100 * asym:.2g}% (C12 must equal C21)"))
    vv = np.array([s.conductors[c] for c in fixed], float)
    w_c = 0.5 * float(vv @ C @ vv) if len(fixed) and not any(s.conductors[c] is None for c in conds) else None
    if w_c:
        de = abs(op["W"] - w_c) / max(abs(w_c), 1e-30)
        checks.append(Check("energy balance", "pass" if de < 0.01 else ("warn" if de < 0.03 else "fail"),
                            f"the field stores {fmt(op['W'] * 1e6)} uJ and half V C V gives {fmt(w_c * 1e6)} uJ "
                            f"({100 * de:.2g}% apart)"))
    if len(fixed):
        c0 = [lv["C"][0, 0] for lv in levels]
        dc = abs(c0[1] - c0[0]) / abs(c0[1])
        checks.append(Check("mesh convergence", "pass" if dc < 0.01 else ("warn" if dc < 0.03 else "fail"),
                            f"the capacitance of '{fixed[0]}' moved from {fmt(c0[0] * 1e12)} to {fmt(c0[1] * 1e12)} pF "
                            f"on the finer mesh ({100 * dc:.2g}%)"))
    # breakdown in every region next to a conductor
    peaks_c = coarse["op"].get("peaks", {})
    peaks = op["peaks"]
    strength = {n: (s.dielectrics[n][1] if n in s.dielectrics else s.air_strength) for n in peaks}
    margins = {n: strength[n] / max(p[0], 1e-30) for n, p in peaks.items()}
    worst = min(margins, key=margins.get) if margins else None
    singular = False
    if worst:
        pk, where = peaks[worst]
        if worst in peaks_c:
            grow = (pk - peaks_c[worst][0]) / pk
            singular = grow > 0.08
            checks.append(Check("peak field convergence", "warn" if singular else "pass",
                                f"the peak field in {_reg(worst)} went from {fmt(peaks_c[worst][0] / 1e6)} to "
                                f"{fmt(pk / 1e6)} kV/mm on the finer mesh"
                                + (". It keeps rising, so it sits on a sharp edge where the true field depends on "
                                   "the edge radius" if singular else "")))
    lvl, reasons = confidence(checks)
    if worst:
        pk, where = peaks[worst]
        m = margins[worst]
        verdict = "ok" if m >= s.target else ("marginal" if m >= 1 else "fails")
        head = (f"The peak field is {fmt(pk / 1e6)} kV/mm in {_reg(worst)} near {_pt(where)} mm, "
                f"{m:.2g}x below its {fmt(strength[worst] / 1e6)} kV/mm strength"
                + ("" if verdict == "ok" else f", under the {s.target:g}x margin wanted") + ".")
        if verdict == "fails":
            head = (f"It breaks down. The field reaches {fmt(pk / 1e6)} kV/mm in {_reg(worst)} near {_pt(where)} mm, "
                    f"above the {fmt(strength[worst] / 1e6)} kV/mm it withstands.")
    else:
        verdict, head, m, pk, where = "ok", "", float("inf"), 0.0, (0, 0, 0)
    if len(fixed):
        head += " " + _cap_text(fixed, C)
    fixes = []
    if verdict != "ok" and worst:
        vmax = max(abs(v) for v in vv) if len(vv) else 0.0
        fixes.append({"kind": "voltage", "title": f"Keep the voltage under {fmt(vmax * m / s.target)} V",
                      "summary": f"The field scales with voltage, so that gives the {s.target:g}x margin.",
                      "auto": False})
        fixes.append({"kind": "edge", "title": f"Round the edge near {_pt(where)} mm",
                      "summary": "The field peaks where a conductor is sharpest. A radius of a few millimeters "
                                 "on that edge, or more distance there, lowers it the most.", "auto": False})
    images = {}
    if pl is not None and "slice_e" in op:
        images["field"] = plots.grid_map(out / "images" / "field.png", op["slice_e"] / 1e6, pl, "|E|, kV/mm",
                                         f"field strength on the plane {pl['normal']} = {fmt(pl['at'] * 1e3)} mm",
                                         lines=op["slice_v"])
    metrics = [("peak field", f"{fmt(pk / 1e6)} kV/mm"), ("margin", f"{m:.2g}x" if worst else "none"),
               ("capacitance", f"{fmt(C[0, 0] * 1e12)} pF" if len(fixed) else "none"),
               ("mesh", f"{msh.n_nodes:,} nodes")]
    assumptions = ["Electrostatic. Conductors are perfect and the air reaches to infinity.",
                   f"Air breaks down at {fmt(s.air_strength / 1e6)} kV/mm, the uniform field value across gaps of "
                   "several millimeters at sea level. Narrow gaps hold somewhat more (Paschen), and humidity, "
                   "contamination and creepage along surfaces lower it."]
    assumptions += [f"'{n}' is {v[2]}, relative permittivity {v[0]:g}, strength {fmt(v[1] / 1e6)} kV/mm."
                    for n, v in s.dielectrics.items() if n not in conds]
    assumptions += scene.notes
    values = {"capacitance_matrix_F": C.tolist(), "conductors": fixed, "charges_C": op["Q"],
              "potentials_V": op["U"], "energy_J": op["W"],
              "peaks": {n: {"field_V_per_m": p[0], "at_mm": [1e3 * x for x in p[1]], "strength_V_per_m": strength[n],
                            "margin": margins[n]} for n, p in peaks.items()},
              "singular_edge": singular, "nodes": [lv["msh"].n_nodes for lv in levels],
              "_figures": [["field", "Field strength and equipotentials on a slice"]]}
    o = GenericOutcome("electric", title or ", ".join(Path(b.file).name for b in scene.bodies[:1]), question, verdict,
                       head.strip(), metrics, lvl, reasons, checks, assumptions, values, images, fixes,
                       solver=f"GetDP {solve.getdp().version()} with gmsh", t_total=time.time() - t0)
    _playback(out, op, pl, scene, fixed, C, peaks, strength)
    _view(out, o, msh, op, pl, s, conds, diel, worst, peaks, title or scene.bodies[0].file, question)
    _ = unit0 if len(fixed) else None
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _view(out, o, msh, op, pl, s, conds, diel, worst, peaks, title, question):
    """The conductors and dielectrics as solved, the field on the slice, and the peak marked where it is."""
    try:
        from ..viz import Scene

        from . import view as V
    except ImportError:
        return None
    sc = Scene(f"{Path(title).stem} electric field", domain="electric", question=question, length="mm")
    V.draw_bodies(sc, msh, {**{c: "conductor" for c in conds}, **{d: "dielectric" for d in diel}})
    if pl is not None and "slice_e" in op:
        V.slice_mesh(sc, pl, op["slice_e"] / 1e6, "electric field", "kV/mm")
    if worst:
        pk, where = peaks[worst]
        sc.label(f"peak {pk / 1e6:.3g} kV/mm", np.array(where) * 1e3, kind="fail" if o.verdict == "fails" else "note")
        sc.readout("peak field", [pk / 1e6], unit="kV/mm", limit=s.dielectrics.get(worst, (0, s.air_strength))[1] / 1e6)
    C = o.values.get("capacitance_matrix_F") or []
    if C:
        sc.readout("capacitance", [C[0][0] * 1e12], unit="pF")
    word, tn = V.tone(o.verdict)
    sc.verdict(word, o.headline, tn)
    sc.note("The field is the solved slice through the parts; the parts are drawn from their own mesh.")
    return sc.save(out)


def _reg(n: str) -> str:
    return "the air" if n == "air" else f"'{n}'"


def _pt(p) -> str:
    return ", ".join(f"{1e3 * x:.3g}" for x in p)


def _cap_text(fixed: list[str], C: np.ndarray) -> str:
    if len(fixed) == 1:
        return f"'{fixed[0]}' has {fmt(C[0, 0] * 1e12)} pF to its surroundings."
    if len(fixed) == 2:
        c12 = -C[0, 1]
        return (f"Between '{fixed[0]}' and '{fixed[1]}' the mutual capacitance is {fmt(c12 * 1e12)} pF, with "
                f"{fmt((C[0, 0] - c12) * 1e12)} and {fmt((C[1, 1] - c12) * 1e12)} pF to the far surroundings.")
    return f"The {len(fixed)} conductors' capacitance matrix is in the values, with {fmt(C[0, 0] * 1e12)} pF on '{fixed[0]}'."


def _playback(out, op, pl, scene, fixed, C, peaks, strength):
    from . import playback as pb

    if pl is None or "slice_e" not in op:
        return None
    manifest = {"e": "electric field magnitude on the slice, V/m, packed", "v": "potential on the slice, V, packed",
                "slice": "the plane: its normal axis and position, the two in-plane axes and their ranges in mm",
                "bodies": "every body's name and box in mm", "capacitance_pF": "Maxwell capacitance matrix, pF",
                "peaks": "peak field per region with its location in mm and the strength there, V/m"}
    return pb.write(out, "electric-3d", manifest, {
        "slice": {k: (round(v * 1e3, 4) if k in ("at", "u0", "u1", "v0", "v1") else v) for k, v in pl.items()
                  if k not in ("iu", "iv", "ik")},
        "e": pb.pack(op["slice_e"], 0.0, clip=99.5), "v": pb.pack(op["slice_v"]),
        "bodies": [{"name": b.name, "box_mm": [round(1e3 * x, 4) for x in b.bbox]} for b in scene.bodies],
        "conductors": fixed, "capacitance_pF": [[round(x * 1e12, 5) for x in row] for row in C],
        "peaks": {n: {"field": round(p[0], 2), "at_mm": [round(1e3 * x, 3) for x in p[1]],
                      "strength": strength[n]} for n, p in peaks.items()}})


_ = math
