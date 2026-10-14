"""Magnetostatics on any CAD: forces, torques, inductance, saturation and demagnetization.

Steel follows its BH curve with Newton iterations, magnets carry their remanence in any direction and coils
carry their amp turns. The magnetic vector potential lives on mesh edges with a spanning tree gauge, and the
air ends in a shell mapped to infinity. A coil's current density first circulates about its axis and is then
projected onto a field with no divergence that follows the winding's own shape, so a racetrack or a square
coil works as well as a round one, and it is scaled so the amp turns through any cross section are exactly
N times I.

The force and torque on a body come from the Maxwell stress tensor in the layer of air elements around it.
The run checks that the forces on every body add up to zero (action and reaction), moves the body a little
both ways and compares the force with the change in co-energy (virtual work), repeats on a finer mesh, and
for coils without magnets closes the energy balance W + W' = sum of flux linkage times current.
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
from .assign import MagBody, _fmt_vec
from .forms import MU0, _pairs, bh_tables, rname
from .pro3d import num
from .scene import Domain, Scene, mesh


@dataclass
class MagSetup:
    bodies: dict[str, MagBody]
    targets: list[str] = field(default_factory=list)   # bodies whose force and torque are wanted
    need: float = 0.0                                  # force wanted on the first target, N
    notes: list[str] = field(default_factory=list)
    centers: dict = field(default_factory=dict)        # body: center of mass, m (filled from the scene)
    coil_size: float = 0.01                            # smallest coil extent, m (filled from the scene)


def _v(x) -> str:
    return f"Vector[{num(x[0])}, {num(x[1])}, {num(x[2])}]"


def problem(msh, s: MagSetup, mode: str, scale: dict | None = None, pl: dict | None = None,
            clusters: dict | None = None, fields: bool = True) -> str:
    """mode 'proj' solves only the coils' current paths and prints their fluxes, 'main' the whole problem.
    clusters maps an eggshell name to the bodies it wraps (a body and those touching it)."""
    vols = [n for n, t in msh.tags.items() if t < 100]
    bodies = {n: b for n, b in s.bodies.items() if n in msh.tags}
    steel = [n for n, b in bodies.items() if b.kind == "steel"]
    mags = [n for n, b in bodies.items() if b.kind == "magnet"]
    coils = [n for n, b in bodies.items() if b.kind == "coil"]
    lin = [n for n in vols if n not in steel]
    clusters = clusters or {}
    extra = {"Dom": vols, "DomL": lin, "DomNL": steel, "DomM": mags, "Coils": coils, "Outer": ["outer"],
             "Air": ["air"]}
    for k, members in clusters.items():
        extra[f"Body_{k}"] = members
    out = [pro3d.groups(msh.tags, extra)]
    fn = ["Function {", "  mu0 = 4e-7*Pi;"]
    for n in vols:
        b = bodies.get(n)
        g = rname(n)
        if b is not None and b.kind == "steel":
            sq, nu, w = bh_tables(*b.bh)
            fn += [f"  nu_{g}[] = InterpolationLinear[SquNorm[$1]]{{ {_pairs(sq, nu)} }};",
                   f"  dnudb2_{g}[] = dInterpolationLinear[SquNorm[$1]]{{ {_pairs(sq, nu)} }};",
                   f"  w_{g}[] = InterpolationLinear[SquNorm[$1]]{{ {_pairs(sq, w)} }};",
                   f"  nu[{g}] = nu_{g}[$1];", f"  dnudb2[{g}] = dnudb2_{g}[$1];", f"  wd[{g}] = w_{g}[$1];"]
        else:
            fn.append(f"  nu[{g}] = {num(1 / (MU0 * (b.mur if b else 1.0)))};")
        if b is not None and b.kind == "magnet":
            fn.append(f"  br[{g}] = {_v(b.br)};")
        if b is not None and b.kind == "coil":
            fn.append(f"  axr[{g}] = CrossProduct[{_v(b.axis)}, XYZ[] - {_v(b.center)}];")
            fn.append(f"  ephi[{g}] = axr[] / Norm[axr[]];")
            fn.append(f"  gth[{g}] = axr[] / SquNorm[axr[]];")
            if scale:
                fn.append(f"  ic[{g}] = {num(scale[n])};")
    if steel:
        fn.append("  dhdb[] = TensorDiag[1, 1, 1] * nu[$1#1] + 2 * dnudb2[#1] * SquDyadicProduct[#1];")
    fn.append("  TM[] = (SquDyadicProduct[$1] - SquNorm[$1] * TensorDiag[0.5, 0.5, 0.5]) / mu0;")
    fn.append("}")
    out.append("\n".join(fn) + "\n")
    cons = ["Constraint {", "  { Name Dir_a; Case { { Region Outer; Value 0; } } }",
            "  { Name Gauge; Type Assign; Case { { Region Dom; SubRegion Outer; Value 0; } } }"]
    for k in clusters:
        cons.append(f"  {{ Name Unit_{k}; Case {{ {{ Region Body_{k}; Value 1; }} }} }}")
    cons.append("}")
    out.append("\n".join(cons) + "\n")
    out.append(pro3d.jacobian(msh, True))
    out.append(pro3d.INTEGRATION)
    fs = ["FunctionSpace {"]
    if mode == "main":
        fs.append("  { Name Ha; Type Form1; BasisFunction {\n"
                  "      { Name se; NameOfCoef ae; Function BF_Edge; Support Dom; Entity EdgesOf[All]; } }\n"
                  "    Constraint { { NameOfCoef ae; EntityType EdgesOf; NameOfConstraint Dir_a; }\n"
                  "      { NameOfCoef ae; EntityType EdgesOfTreeIn; EntitySubType StartingOn; "
                  "NameOfConstraint Gauge; } }\n  }")
        for k in clusters:
            fs.append(f"  {{ Name Hun_{k}; Type Form0; BasisFunction {{\n"
                      f"      {{ Name sn; NameOfCoef un; Function BF_GroupOfNodes; Support Air; "
                      f"Entity GroupsOfNodesOf[Body_{k}]; }} }}\n"
                      f"    Constraint {{ {{ NameOfCoef un; EntityType GroupsOfNodesOf; NameOfConstraint Unit_{k}; }} }}"
                      "\n  }")
    if coils:
        fs.append("  { Name Hpsi; Type Form0; BasisFunction {\n"
                  "      { Name sn; NameOfCoef pn; Function BF_Node; Support Coils; Entity NodesOf[All]; } }\n  }")
    out.append("\n".join(fs) + "\n}\n")
    form = ["Formulation {"]
    if coils:
        form.append("  { Name P; Type FemEquation;\n    Quantity { { Name psi; Type Local; NameOfSpace Hpsi; } }\n"
                    "    Equation {\n"
                    "      Integral { [ Dof{d psi}, {d psi} ]; In Coils; Jacobian Vol; Integration I1; }\n"
                    f"      Integral {{ [ {num(1e-6 / s.coil_size ** 2)} * Dof{{psi}}, {{psi}} ]; In Coils; Jacobian Vol; "
                    "Integration I1; }\n"
                    "      Integral { [ -ephi[], {d psi} ]; In Coils; Jacobian Vol; Integration I1; }\n"
                    "    }\n  }")
    if mode == "main":
        q = ["      { Name a; Type Local; NameOfSpace Ha; }"]
        q += [f"      {{ Name un_{k}; Type Local; NameOfSpace Hun_{k}; }}" for k in clusters]
        if coils:
            q.append("      { Name psi; Type Local; NameOfSpace Hpsi; }")
        eq = ["      Integral { [ nu[] * Dof{d a}, {d a} ]; In DomL; Jacobian Vol; Integration I1; }"]
        if steel:
            eq += ["      Integral { [ nu[{d a}] * {d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }",
                   "      Integral { [ dhdb[{d a}] * Dof{d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }",
                   "      Integral { [ -dhdb[{d a}] * {d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }"]
        if mags:
            eq.append("      Integral { [ -nu[] * br[], {d a} ]; In DomM; Jacobian Vol; Integration I1; }")
        if coils:
            eq.append("      Integral { [ -ic[] * (ephi[] - {d psi}), {a} ]; In Coils; Jacobian Vol; Integration I1; }")
        eq += [f"      Integral {{ [ 0 * Dof{{un_{k}}}, {{un_{k}}} ]; In Air; Jacobian Vol; Integration I1; }}"
               for k in clusters]
        form.append("  { Name M; Type FemEquation;\n    Quantity {\n" + "\n".join(q) + "\n    }\n    Equation {\n" +
                    "\n".join(eq) + "\n    }\n  }")
    out.append("\n".join(form) + "\n}\n")
    if mode == "proj":
        res = ("Resolution {\n  { Name R; System { { Name B; NameOfFormulation P; } }\n"
               "    Operation { Generate[B]; Solve[B]; SaveSolution[B]; }\n  }\n}\n")
    else:
        sysl = "{ Name A; NameOfFormulation M; }" + (" { Name B; NameOfFormulation P; }" if coils else "")
        ops = ("Generate[B]; Solve[B]; SaveSolution[B]; " if coils else "") + "InitSolution[A]; Generate[A]; Solve[A];"
        if steel:
            ops += pro3d.newton(tol=1e-6)
        ops += " SaveSolution[A];"
        res = f"Resolution {{\n  {{ Name R; System {{ {sysl} }}\n    Operation {{ {ops} }}\n  }}\n}}\n"
    out.append(res)
    pq, po = [], []
    if coils:
        form_name = "P" if mode == "proj" else "M"
        pq.append(f"  {{ Name Q{form_name}; NameOfFormulation {form_name}; Quantity {{")
        for n in coils:
            g = rname(n)
            pq.append(f"      {{ Name F_{g}; Value {{ Integral {{ [ (ephi[] - {{d psi}}) * gth[] / (2 * Pi) ]; In {g}; "
                      "Jacobian Vol; Integration I1; } } }")
            if mode == "main":
                tn = 1.0 / s.bodies[n].current if s.bodies[n].current else 0.0     # flux linkage per amp
                pq.append(f"      {{ Name Psi_{g}; Value {{ Integral {{ [ {num(tn)} * ic[] * (ephi[] - {{d psi}}) * {{a}} ]; "
                          f"In {g}; Jacobian Vol; Integration I1; }} }} }}")
        pq.append("  } }")
        for n in coils:
            g = rname(n)
            po.append(f'Print[ F_{g}[{g}], OnGlobal, Format Table, File "F_{g}.txt" ];')
            if mode == "main":
                po.append(f'Print[ Psi_{g}[{g}], OnGlobal, Format Table, File "Psi_{g}.txt" ];')
    if mode == "main":
        plain = [n for n in vols if n not in steel and n not in mags]
        wq = [f"Integral {{ [ 0.5 * nu[] * SquNorm[{{d a}}] ]; In {_grp(plain)}; Jacobian Vol; Integration I1; }}"]
        cq = list(wq)
        if steel:
            wq.append("Integral { [ wd[{d a}] ]; In DomNL; Jacobian Vol; Integration I1; }")
            cq.append("Integral { [ nu[{d a}] * SquNorm[{d a}] - wd[{d a}] ]; In DomNL; Jacobian Vol; Integration I1; }")
        if mags:
            wq.append("Integral { [ 0.5 * nu[] * SquNorm[{d a} - br[]] ]; In DomM; Jacobian Vol; Integration I1; }")
            cq.append("Integral { [ 0.5 * nu[] * SquNorm[{d a} - br[]] + br[] * (nu[] * ({d a} - br[])) ]; "
                      "In DomM; Jacobian Vol; Integration I1; }")
        m = ["  { Name M; NameOfFormulation M; Quantity {",
             "      { Name b; Value { Local { [ {d a} ]; In Dom; Jacobian Vol; } } }",
             "      { Name nb; Value { Local { [ Norm[{d a}] ]; In Dom; Jacobian Vol; } } }",
             "      { Name W; Value {\n        " + "\n        ".join(wq) + " } }",
             "      { Name Wc; Value {\n        " + "\n        ".join(cq) + " } }"]
        if mags:
            m.append("      { Name hpar; Value { Local { [ (nu[] * ({d a} - br[])) * br[] / Norm[br[]] ]; In DomM; "
                     "Jacobian Vol; } } }")
        for k, members in clusters.items():
            c = np.mean([np.array(s.centers.get(x, (0.0, 0.0, 0.0))) for x in members], axis=0)
            m.append(f"      {{ Name F_{k}; Value {{ Integral {{ [ -TM[{{d a}}] * {{d un_{k}}} ]; In Air; Jacobian Vol; "
                     "Integration I1; } } }")
            m.append(f"      {{ Name T_{k}; Value {{ Integral {{ [ -(XYZ[] - {_v(c)}) /\\ (TM[{{d a}}] * {{d un_{k}}}) ]; "
                     "In Air; Jacobian Vol; Integration I1; } } }")
        # the coils' quantities belong to the same post processing in the main run
        m += [x for x in pq if not x.startswith("  { Name Q") and x != "  } }"]
        m.append("  } }")
        pq = m
        po += ['Print[ W[Dom], OnGlobal, Format Table, File "W.txt" ];',
               'Print[ Wc[Dom], OnGlobal, Format Table, File "Wc.txt" ];']
        for k in clusters:
            po += [f'Print[ F_{k}[Air], OnGlobal, Format Table, File "F_{k}.txt" ];',
                   f'Print[ T_{k}[Air], OnGlobal, Format Table, File "T_{k}.txt" ];']
        if fields:
            for n in steel:
                po.append(f'Print[ nb, OnElementsOf {rname(n)}, Format SimpleTable, File "bpk_{rname(n)}.txt" ];')
            for n in mags:
                po.append(f'Print[ hpar, OnElementsOf {rname(n)}, Format SimpleTable, File "hm_{rname(n)}.txt" ];')
            if pl:
                po.append(pro3d.print_plane("b", pl, "slice_b.txt"))
    out.append("PostProcessing {\n" + "\n".join(pq) + "\n}\n")
    pp = "M" if mode == "main" else "QP"
    out.append(f"PostOperation {{\n  {{ Name P; NameOfPostProcessing {pp}; Operation {{\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)


def _grp(names: list[str]) -> str:
    return "Region[{" + ", ".join(rname(n) for n in names) + "}]"


def _clusters(targets: list[str], all_bodies: list[str], touching: list) -> dict[str, list[str]]:
    """Each body with every body it touches, since the stress tensor needs air all around what it acts on."""
    parent = {b: b for b in all_bodies}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b in touching:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for b in all_bodies:
        groups.setdefault(find(b), []).append(b)
    out = {}
    for i, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        out[f"c{i}"] = sorted(members)
    _ = targets
    return out


def solve_once(scene: Scene, s: MagSetup, d: Path, k: float, pl: dict | None = None, fields: bool = True,
               log=lambda a, b="": None, msh=None) -> dict:
    """Mesh (unless a mesh is given) and solve. Returns energies, forces and torques on every body cluster."""
    d.mkdir(parents=True, exist_ok=True)
    vols = [n for n in s.bodies]
    if msh is None:
        msh = mesh(scene, [], vols, d / "m.msh", k=k, domain=Domain("shell", shell=2.0, gap_cells=3.0), order=1)
    s.centers = {b.name: b.center for b in scene.bodies}
    ext = [b.extent for b in scene.bodies if b.name in s.bodies and s.bodies[b.name].kind == "coil"]
    s.coil_size = min(ext) if ext else 0.01
    present = [n for n in vols if n in msh.tags]
    clusters = _clusters(s.targets, present, msh.touching)
    coils = [n for n in present if s.bodies[n].kind == "coil"]
    scale = {}
    if coils:
        log("solve", f"current paths in {len(coils)} coil{'s' if len(coils) > 1 else ''}")
        solve.run(problem(msh, s, "proj", clusters={}), msh.path, d / "proj", resolution="R")
        for n in coils:
            f = abs(float(solve.read_global(d / "proj", f"F_{rname(n)}.txt")[-1]))
            b = s.bodies[n]
            scale[n] = b.turns * b.current / max(f, 1e-30)
    log("solve", f"field ({msh.n_nodes:,} nodes)")
    lg = solve.run(problem(msh, s, "main", scale, pl, clusters, fields), msh.path, d / "main", resolution="R",
                   timeout=1800)
    w = d / "main"
    out = {"msh": msh, "clusters": clusters, "W": float(solve.read_global(w, "W.txt")[-1]),
           "Wc": float(solve.read_global(w, "Wc.txt")[-1]), "residual": solve.residual(lg)}
    out["F"] = {k_: np.array(solve.read_global(w, f"F_{k_}.txt")[-3:], float) for k_ in clusters}
    out["T"] = {k_: np.array(solve.read_global(w, f"T_{k_}.txt")[-3:], float) for k_ in clusters}
    out["psi"] = {n: float(solve.read_global(w, f"Psi_{rname(n)}.txt")[-1]) for n in coils}
    out["flux_check"] = {n: abs(float(solve.read_global(w, f"F_{rname(n)}.txt")[-1])) * scale[n] /
                         (s.bodies[n].turns * s.bodies[n].current) for n in coils}
    if fields:
        from .es3d import peak_nodes
        out["bpk"] = {n: peak_nodes(w / f"bpk_{rname(n)}.txt") for n in present if s.bodies[n].kind == "steel"}
        out["steel_b"] = {n: _values(w / f"bpk_{rname(n)}.txt") for n in present if s.bodies[n].kind == "steel"}
        out["hm"] = {n: _values(w / f"hm_{rname(n)}.txt") for n in present if s.bodies[n].kind == "magnet"}
        if pl:
            out["slice_b"] = pro3d.read_plane(w / "slice_b.txt", pl, comps=3)
    return out


def morph(msh, members: list[str], bbox: tuple, halo: float, shift, path: Path):
    """The same mesh with the bodies in members moved by shift and the air around them stretched over a halo, so
    the co-energy changes smoothly with no remeshing noise. Nodes of every other body stay put."""
    from dataclasses import replace

    lines = Path(msh.path).read_text().splitlines()
    i0 = lines.index("$Nodes") + 2
    n = int(lines[i0 - 1])
    ids = np.array([int(lines[i0 + j].split()[0]) for j in range(n)])
    X = np.array([[float(v) for v in lines[i0 + j].split()[1:4]] for j in range(n)])
    e0 = lines.index("$Elements") + 2
    ne = int(lines[e0 - 1])
    tag_of = {t: name for name, t in msh.tags.items()}
    moving, fixed = set(), set()
    for j in range(ne):
        t = lines[e0 + j].split()
        if t[1] != "4":                          # tetrahedra only
            continue
        nt = int(t[2])
        reg = tag_of.get(int(t[3]), "")
        nodes = [int(x) for x in t[3 + nt:]]
        if reg in members:
            moving.update(nodes)
        elif reg not in ("air", "shell"):
            fixed.update(nodes)
    lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
    dist = np.linalg.norm(np.maximum(np.maximum(lo - X, X - hi), 0.0), axis=1)
    w = np.clip(1 - dist / halo, 0.0, 1.0)
    pos = {nid: k for k, nid in enumerate(ids)}
    for nid in fixed - moving:
        w[pos[nid]] = 0.0
    for nid in moving:
        w[pos[nid]] = 1.0
    X2 = X + w[:, None] * np.asarray(shift, float)[None, :]
    for j in range(n):
        lines[i0 + j] = f"{ids[j]} {float(X2[j, 0])!r} {float(X2[j, 1])!r} {float(X2[j, 2])!r}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return replace(msh, path=str(path))


def _values(path) -> np.ndarray:
    try:
        a = np.loadtxt(path, ndmin=2)
    except (OSError, ValueError):
        return np.zeros(0)
    if not a.size:
        return np.zeros(0)
    nv = a.shape[1] // 4
    return a[:, 3 * nv:4 * nv].ravel()


def cluster_of(r: dict, name: str) -> str:
    return next(k for k, m in r["clusters"].items() if name in m)


def run(scene: Scene, s: MagSetup, out_dir: str | Path, question: str = "", quality: str = "normal",
        keep: bool = False, log=lambda stage, detail="": None, title: str = ""):
    from ..outcome_generic import GenericOutcome

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    if not s.bodies:
        raise ValueError("a magnetic study needs at least one magnet, coil or piece of steel, for example "
                         "--assign 'magnet: n42 +z' --assign 'plate: steel 1010'")
    if not any(b.kind in ("magnet", "coil") for b in s.bodies.values()):
        raise ValueError("nothing drives the field. Give a magnet with its direction or a coil with its turns and "
                         "current")
    targets = s.targets or _default_targets(scene, s)
    s.targets = targets
    ks = {"fast": (1.4,), "normal": (1.0, 0.75), "fine": (0.8, 0.6)}[quality]
    pl = pro3d.slice_plane(scene.bbox)
    levels = []
    for j, k in enumerate(ks):
        log("mesh", f"mesh {j + 1} of {len(ks)}")
        levels.append(solve_once(scene, s, work / f"l{j}", k, pl if j == len(ks) - 1 else None, log=log))
    fine = levels[-1]
    tgt = targets[0]
    ck = cluster_of(fine, tgt)
    F = fine["F"][ck]
    Fm = float(np.linalg.norm(F))
    checks = []
    if len(levels) > 1:
        F0 = float(np.linalg.norm(levels[0]["F"][cluster_of(levels[0], tgt)]))
        dF = abs(Fm - F0) / max(Fm, 1e-30)
        checks.append(Check("mesh convergence", "pass" if dF < 0.03 else ("warn" if dF < 0.08 else "fail"),
                            f"the force on '{tgt}' moved from {fmt(F0)} to {fmt(Fm)} N on the finer mesh "
                            f"({100 * dF:.2g}%)"))
    tot = sum(fine["F"].values())
    big = max(float(np.linalg.norm(f)) for f in fine["F"].values())
    if len(fine["F"]) > 1 and big > 0:
        r3 = float(np.linalg.norm(tot)) / big
        checks.append(Check("action and reaction", "pass" if r3 < 0.05 else ("warn" if r3 < 0.12 else "fail"),
                            f"the forces on all {len(fine['F'])} bodies add up to {fmt(float(np.linalg.norm(tot)))} N, "
                            f"{100 * r3:.2g}% of the largest, where they should cancel"))
    vw = None
    if Fm > 0 and quality != "fast" or Fm > 0 and len(ks) == 1:
        dirn = F / Fm
        members = fine["clusters"][ck]
        b0 = [b for b in scene.bodies if b.name in members]
        gap = _gap(scene, members)
        # move the body a small fraction of its gap on the same mesh, the air around it stretching over a halo
        # half the gap wide, and compare the change in co-energy with the force
        size = min(b.extent for b in b0)
        halo = 0.5 * gap if gap > 0 else 0.1 * size
        dx = 0.02 * halo
        bb = np.array([b.bbox for b in b0])
        box = tuple(bb[:, :3].min(axis=0)) + tuple(bb[:, 3:].max(axis=0))
        base = levels[0]
        log("verify", f"virtual work, moving '{tgt}' {fmt(dx * 1e3)} mm each way on the same mesh")
        wc = []
        for sgn in (-1, 1):
            m2 = morph(base["msh"], members, box, halo, sgn * dx * dirn, work / f"vw{sgn}" / "m.msh")
            wc.append(solve_once(scene, s, work / f"vw{sgn}", ks[0], None, fields=False, log=log, msh=m2)["Wc"])
        vw = (wc[1] - wc[0]) / (2 * dx)
        F_lvl0 = float(np.dot(base["F"][cluster_of(base, tgt)], dirn))
        dv = abs(vw - F_lvl0) / max(abs(F_lvl0), 1e-30)
        checks.append(Check("virtual work", "pass" if dv < 0.05 else ("warn" if dv < 0.12 else "fail"),
                            f"Maxwell stress gives {fmt(F_lvl0)} N on '{tgt}' along its pull, and moving it "
                            f"{fmt(dx * 1e3)} mm each way changes the co-energy by {fmt(vw)} N times the move "
                            f"({100 * dv:.2g}% apart)"))
    coils = [n for n, b in s.bodies.items() if b.kind == "coil" and n in fine["psi"]]
    mags = [n for n, b in s.bodies.items() if b.kind == "magnet"]
    if coils and not mags:
        psi_i = sum(fine["psi"][n] * s.bodies[n].current for n in coils)
        e = abs(fine["W"] + fine["Wc"] - psi_i) / max(abs(psi_i), 1e-30)
        checks.append(Check("energy balance", "pass" if e < 0.02 else ("warn" if e < 0.05 else "fail"),
                            f"stored energy plus co-energy is {fmt((fine['W'] + fine['Wc']) * 1e3)} mJ and flux linkage "
                            f"times current is {fmt(psi_i * 1e3)} mJ ({100 * e:.2g}% apart)"))
    if coils:
        fc = max(abs(v - 1) for v in fine["flux_check"].values())
        checks.append(Check("amp turns", "pass" if fc < 1e-3 else "warn",
                            "the current through every coil's cross section is N times I to "
                            f"{100 * fc:.2g}%, along the winding's own shape"))
    it, rel = fine["residual"]
    if any(b.kind == "steel" for b in s.bodies.values()):
        checks.append(Check("nonlinear solve", "pass" if rel < 1e-5 else "warn",
                            f"Newton iterations on the steel's BH curves reached a relative residual of {rel:.1e} "
                            f"in {it} iterations"))
    sat = {}
    for n, arr in fine.get("steel_b", {}).items():
        if len(arr):
            knee = _knee(s.bodies[n])
            sat[n] = (float(arr.max()), knee, float((arr > knee).mean()))
    for n, (bmax, knee, frac) in sat.items():
        checks.append(Check(f"saturation of {n}", "info",
                            f"'{n}' peaks at {fmt(bmax)} T against its {fmt(knee)} T knee"
                            + (f", past it over {100 * frac:.0f}% of the sampled points" if bmax > knee else "")))
    demag = {}
    for n in mags:
        arr = fine.get("hm", {}).get(n, np.zeros(0))
        b = s.bodies[n]
        if len(arr) and b.hcj:
            demag[n] = float(max(0.0, -arr.min()) / b.hcj)
            checks.append(Check(f"demagnetization of {n}", "info" if demag[n] < 0.8 else
                                ("warn" if demag[n] < 1 else "fail"),
                                f"the worst reverse field in '{n}' is {fmt(-arr.min() / 1e3)} kA/m, "
                                f"{100 * demag[n]:.0f}% of its {fmt(b.hcj / 1e3)} kA/m coercivity at 20 C"))
    lvl, reasons = confidence(checks)
    T = fine["T"][ck]
    members = fine["clusters"][ck]
    who = f"'{tgt}'" if len(members) == 1 else "'" + "' + '".join(members) + "'"
    head = (f"The field pushes {who} with {fmt(Fm)} N, mostly along {_fmt_vec(F / Fm) if Fm else 'nothing'}"
            f" ({fmt(F[0])}, {fmt(F[1])}, {fmt(F[2])} N), with {fmt(float(np.linalg.norm(T)) * 1e3)} mN m of torque "
            "about its center.")
    verdict = "ok"
    if s.need:
        verdict = "ok" if Fm >= 1.15 * s.need else ("marginal" if Fm >= s.need else "fails")
        head += (f" That is {'enough for' if Fm >= s.need else 'short of'} the {fmt(s.need)} N needed.")
    if coils:
        head += " " + "; ".join(f"'{n}' has {fmt(fine['psi'][n] / s.bodies[n].current * 1e3)} mH" for n in coils) + "."
    for n, (bmax, knee, frac) in sat.items():
        if bmax > knee * 1.05:
            head += f" '{n}' saturates at {fmt(bmax)} T."
    fixes = []
    if verdict != "ok":
        if coils:
            fixes.append({"kind": "amp-turns", "title": "Raise the amp turns",
                          "summary": "Below saturation the force grows with the square of N times I.", "auto": False})
        if mags:
            fixes.append({"kind": "magnet", "title": "Use a stronger grade or a longer magnet",
                          "summary": "Force grows with the square of the remanence.", "auto": False})
        fixes.append({"kind": "gap", "title": "Close the air gap",
                      "summary": "The gap holds most of the magnetic reluctance, so halving it roughly doubles the "
                                 "flux until the steel saturates.", "auto": False})
    images = {}
    if "slice_b" in fine:
        bvec = fine["slice_b"]
        bn = np.linalg.norm(bvec, axis=-1) if bvec.ndim == 3 else bvec
        images["field"] = plots.grid_map(out / "images" / "field.png", bn, pl, "|B|, T",
                                         f"flux density on the plane {pl['normal']} = {fmt(pl['at'] * 1e3)} mm",
                                         solid=_solid(scene, s, pl))
    metrics = [("force", f"{fmt(Fm)} N"), ("torque", f"{fmt(float(np.linalg.norm(T)) * 1e3)} mN m"),
               ("inductance", f"{fmt(fine['psi'][coils[0]] / s.bodies[coils[0]].current * 1e3)} mH" if coils else
                "no coil"), ("mesh", f"{fine['msh'].n_nodes:,} nodes")]
    assumptions = ["Magnetostatic: steady currents and fields, no eddy currents.",
                   "The air reaches to infinity, mapped by a spherical shell."]
    for n, b in s.bodies.items():
        if b.kind == "steel":
            assumptions.append(f"'{n}' is {b.material}, nonlinear BH curve, no hysteresis.")
        elif b.kind == "magnet":
            assumptions.append(f"'{n}' is a {b.material} magnet, remanence {fmt(float(np.linalg.norm(b.br)))} T along "
                               f"{_fmt_vec(b.br / np.linalg.norm(b.br))}, recoil permeability {b.mur:g}, at 20 C.")
        elif b.kind == "coil":
            assumptions.append(f"'{n}' is a coil of {fmt(b.turns)} turns at {fmt(b.current)} A, the current spread "
                               "evenly over its cross section.")
        else:
            assumptions.append(f"'{n}' has relative permeability {b.mur:g}.")
    assumptions += s.notes + scene.notes
    values = {"force_N": {"/".join(fine["clusters"][k_]): v.tolist() for k_, v in fine["F"].items()},
              "torque_Nm": {"/".join(fine["clusters"][k_]): v.tolist() for k_, v in fine["T"].items()},
              "target": tgt, "force_target_N": F.tolist(), "virtual_work_N": vw,
              "flux_linkage_Wb": fine["psi"], "inductance_H": {n: fine["psi"][n] / s.bodies[n].current for n in coils},
              "energy_J": fine["W"], "coenergy_J": fine["Wc"],
              "saturation": {n: {"b_max_T": v[0], "knee_T": v[1], "fraction": v[2]} for n, v in sat.items()},
              "demagnetization": demag, "nodes": [lv["msh"].n_nodes for lv in levels],
              "_figures": [["field", "Flux density on a slice"]]}
    o = GenericOutcome("magnetic", title or scene.bodies[0].file, question, verdict, head, metrics, lvl, reasons,
                       checks, assumptions, values, images, fixes, solver=f"GetDP {solve.getdp().version()} with gmsh",
                       t_total=time.time() - t0)
    _playback(out, fine, pl, scene, s, F, T, tgt)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return o.write(out)


def _knee(b: MagBody) -> float:
    from ..materials import MagneticMaterial
    return MagneticMaterial("x", "x", bh=b.bh).knee()


def _default_targets(scene: Scene, s: MagSetup) -> list[str]:
    """Without a target, the force on the steel part that is free to move: the smallest steel body that a magnet or
    coil pulls, else the first magnet."""
    steel = [n for n, b in s.bodies.items() if b.kind in ("steel", "linear")]
    if steel:
        sizes = {b.name: b.volume for b in scene.bodies}
        return [min(steel, key=lambda n: sizes[n])]
    return [next(n for n, b in s.bodies.items() if b.kind == "magnet" or b.kind == "coil")]


def _gap(scene: Scene, members: list[str]) -> float:
    a = [b for b in scene.bodies if b.name in members]
    rest = [b for b in scene.bodies if b.name not in members]
    best = math.inf
    for x in a:
        for y in rest:
            g = max(max(x.bbox[i] - y.bbox[i + 3], y.bbox[i] - x.bbox[i + 3]) for i in range(3))
            best = min(best, max(g, 0.0))
    return 0.0 if best is math.inf else best


def _solid(scene: Scene, s: MagSetup, pl: dict) -> np.ndarray:
    """Cells of the slice inside a modeled body's box, for outlines."""
    gu = np.linspace(pl["u0"], pl["u1"], pl["nu"])
    gv = np.linspace(pl["v0"], pl["v1"], pl["nv"])
    U, V = np.meshgrid(gu, gv)
    m = np.zeros(U.shape, bool)
    for b in scene.bodies:
        if b.name not in s.bodies:
            continue
        bb = b.bbox
        if not bb[pl["ik"]] <= pl["at"] <= bb[pl["ik"] + 3]:
            continue
        m |= (U >= bb[pl["iu"]]) & (U <= bb[pl["iu"] + 3]) & (V >= bb[pl["iv"]]) & (V <= bb[pl["iv"] + 3])
    return m


def _playback(out, fine, pl, scene, s, F, T, tgt):
    from . import playback as pb

    if "slice_b" not in fine:
        return None
    bvec = fine["slice_b"]
    u, v = pl["iu"], pl["iv"]
    m = float(np.nanpercentile(np.linalg.norm(bvec, axis=-1), 99.5)) if np.isfinite(bvec).any() else 1.0
    manifest = {"b_u, b_v": "the in-plane components of the flux density on the slice, T, packed on one range; "
                            "draw them as arrows or trace field lines from them",
                "b": "flux density magnitude on the slice, T, packed",
                "slice": "the plane: its normal axis and position, the two in-plane axes and their ranges in mm",
                "bodies": "every body's name, role and box in mm",
                "force": "force (N) and torque (N m) on the target body"}
    return pb.write(out, "magnetic-3d", manifest, {
        "slice": {k: (round(x * 1e3, 4) if k in ("at", "u0", "u1", "v0", "v1") else x) for k, x in pl.items()
                  if k not in ("iu", "iv", "ik")},
        "b": pb.pack(np.linalg.norm(bvec, axis=-1), 0.0, m), "b_u": pb.pack(bvec[..., u], -m, m),
        "b_v": pb.pack(bvec[..., v], -m, m),
        "bodies": [{"name": b.name, "role": s.bodies[b.name].kind if b.name in s.bodies else "air",
                    "box_mm": [round(1e3 * x, 4) for x in b.bbox]} for b in scene.bodies],
        "force": {"body": tgt, "force_N": [round(float(x), 5) for x in F],
                  "torque_Nm": [round(float(x), 7) for x in T]}})
