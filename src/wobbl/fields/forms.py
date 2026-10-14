"""GetDP problem files for the field studies, written from Python.

Each builder returns the text of a .pro file for a mesh made by fields.mesh2d
or fields.mesh3d, with every number in SI units. Global results (energy, flux
linkage, force, charge, power) are printed to small text files next to the
mesh, and fields to node tables for plotting. fields.solve runs them and reads
the files back.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
C0 = 299792458.0


def rname(name: str) -> str:
    return "R_" + re.sub(r"\W", "_", name)


def _group(name: str, tags: list[int]) -> str:
    return f"  {name} = Region[{{{', '.join(str(t) for t in tags)}}}];"


def _groups(mesh_tags: dict[str, int], extra: dict[str, list[str]]) -> str:
    lines = [f"  {rname(n)} = Region[{t}];" for n, t in mesh_tags.items()]
    for g, names in extra.items():
        lines.append(_group(g, [mesh_tags[n] for n in names if n in mesh_tags]))
    return "Group {\n" + "\n".join(lines) + "\n}\n"


def _jacobian(kind: str, shell: tuple | None, shell_name: str = "shell") -> str:
    vol = "VolAxiSqu" if kind == "axi" else "Vol"
    sur = "SurAxi" if kind == "axi" else "Sur"
    sh = ""
    if shell:
        rint, rext, cx, cy = shell
        sh = f"      {{ Region {rname(shell_name)}; Jacobian {vol}SphShell{{{rint}, {rext}, {cx}, {cy}, 0}}; }}\n"
    return ("Jacobian {\n"
            f"  {{ Name Vol; Case {{\n{sh}      {{ Region All; Jacobian {vol}; }} }} }}\n"
            f"  {{ Name Sur; Case {{ {{ Region All; Jacobian {sur}; }} }} }}\n}}\n")


INTEGRATION = """Integration {
  { Name I1; Case { { Type Gauss; Case {
      { GeoElement Point; NumberOfPoints 1; } { GeoElement Line; NumberOfPoints 4; }
      { GeoElement Triangle; NumberOfPoints 6; } { GeoElement Tetrahedron; NumberOfPoints 4; } } } } }
}
"""


def bh_tables(b: list[float], h: list[float], n: int = 120) -> tuple[list[float], list[float], list[float]]:
    """Resample a BH curve and return (b squared, reluctivity nu = H/B, stored energy density) on the same points.

    Reluctivity is interpolated linearly in B squared, the way GetDP's material library does it, and the energy
    density w(B) = integral of H dB = 0.5 * integral of nu d(B^2) is exact for that interpolation.
    """
    b = np.asarray(b, float)
    h = np.asarray(h, float)
    top = b[-1]
    bs = np.linspace(0.0, top, n)
    hs = np.interp(bs, b, h)
    nu = np.empty(n)
    nu[1:] = hs[1:] / bs[1:]
    nu[0] = nu[1]
    s = bs ** 2
    w = np.concatenate([[0.0], np.cumsum(0.25 * (nu[1:] + nu[:-1]) * np.diff(s))])
    # beyond the table the material behaves like air past saturation: extend with mu0 slope
    s_ext = (top * 4) ** 2
    nu_ext = (hs[-1] + (4 * top - top) / MU0) / (4 * top)
    s = np.append(s, s_ext)
    w = np.append(w, w[-1] + 0.25 * (nu[-1] + nu_ext) * (s_ext - s[-2]))
    nu = np.append(nu, nu_ext)
    return list(s), list(nu), list(w)


def _pairs(x, y) -> str:
    return ", ".join(f"{a:.9g}, {c:.9g}" for a, c in zip(x, y))


@dataclass
class MagRegion:
    name: str
    mur: float = 1.0
    bh: tuple[list[float], list[float]] | None = None    # nonlinear iron: (B in T, H in A/m)
    js: float = 0.0                                     # current density out of the plane (around the axis), A/m2
    br: tuple[float, float] | None = None               # remanent flux density of a magnet, T


@dataclass
class Coil:
    name: str               # region of one side of the winding
    turns: float
    area: float             # cross section of that region, m2
    sign: float = 1.0       # -1 for the return side of a planar coil


@dataclass
class MagProblem:
    kind: str                                   # planar or axi
    regions: list[MagRegion]
    dirichlet: list[str]                        # boundaries where a = 0 (outer edge, axis)
    shell: tuple | None = None                  # (r inner, r outer, cx, cy) of the region named shell
    coils: list[Coil] = field(default_factory=list)
    body: list[str] = field(default_factory=list)    # regions the force and torque act on
    air: str = "air"                                 # region around the body where the force is integrated
    depth: float = 1.0                               # planar models: length out of the plane, m
    center: tuple[float, float] = (0.0, 0.0)         # torque center
    probes: list[tuple[float, float]] = field(default_factory=list)
    order: int = 2
    fields: bool = True                              # write node tables for plots

    @property
    def geo(self) -> float:
        return 2 * math.pi if self.kind == "axi" else self.depth


def magnetostatic(p: MagProblem, tags: dict[str, int]) -> str:
    """a formulation: a is the out of plane (planar) or azimuthal (axisymmetric) vector potential."""
    present = [r for r in p.regions if r.name in tags]
    if "shell" in tags and all(r.name != "shell" for r in present):
        present.append(MagRegion("shell"))
    lin = [r.name for r in present if r.bh is None]
    nl = [r.name for r in present if r.bh is not None]
    mags = [r.name for r in present if r.br is not None]
    src = [r.name for r in present if r.js]
    plain = [r.name for r in present if r.bh is None and r.br is None]
    vol = [n for n in tags if tags[n] < 100]
    extra = {"Dom": vol, "DomL": lin, "DomNL": nl, "DomM": mags, "DomS": src, "DomP": plain,
             "Dir": [d for d in p.dirichlet if d in tags]}
    force = bool(p.body) and p.air in tags
    if force:
        extra["Body"] = p.body
        extra["Air"] = [p.air]
    out = [_groups(tags, extra)]
    fn = ["Function {", "  mu0 = 4e-7*Pi;"]
    for r in present:
        g = rname(r.name)
        if r.bh is None:
            fn.append(f"  nu[{g}] = {1 / (MU0 * r.mur):.9g};")
        else:
            s, nu, w = bh_tables(*r.bh)
            fn.append(f"  nu_{g}[] = InterpolationLinear[SquNorm[$1]]{{ {_pairs(s, nu)} }};")
            fn.append(f"  dnudb2_{g}[] = dInterpolationLinear[SquNorm[$1]]{{ {_pairs(s, nu)} }};")
            fn.append(f"  w_{g}[] = InterpolationLinear[SquNorm[$1]]{{ {_pairs(s, w)} }};")
            fn.append(f"  nu[{g}] = nu_{g}[$1];")
            fn.append(f"  dnudb2[{g}] = dnudb2_{g}[$1];")
            fn.append(f"  wd[{g}] = w_{g}[$1];")
        if r.js:
            fn.append(f"  js[{g}] = Vector[0, 0, {r.js:.9g}];")
        if r.br is not None:
            fn.append(f"  br[{g}] = Vector[{r.br[0]:.9g}, {r.br[1]:.9g}, 0];")
    if nl:
        fn.append("  dhdb[] = TensorDiag[1, 1, 1] * nu[$1#1] + 2 * dnudb2[#1] * SquDyadicProduct[#1];")
    fn.append("  TM[] = (SquDyadicProduct[$1] - SquNorm[$1] * TensorDiag[0.5, 0.5, 0.5]) / mu0;")
    fn.append("}")
    out.append("\n".join(fn) + "\n")
    cons = ["Constraint {", "  { Name Dir_a; Case { { Region Dir; Value 0; } } }"]
    if force:
        cons.append("  { Name Unit; Case { { Region Body; Value 1; } } }")
    cons.append("}")
    out.append("\n".join(cons) + "\n")
    out.append(_jacobian(p.kind, p.shell))
    out.append(INTEGRATION)
    second = ("      { Name se2; NameOfCoef ae2; Function BF_PerpendicularEdge_2E; Support Dom; Entity EdgesOf[All]; }\n"
              if p.order == 2 else "")
    second_c = "      { NameOfCoef ae2; EntityType EdgesOf; NameOfConstraint Dir_a; }\n" if p.order == 2 else ""
    fs = ("FunctionSpace {\n  { Name Ha; Type Form1P; BasisFunction {\n"
          "      { Name se; NameOfCoef ae; Function BF_PerpendicularEdge; Support Dom; Entity NodesOf[All]; }\n"
          f"{second}    }}\n    Constraint {{\n      {{ NameOfCoef ae; EntityType NodesOf; NameOfConstraint Dir_a; }}\n"
          f"{second_c}    }}\n  }}\n")
    if force:
        fs += ("  { Name Hun; Type Form0; BasisFunction {\n"
               "      { Name sn; NameOfCoef un; Function BF_GroupOfNodes; Support Air; Entity GroupsOfNodesOf[Body]; } }\n"
               "    Constraint { { NameOfCoef un; EntityType GroupsOfNodesOf; NameOfConstraint Unit; } }\n  }\n")
    out.append(fs + "}\n")
    eq = ["      Integral { [ nu[] * Dof{d a}, {d a} ]; In DomL; Jacobian Vol; Integration I1; }"]
    if nl:
        eq += ["      Integral { [ nu[{d a}] * {d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }",
               "      Integral { [ dhdb[{d a}] * Dof{d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }",
               "      Integral { [ -dhdb[{d a}] * {d a}, {d a} ]; In DomNL; Jacobian Vol; Integration I1; }"]
    if mags:
        eq.append("      Integral { [ -nu[] * br[], {d a} ]; In DomM; Jacobian Vol; Integration I1; }")
    if src:
        eq.append("      Integral { [ -js[], {a} ]; In DomS; Jacobian Vol; Integration I1; }")
    q = "      { Name a; Type Local; NameOfSpace Ha; }"
    if force:
        q += "\n      { Name un; Type Local; NameOfSpace Hun; }"
        eq.append("      Integral { [ 0 * Dof{un}, {un} ]; In Air; Jacobian Vol; Integration I1; }")
    out.append("Formulation {\n  { Name M; Type FemEquation;\n    Quantity {\n" + q + "\n    }\n    Equation {\n" +
               "\n".join(eq) + "\n    }\n  }\n}\n")
    ops = ["InitSolution[A];", "Generate[A]; Solve[A];"]
    if nl:
        ops += ["Generate[A]; GetResidual[A, $res0];", "Evaluate[$res = $res0, $iter = 0];",
                'Print[{$iter, $res, $res / $res0}, Format "Residual %03g: abs %14.12e rel %14.12e"];',
                "While[$res > 1e-14 && $res / $res0 > 1e-8 && $res / $res0 <= 1 && $iter < 50] {",
                "  Solve[A]; Generate[A]; GetResidual[A, $res]; Evaluate[$iter = $iter + 1];",
                '  Print[{$iter, $res, $res / $res0}, Format "Residual %03g: abs %14.12e rel %14.12e"]; }']
    ops.append("SaveSolution[A];")
    out.append("Resolution {\n  { Name M; System { { Name A; NameOfFormulation M; } }\n    Operation {\n      " +
               "\n      ".join(ops) + "\n    }\n  }\n}\n")
    geo = f"{p.geo:.12g}"
    wq = [f"Integral {{ [ {geo} * 0.5 * nu[] * SquNorm[{{d a}}] ]; In DomP; Jacobian Vol; Integration I1; }}"]
    cq = list(wq)
    if nl:
        wq.append(f"Integral {{ [ {geo} * wd[{{d a}}] ]; In DomNL; Jacobian Vol; Integration I1; }}")
        cq.append(f"Integral {{ [ {geo} * (nu[{{d a}}] * SquNorm[{{d a}}] - wd[{{d a}}]) ]; In DomNL; Jacobian Vol; "
                  "Integration I1; }")
    if mags:
        m = f"Integral {{ [ {geo} * 0.5 * nu[] * SquNorm[{{d a}} - br[]] ]; In DomM; Jacobian Vol; Integration I1; }}"
        wq.append(m)
        cq.append(m)
    pq = ["      { Name az; Value { Local { [ CompZ[{a}] ]; In Dom; Jacobian Vol; } } }",
          "      { Name b; Value { Local { [ {d a} ]; In Dom; Jacobian Vol; } } }",
          "      { Name nb; Value { Local { [ Norm[{d a}] ]; In Dom; Jacobian Vol; } } }",
          "      { Name W; Value {\n        " + "\n        ".join(wq) + " } }",
          "      { Name hpar; Value { Local { [ (nu[] * ({d a} - br[])) * br[] / Norm[br[]] ]; In DomM; "
          "Jacobian Vol; } } }" if mags else "",
          "      { Name Wc; Value {\n        " + "\n        ".join(cq) + " } }"]
    for c in p.coils:
        if c.name in tags:
            pq.append(f"      {{ Name Psi_{rname(c.name)}; Value {{ Integral {{ [ {geo} * {c.sign * c.turns / c.area:.12g} * "
                      f"CompZ[{{a}}] ]; In {rname(c.name)}; Jacobian Vol; Integration I1; }} }} }}")
    if force:
        cx, cy = p.center
        pq.append(f"      {{ Name F; Value {{ Integral {{ [ -{geo} * TM[{{d a}}] * {{d un}} ]; In Air; Jacobian Vol; "
                  "Integration I1; } } }")
        pq.append(f"      {{ Name T; Value {{ Integral {{ [ -{geo} * CompZ[(XYZ[] - Vector[{cx}, {cy}, 0]) /\\ "
                  f"(TM[{{d a}}] * {{d un}})] ]; In Air; Jacobian Vol; Integration I1; }} }} }}")
    out.append("PostProcessing {\n  { Name M; NameOfFormulation M; Quantity {\n" + "\n".join(pq) + "\n  } }\n}\n")
    po = ['Print[ W[Dom], OnGlobal, Format Table, File "W.txt" ];',
          'Print[ Wc[Dom], OnGlobal, Format Table, File "Wc.txt" ];']
    for c in p.coils:
        if c.name in tags:
            po.append(f'Print[ Psi_{rname(c.name)}[{rname(c.name)}], OnGlobal, Format Table, '
                      f'File "Psi_{c.name}.txt" ];')
    if force:
        po.append('Print[ F[Air], OnGlobal, Format Table, File "F.txt" ];')
        po.append('Print[ T[Air], OnGlobal, Format Table, File "T.txt" ];')
    for i, (x, y) in enumerate(p.probes):
        po.append(f'Print[ b, OnPoint {{{x:.9g}, {y:.9g}, 0}}, Format Table, File "probe{i}.txt" ];')
    if p.fields:
        po.append('Print[ az, OnElementsOf Dom, Format NodeTable, File "az.txt" ];')
        po.append('Print[ nb, OnElementsOf Dom, Format Table, File "bn.txt" ];')
    if nl:
        po.append('Print[ nb, OnElementsOf DomNL, Format Table, File "bnl.txt" ];')
    if mags:
        po.append('Print[ hpar, OnElementsOf DomM, Format Table, File "hm.txt" ];')
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing M; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)
