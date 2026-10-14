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


def _jacobian(kind: str, shell: tuple | None, shell_name: str = "shell", scalar: bool = False) -> str:
    """Axisymmetric vector potentials use GetDP's squared radius mapping, nodal scalars the plain one."""
    vol = ("VolAxi" if scalar else "VolAxiSqu") if kind == "axi" else "Vol"
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
        # in a magnet b = mu h + br: the energy density is mu h^2 / 2, the co-energy density mu h^2 / 2 + br . h
        wq.append(f"Integral {{ [ {geo} * 0.5 * nu[] * SquNorm[{{d a}} - br[]] ]; In DomM; Jacobian Vol; "
                  "Integration I1; }")
        cq.append(f"Integral {{ [ {geo} * (0.5 * nu[] * SquNorm[{{d a}} - br[]] + br[] * (nu[] * ({{d a}} - br[]))) ]; "
                  "In DomM; Jacobian Vol; Integration I1; }")
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


def _space(name: str, coef: str, support: str, cons: list[tuple[str, str]], order: int, form: str = "Form0",
           extra: str = "") -> str:
    """A nodal function space, with second order hierarchical functions when order is 2."""
    bf = (f"      {{ Name sn; NameOfCoef {coef}; Function BF_Node; Support {support}; Entity NodesOf[All{extra}]; }}\n")
    cs = "".join(f"      {{ NameOfCoef {coef}; EntityType NodesOf; NameOfConstraint {c}; }}\n" for c, _ in cons)
    if order == 2:
        bf += (f"      {{ Name sn2; NameOfCoef {coef}2; Function BF_Node_2E; Support {support}; "
               "Entity EdgesOf[All]; }\n")
        cs += "".join(f"      {{ NameOfCoef {coef}2; EntityType EdgesOf; NameOfConstraint {c}; }}\n" for c, _ in cons)
    return f"  {{ Name {name}; Type {form}; BasisFunction {{\n{bf}    }}\n    Constraint {{\n{cs}    }}\n  }}\n"


@dataclass
class ElecProblem:
    kind: str                                   # planar or axi
    epsr: dict[str, float]                      # region name: relative permittivity
    potentials: dict[str, float]                # conductor surface (a boundary) name: volts
    shell: tuple | None = None
    depth: float = 1.0
    order: int = 2
    fields: bool = True
    field_regions: list[str] = field(default_factory=list)   # where the peak field is reported
    probes: list[tuple[float, float]] = field(default_factory=list)

    @property
    def geo(self) -> float:
        return 2 * math.pi if self.kind == "axi" else self.depth


def electrostatic(p: ElecProblem, tags: dict[str, int]) -> str:
    """v formulation. Every conductor is a surface held at a potential, and its charge comes back as the global
    quantity associated with that potential, so charge, energy and capacitance are computed independently."""
    vol = [n for n in tags if tags[n] < 100]
    conds = [c for c in p.potentials if c in tags]
    eps = {**{n: 1.0 for n in vol}, **{n: e for n, e in p.epsr.items() if n in tags}}
    extra = {"Dom": vol, "Cond": conds, "Peak": [r for r in p.field_regions if r in tags] or vol}
    out = [_groups(tags, extra)]
    fn = ["Function {", f"  eps0 = {EPS0!r};"] + [f"  epsr[{rname(n)}] = {e:.9g};" for n, e in eps.items()] + ["}"]
    out.append("\n".join(fn) + "\n")
    cons = "".join(f"      {{ Region {rname(c)}; Value {p.potentials[c]:.12g}; }}\n" for c in conds)
    out.append("Constraint {\n  { Name Volt; Case {\n" + cons + "  } }\n}\n")
    out.append(_jacobian(p.kind, p.shell, scalar=True))
    out.append(INTEGRATION)
    # quadratic functions live only on edges off the conductors, which stay exactly at their potential
    second = ("      { Name sn2; NameOfCoef vn2; Function BF_Node_2E; Support Dom; Entity EdgesOf[All, Not Cond]; }\n"
              if p.order == 2 else "")
    out.append("FunctionSpace {\n  { Name Hv; Type Form0; BasisFunction {\n"
               "      { Name sn; NameOfCoef vn; Function BF_Node; Support Dom; Entity NodesOf[All, Not Cond]; }\n"
               "      { Name sf; NameOfCoef vf; Function BF_GroupOfNodes; Support Dom; Entity GroupsOfNodesOf[Cond]; }\n"
               f"{second}    }}\n"
               "    GlobalQuantity { { Name U; Type AliasOf; NameOfCoef vf; } "
               "{ Name Q; Type AssociatedWith; NameOfCoef vf; } }\n"
               "    Constraint { { NameOfCoef U; EntityType GroupsOfNodesOf; NameOfConstraint Volt; } }\n  }\n}\n")
    out.append("Formulation {\n  { Name E; Type FemEquation;\n    Quantity {\n"
               "      { Name v; Type Local; NameOfSpace Hv; }\n"
               "      { Name U; Type Global; NameOfSpace Hv [U]; }\n"
               "      { Name Q; Type Global; NameOfSpace Hv [Q]; }\n    }\n    Equation {\n"
               "      Integral { [ eps0 * epsr[] * Dof{d v}, {d v} ]; In Dom; Jacobian Vol; Integration I1; }\n"
               "      GlobalTerm { [ -Dof{Q}, {U} ]; In Cond; }\n    }\n  }\n}\n")
    out.append("Resolution {\n  { Name E; System { { Name A; NameOfFormulation E; } }\n"
               "    Operation { Generate[A]; Solve[A]; SaveSolution[A]; }\n  }\n}\n")
    geo = f"{p.geo:.12g}"
    out.append("PostProcessing {\n  { Name E; NameOfFormulation E; Quantity {\n"
               "      { Name v; Value { Local { [ {v} ]; In Dom; Jacobian Vol; } } }\n"
               "      { Name e; Value { Local { [ -{d v} ]; In Dom; Jacobian Vol; } } }\n"
               "      { Name en; Value { Local { [ Norm[{d v}] ]; In Dom; Jacobian Vol; } } }\n"
               f"      {{ Name W; Value {{ Integral {{ [ {geo} * 0.5 * eps0 * epsr[] * SquNorm[{{d v}}] ]; In Dom; "
               "Jacobian Vol; Integration I1; } } }\n"
               f"      {{ Name Q; Value {{ Term {{ [ {geo} * {{Q}} ]; In Cond; }} }} }}\n  }} }}\n}}\n")
    po = ['Print[ W[Dom], OnGlobal, Format Table, File "W.txt" ];']
    po += [f'Print[ Q, OnRegion {rname(c)}, Format Table, File "Q_{c}.txt" ];' for c in conds]
    po.append('Print[ en, OnElementsOf Peak, Format Table, File "epeak.txt" ];')
    for i, (x, y) in enumerate(p.probes):
        po.append(f'Print[ e, OnPoint {{{x:.9g}, {y:.9g}, 0}}, Format Table, File "probe{i}.txt" ];')
    if p.fields:
        po.append('Print[ v, OnElementsOf Dom, Format NodeTable, File "v.txt" ];')
        po.append('Print[ en, OnElementsOf Dom, Format Table, File "en.txt" ];')
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing E; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)


@dataclass
class SoundProblem:
    kind: str                                   # planar, axi or 3d
    c: float = 343.2                            # speed of sound, m/s (air at 20 C)
    rho: float = 1.204                          # density, kg/m3
    freq: float = 0.0                           # Hz, 0 for an eigenvalue search
    inlet: str = ""                             # boundary where a plane wave of amplitude 1 Pa comes in
    outlets: list[str] = field(default_factory=list)   # anechoic plane wave ends
    radiation: tuple[str, float] | None = None  # (boundary, radius) of a spherical open boundary
    piston: tuple[str, float] | None = None     # (boundary, normal velocity in m/s into the air)
    modes: int = 0                              # eigenvalue search: how many modes
    shift: float = 0.0                          # eigenvalue search: target frequency, Hz
    order: int = 2
    fields: bool = True
    probes: list[tuple[float, float, float]] = field(default_factory=list)
    plane: tuple | None = None                  # eigen search: (z, x0, x1, y0, y1, nx, ny) to sample mode shapes on
    shapes: int = 0                             # how many mode shapes to sample

    def __post_init__(self):                    # plain floats, so the problem file never sees numpy reprs
        self.freq, self.c, self.rho, self.shift = float(self.freq), float(self.c), float(self.rho), float(self.shift)
        if self.radiation:
            self.radiation = (self.radiation[0], float(self.radiation[1]))
        if self.piston:
            self.piston = (self.piston[0], float(self.piston[1]))
        if self.plane:
            self.plane = tuple(float(v) for v in self.plane)
        self.probes = [tuple(float(v) for v in pt) for pt in self.probes]


def helmholtz(p: SoundProblem, tags: dict[str, int]) -> str:
    """Acoustic pressure, time harmonic (e^{j w t}, outgoing waves e^{-j k r}) or as an eigenvalue problem."""
    dim_vol = [n for n in tags if tags[n] < 100]
    bnd = [b for b in [p.inlet, *p.outlets, p.radiation[0] if p.radiation else "", p.piston[0] if p.piston else ""]
           if b and b in tags]
    out = [_groups(tags, {"Dom": dim_vol, "Bnd": bnd, "All_": dim_vol + bnd})]
    k = 2 * math.pi * p.freq / p.c if p.freq else 0.0
    out.append("Function {\n" + f"  k = {k!r};\n  c = {p.c!r};\n  rho = {p.rho!r};\n  w = {2 * math.pi * p.freq!r};\n}}\n")
    vol = {"axi": "VolAxi", "planar": "Vol", "3d": "Vol"}[p.kind]
    sur = {"axi": "SurAxi", "planar": "Sur", "3d": "Sur"}[p.kind]
    out.append("Jacobian {\n"
               f"  {{ Name Vol; Case {{ {{ Region All; Jacobian {vol}; }} }} }}\n"
               f"  {{ Name Sur; Case {{ {{ Region All; Jacobian {sur}; }} }} }}\n}}\n")
    out.append("Integration {\n  { Name I1; Case { { Type Gauss; Case {\n"
               "      { GeoElement Line; NumberOfPoints 4; } { GeoElement Triangle; NumberOfPoints 6; }\n"
               "      { GeoElement Tetrahedron; NumberOfPoints 15; } } } } }\n}\n")
    out.append("FunctionSpace {\n" + _space("Hp", "pn", "All_", [], p.order) + "}\n")
    eq = ["      Integral { [ Dof{d p}, {d p} ]; In Dom; Jacobian Vol; Integration I1; }"]
    if p.freq:
        eq.append("      Integral { [ -k^2 * Dof{p}, {p} ]; In Dom; Jacobian Vol; Integration I1; }")
        for b in [p.inlet, *p.outlets]:
            if b in tags:
                eq.append(f"      Integral {{ [ Complex[0, k] * Dof{{p}}, {{p}} ]; In {rname(b)}; Jacobian Sur; "
                          "Integration I1; }")
        if p.inlet in tags:
            eq.append(f"      Integral {{ [ -Complex[0, 2 * k], {{p}} ]; In {rname(p.inlet)}; Jacobian Sur; "
                      "Integration I1; }")
        if p.radiation and p.radiation[0] in tags:
            r = p.radiation[1]
            eq.append(f"      Integral {{ [ Complex[{1 / r!r}, k] * Dof{{p}}, {{p}} ]; In {rname(p.radiation[0])}; "
                      "Jacobian Sur; Integration I1; }")
        if p.piston and p.piston[0] in tags:
            eq.append(f"      Integral {{ [ -Complex[0, w * rho * {p.piston[1]!r}], {{p}} ]; In {rname(p.piston[0])}; "
                      "Jacobian Sur; Integration I1; }")
        system = "Type Complex; Frequency " + repr(p.freq)
        ops = "Generate[A]; Solve[A]; SaveSolution[A];"
    else:
        eq.append("      Integral { DtDtDof [ 1 / c^2 * Dof{p}, {p} ]; In Dom; Jacobian Vol; Integration I1; }")
        system = "Type Complex"
        ops = (f"GenerateSeparate[A]; EigenSolve[A, {max(p.modes, 1)}, {(2 * math.pi * p.shift) ** 2!r}, 0]; "
               "SaveSolutions[A];")
    out.append("Formulation {\n  { Name H; Type FemEquation;\n    Quantity { { Name p; Type Local; NameOfSpace Hp; } }\n"
               "    Equation {\n" + "\n".join(eq) + "\n    }\n  }\n}\n")
    out.append(f"Resolution {{\n  {{ Name H; System {{ {{ Name A; NameOfFormulation H; {system}; }} }}\n"
               f"    Operation {{ {ops} }}\n  }}\n}}\n")
    geo = "2 * Pi * " if p.kind == "axi" else ""
    pq = ["      { Name p; Value { Local { [ {p} ]; In All_; Jacobian Vol; } } }",
          "      { Name pa; Value { Local { [ Norm[{p}] ]; In All_; Jacobian Vol; } } }",
          f"      {{ Name S; Value {{ Integral {{ [ {geo}1 ]; In Bnd; Jacobian Sur; Integration I1; }} }} }}",
          f"      {{ Name P1; Value {{ Integral {{ [ {geo}{{p}} ]; In Bnd; Jacobian Sur; Integration I1; }} }} }}",
          f"      {{ Name P2; Value {{ Integral {{ [ {geo}SquNorm[{{p}}] ]; In Bnd; Jacobian Sur; Integration I1; }} }} }}"]
    out.append("PostProcessing {\n  { Name H; NameOfFormulation H; Quantity {\n" + "\n".join(pq) + "\n  } }\n}\n")
    po = []
    if p.freq:
        for b in bnd:
            po += [f'Print[ S[{rname(b)}], OnGlobal, Format Table, File "S_{b}.txt" ];',
                   f'Print[ P1[{rname(b)}], OnGlobal, Format Table, File "P1_{b}.txt" ];',
                   f'Print[ P2[{rname(b)}], OnGlobal, Format Table, File "P2_{b}.txt" ];']
    for i, pt in enumerate(p.probes):
        po.append(f'Print[ p, OnPoint {{{pt[0]:.9g}, {pt[1]:.9g}, {pt[2]:.9g}}}, Format Table, File "probe{i}.txt" ];')
    if p.plane and not p.freq:
        z, x0, x1, y0, y1, nx, ny = p.plane
        for i in range(1, p.shapes + 1):        # step 0 is the constant pressure mode at 0 Hz of a closed volume
            po.append(f'Print[ p, OnPlane {{{{{x0!r}, {y0!r}, {z!r}}}{{{x1!r}, {y0!r}, {z!r}}}{{{x0!r}, {y1!r}, {z!r}}}}} '
                      f'{{{int(nx)}, {int(ny)}}}, TimeStep {i}, Format SimpleTable, File "mode{i - 1}.txt" ];')
    if p.fields:
        po.append('Print[ p, OnElementsOf Dom, Format NodeTable, File "p.txt" ];')
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing H; Operation {\n    " + "\n    ".join(po or [""]) +
               "\n  } }\n}\n")
    return "".join(out)


@dataclass
class WaveProblem:
    """A wire antenna on a ground plane, axisymmetric, solved for the azimuthal magnetic field H.

    The wire is a hole in the mesh, so its surface and the ground plane are perfect conductors without any
    boundary term. A gap on the wire surface carries a uniform axial field, the delta gap feed, and the outer
    boundary absorbs outgoing waves (first order Silver-Muller condition on a sphere)."""
    freq: float
    gap: str = "gap"
    gap_length: float = 0.0
    volts: float = 1.0
    absorber: str = "outer"
    axis: str = "axis"
    order: int = 2
    probe_line: tuple | None = None             # (x, y0, y1, n): sample H along the wire surface
    fields: bool = False

    def __post_init__(self):
        self.freq, self.gap_length, self.volts = float(self.freq), float(self.gap_length), float(self.volts)
        if self.probe_line:
            self.probe_line = tuple(float(v) for v in self.probe_line)


def fullwave_axi(p: WaveProblem, tags: dict[str, int]) -> str:
    vol = [n for n in tags if tags[n] < 100]
    out = [_groups(tags, {"Dom": vol + [b for b in (p.gap, p.absorber) if b in tags], "Air": vol,
                          "Axis_": [p.axis]})]
    k0 = 2 * math.pi * p.freq / C0
    ez = p.volts / p.gap_length
    out.append("Function {\n" + f"  k0 = {k0!r};\n  we = {2 * math.pi * p.freq * EPS0!r};\n  ez = {ez!r};\n" +
               f"  eta = {MU0 * C0!r};\n}}\n")
    out.append("Constraint {\n  { Name Hax; Case { { Region Axis_; Value 0; } } }\n}\n")
    out.append("Jacobian {\n  { Name Vol; Case { { Region All; Jacobian VolAxiSqu; } } }\n"
               "  { Name Sur; Case { { Region All; Jacobian SurAxi; } } }\n}\n")
    out.append(INTEGRATION)
    second = ("      { Name se2; NameOfCoef he2; Function BF_PerpendicularEdge_2E; Support Dom; Entity EdgesOf[All]; }\n"
              if p.order == 2 else "")
    second_c = "      { NameOfCoef he2; EntityType EdgesOf; NameOfConstraint Hax; }\n" if p.order == 2 else ""
    out.append("FunctionSpace {\n  { Name Hh; Type Form1P; BasisFunction {\n"
               "      { Name se; NameOfCoef he; Function BF_PerpendicularEdge; Support Dom; Entity NodesOf[All]; }\n"
               f"{second}    }}\n    Constraint {{\n"
               "      { NameOfCoef he; EntityType NodesOf; NameOfConstraint Hax; }\n"
               f"{second_c}    }}\n  }}\n}}\n")
    out.append("Formulation {\n  { Name F; Type FemEquation;\n    Quantity { { Name h; Type Local; NameOfSpace Hh; } }\n"
               "    Equation {\n"
               "      Integral { [ Dof{d h}, {d h} ]; In Air; Jacobian Vol; Integration I1; }\n"
               "      Integral { [ -k0^2 * Dof{h}, {h} ]; In Air; Jacobian Vol; Integration I1; }\n"
               f"      Integral {{ [ Complex[0, k0] * Dof{{h}}, {{h}} ]; In {rname(p.absorber)}; Jacobian Sur; "
               "Integration I1; }\n"
               f"      Integral {{ [ Complex[0, we] * Vector[0, 0, ez], {{h}} ]; In {rname(p.gap)}; Jacobian Sur; "
               "Integration I1; }\n    }\n  }\n}\n")
    out.append("Resolution {\n  { Name F; System { { Name A; NameOfFormulation F; Type Complex; "
               f"Frequency {p.freq!r}; }} }}\n    Operation {{ Generate[A]; Solve[A]; SaveSolution[A]; }}\n  }}\n}}\n")
    pq = ["      { Name h; Value { Local { [ CompZ[{h}] ]; In Dom; Jacobian Vol; } } }",
          "      { Name hn; Value { Local { [ Norm[{h}] ]; In Dom; Jacobian Vol; } } }",
          f"      {{ Name Ig; Value {{ Integral {{ [ CompZ[{{h}}] * 2 * Pi / {p.gap_length!r} ]; In {rname(p.gap)}; "
          "Jacobian Sur; Integration I1; } } }",
          f"      {{ Name Prad; Value {{ Integral {{ [ 0.5 * eta * SquNorm[{{h}}] * 2 * Pi ]; In {rname(p.absorber)}; "
          "Jacobian Sur; Integration I1; } } }"]
    out.append("PostProcessing {\n  { Name F; NameOfFormulation F; Quantity {\n" + "\n".join(pq) + "\n  } }\n}\n")
    po = [f'Print[ Ig[{rname(p.gap)}], OnGlobal, Format Table, File "ig.txt" ];',
          f'Print[ Prad[{rname(p.absorber)}], OnGlobal, Format Table, File "prad.txt" ];']
    if p.probe_line:
        x, y0, y1, n = p.probe_line
        po.append(f'Print[ h, OnLine {{{{{x!r}, {y0!r}, 0}}{{{x!r}, {y1!r}, 0}}}} {{{int(n)}}}, Format Table, '
                  'File "hwire.txt" ];')
    if p.fields:
        po.append('Print[ h, OnElementsOf Air, Format NodeTable, File "h.txt" ];')
    out.append("PostOperation {\n  { Name P; NameOfPostProcessing F; Operation {\n    " + "\n    ".join(po) +
               "\n  } }\n}\n")
    return "".join(out)
