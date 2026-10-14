"""OpenFOAM cases written from Python: dictionaries, blockMesh grids, boundary fields, solver settings and the
function objects Wobbl reads its numbers from, plus readers for what comes back.

Every grid is a structured blockMesh grid, which meshes in a second, never fails on a valid shape and refines by
a known ratio, which is what a grid convergence index needs. Turbulent cases use k omega SST with the blended wall
functions of OpenFOAM's ESI line (nutUSpaldingWallFunction and omegaWallFunction), which hold from a resolved
wall at y+ of 1 to a log layer wall at 30 to 300, so the grid decides the wall treatment and the y+ check reports
which one ran.
"""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER = "FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {cls};\n    object      {obj};\n}}\n\n"
PRANDTL_T = 0.85      # turbulent Prandtl number


def val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(val(x) for x in v) + ")"
    return str(v)


def entries(d: dict, ind: int = 0) -> str:
    """A Python dict as OpenFOAM dictionary entries. Nested dicts become sub dictionaries."""
    pad = "    " * ind
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}\n{pad}{{\n{entries(v, ind + 1)}{pad}}}\n")
        else:
            out.append(f"{pad}{k:<15} {val(v)};\n")
    return "".join(out)


def uniform(v) -> str:
    return f"uniform {val(v)}"


@dataclass
class Turbulence:
    """What a case solves for the turbulence: laminar, rans (k omega SST), des (Spalart Allmaras IDDES) or les
    (WALE with a wall model). k and omega are the free stream values for RANS, nu_tilda for DES."""
    model: str = "laminar"
    k: float = 0.0
    omega: float = 0.0
    nu_tilda: float = 0.0

    @classmethod
    def free_stream(cls, model: str, u: float, nu: float, length: float, intensity: float = 0.05,
                    scale: float = 0.07) -> "Turbulence":
        """Inlet turbulence from an intensity and a length scale, a fraction of the hydraulic diameter or body."""
        k = 1.5 * (intensity * u) ** 2
        omega = math.sqrt(k) / (0.09 ** 0.25 * scale * length)
        return cls(model, k, omega, 3 * nu)

    @property
    def fields(self) -> list[str]:
        return {"laminar": [], "rans": ["k", "omega", "nut"], "des": ["nuTilda", "nut"],
                "les": ["nut"]}[self.model]


@dataclass
class Patch:
    name: str
    kind: str                # inlet, outlet, wall, symmetry, empty, freestream, cyclic, wedge
    faces: list[tuple[int, ...]] = field(default_factory=list)
    neighbour: str = ""


class Case:
    """A case folder. Write the grid, the fields and the settings, then run programs through the adapter."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        for d in ("system", "constant", "0"):
            (self.root / d).mkdir(parents=True)

    def write(self, rel: str, obj: str, body: str | dict, cls: str = "dictionary"):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(HEADER.format(cls=cls, obj=obj) + (entries(body) if isinstance(body, dict) else body))

    def run(self, app: str, *args: str, timeout: float = 1800, log_name: str = "") -> str:
        from ..solvers.base import get
        return get("openfoam").run(app, self.root, args, timeout, log_name)

    def clean(self):
        shutil.rmtree(self.root, ignore_errors=True)


# --- grids ------------------------------------------------------------------------------------------------------
def geometric(first: float, total: float, n: int) -> float:
    """blockMesh grading (last cell over first) for n cells over a length whose first cell is about first."""
    if n < 2 or first * n >= total:
        return 1.0
    lo, hi = 1.0 + 1e-9, 2.0
    while first * (hi ** n - 1) / (hi - 1) < total:
        hi *= 1.5
    for _ in range(100):
        q = 0.5 * (lo + hi)
        lo, hi = (q, hi) if first * (q ** n - 1) / (q - 1) < total else (lo, q)
    return float(q ** (n - 1))


def block_mesh(case: Case, vertices, blocks: list[str], patches: list[Patch], edges: list[str] = (),
               scale: float = 1.0):
    vs = "\n".join(f"    ({val(x)} {val(y)} {val(z)})" for x, y, z in vertices)
    bounds = []
    for p in patches:
        kind = {"inlet": "patch", "outlet": "patch", "freestream": "patch", "symmetry": "symmetryPlane"}.get(p.kind,
                                                                                                           p.kind)
        extra = f"\n        neighbourPatch {p.neighbour};" if p.kind == "cyclic" else ""
        faces = "\n".join(f"            ({' '.join(map(str, f))})" for f in p.faces)
        bounds.append(f"    {p.name}\n    {{\n        type {kind};{extra}\n        faces\n        (\n{faces}\n"
                      f"        );\n    }}")
    body = (f"scale {val(scale)};\n\nvertices\n(\n{vs}\n);\n\nblocks\n(\n" + "\n".join(f"    {b}" for b in blocks) +
            "\n);\n\nedges\n(\n" + "\n".join(f"    {e}" for e in edges) + "\n);\n\nboundary\n(\n" +
            "\n".join(bounds) + "\n);\n")
    case.write("system/blockMeshDict", "blockMeshDict", body)
    case.patches = patches


def channel(case: Case, length: float, height: float, nx: int, ny: int, gx: float = 1.0, gy: float = 1.0,
            depth: float | None = None, top: str = "symmetry", lead: float = 0.0):
    """A 2D channel from x 0 to length and y 0 (a wall) to height, one cell deep. top is 'symmetry' for half a
    channel or 'wall'. gx and gy are blockMesh gradings, last cell over first. lead adds that much slip walled
    approach before x 0, so the air meets the wall's leading edge inside the grid rather than at the inlet."""
    dz = depth or height / ny
    L, h = length, height
    if not lead:
        v = [(0, 0, 0), (L, 0, 0), (L, h, 0), (0, h, 0), (0, 0, dz), (L, 0, dz), (L, h, dz), (0, h, dz)]
        pats = [Patch("inlet", "inlet", [(0, 4, 7, 3)]), Patch("outlet", "outlet", [(1, 2, 6, 5)]),
                Patch("wall", "wall", [(0, 1, 5, 4)]), Patch("top", top, [(3, 7, 6, 2)]),
                Patch("frontAndBack", "empty", [(0, 3, 2, 1), (4, 5, 6, 7)])]
        block_mesh(case, v, [f"hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading ({val(gx)} {val(gy)} 1)"], pats)
    else:
        q = gx ** (1 / max(nx - 1, 1))
        first = L * (q - 1) / (q ** nx - 1) if q != 1 else L / nx
        nl = max(4, int(round(math.log(1 + lead * (q - 1) / first) / math.log(q)))) if q != 1 else \
            max(4, int(round(lead / first)))
        gl = 1 / geometric(first, lead, nl)
        v = [(-lead, 0, 0), (0, 0, 0), (L, 0, 0), (L, h, 0), (0, h, 0), (-lead, h, 0)]
        v += [(x, y, dz) for x, y, _ in v]
        pats = [Patch("inlet", "inlet", [(0, 6, 11, 5)]), Patch("outlet", "outlet", [(2, 3, 9, 8)]),
                Patch("wall", "wall", [(1, 2, 8, 7)]), Patch("approach", "symmetry", [(0, 1, 7, 6)]),
                Patch("top", top, [(5, 11, 10, 4), (4, 10, 9, 3)]),
                Patch("frontAndBack", "empty", [(0, 5, 4, 1), (1, 4, 3, 2), (6, 7, 10, 11), (7, 8, 9, 10)])]
        block_mesh(case, v, [f"hex (0 1 4 5 6 7 10 11) ({nl} {ny} 1) simpleGrading ({val(gl)} {val(gy)} 1)",
                             f"hex (1 2 3 4 7 8 9 10) ({nx} {ny} 1) simpleGrading ({val(gx)} {val(gy)} 1)"], pats)
    case.depth = dz


def pipe_wedge(case: Case, radius: float, length: float, nr: int, gr: float, nx: int = 1, angle: float = 5.0,
               periodic: bool = True):
    """An axisymmetric wedge of a round pipe about the x axis, the axis edge collapsed, nr cells from the axis to
    the wall graded by gr (last over first, below 1 to crowd the wall). periodic joins the two ends."""
    a = math.radians(angle / 2)
    c, s = radius, radius * math.tan(a)      # the wall's face centres sit on the true radius
    v = [(0, 0, 0), (length, 0, 0), (length, c, -s), (0, c, -s), (length, c, s), (0, c, s)]
    ends = ("cyclic", "cyclic") if periodic else ("inlet", "outlet")
    pats = [Patch("inlet", ends[0], [(0, 3, 5, 0)], "outlet"), Patch("outlet", ends[1], [(1, 4, 2, 1)], "inlet"),
            Patch("wall", "wall", [(3, 2, 4, 5)]), Patch("front", "wedge", [(0, 1, 2, 3)]),
            Patch("back", "wedge", [(0, 5, 4, 1)])]
    block_mesh(case, v, [f"hex (0 1 2 3 0 1 4 5) ({nx} {nr} 1) simpleGrading (1 {val(gr)} 1)"], pats)
    case.wedge = 2 * a
    q = gr ** (1 / (nr - 1)) if nr > 1 else 1.0
    sizes = q ** np.arange(nr)
    edges = np.concatenate([[0.0], np.cumsum(sizes)]) / sizes.sum() * radius
    case.radii = 0.5 * (edges[1:] + edges[:-1])          # cell centres from the axis, one per cell, in order


def cylinder_ogrid(case: Case, d: float, far: float, n_theta: int, n_r: int, first: float, span: float = 0.0,
                   n_z: int = 1):
    """An O grid around a cylinder of diameter d at the origin, out to a circle of radius far, as four blocks of
    n_theta cells around by n_r cells out, the first cell first thick. span 0 makes a 2D grid one cell deep;
    a span with n_z cells makes a 3D grid whose ends are joined periodically."""
    r = d / 2
    gr = geometric(first, far - r, n_r)
    z1 = span or (math.pi * d / (4 * n_theta))
    v = []
    for z in (0.0, z1):
        for k in range(4):
            t = math.radians(45 + 90 * k)
            v += [(r * math.cos(t), r * math.sin(t), z), (far * math.cos(t), far * math.sin(t), z)]
    blocks, edges = [], []
    cyl, out, front, back = [], [], [], []
    for k in range(4):
        i0, o0, i1, o1 = 2 * k, 2 * k + 1, (2 * k + 2) % 8, (2 * k + 3) % 8
        blocks.append(f"hex ({i0} {o0} {o1} {i1} {i0 + 8} {o0 + 8} {o1 + 8} {i1 + 8}) ({n_r} {n_theta} {n_z}) "
                      f"simpleGrading ({val(gr)} 1 1)")
        t = math.radians(90 + 90 * k)
        for zo, z in ((0, 0.0), (8, z1)):
            edges.append(f"arc {i0 + zo} {i1 + zo} ({val(r * math.cos(t))} {val(r * math.sin(t))} {val(z)})")
            edges.append(f"arc {o0 + zo} {o1 + zo} ({val(far * math.cos(t))} {val(far * math.sin(t))} {val(z)})")
        cyl.append((i0, i1, i1 + 8, i0 + 8))
        out.append((o0, o0 + 8, o1 + 8, o1))
        front.append((i0, i1, o1, o0))
        back.append((i0 + 8, o0 + 8, o1 + 8, i1 + 8))
    pats = [Patch("body", "wall", cyl), Patch("farfield", "freestream", out)]
    if span:
        pats += [Patch("front", "cyclic", front, "back"), Patch("back", "cyclic", back, "front")]
    else:
        pats.append(Patch("frontAndBack", "empty", front + back))
    block_mesh(case, v, blocks, pats, edges)
    case.depth = z1


# --- fields -----------------------------------------------------------------------------------------------------
def write_fields(case: Case, u: tuple[float, float, float], turb: Turbulence, scalar: dict | None = None,
                 p_outlet: float = 0.0, u_cells: np.ndarray | None = None):
    """Initial and boundary fields for every patch kind. scalar is {"inlet": value, "wall": value} for a passive
    temperature T. u_cells starts the velocity from a profile, one vector per cell, instead of uniform."""
    pats = case.patches
    U, P, K, W, NUT, NT, T = {}, {}, {}, {}, {}, {}, {}
    for p in pats:
        n = p.name
        if p.kind in ("empty", "wedge", "cyclic", "symmetry"):
            t = {"empty": "empty", "wedge": "wedge", "cyclic": "cyclic", "symmetry": "symmetryPlane"}[p.kind]
            for f in (U, P, K, W, NUT, NT, T):
                f[n] = {"type": t}
        elif p.kind == "inlet":
            U[n] = {"type": "fixedValue", "value": uniform(u)}
            P[n] = {"type": "zeroGradient"}
            K[n] = {"type": "fixedValue", "value": uniform(turb.k)}
            W[n] = {"type": "fixedValue", "value": uniform(turb.omega)}
            NT[n] = {"type": "fixedValue", "value": uniform(turb.nu_tilda)}
            NUT[n] = {"type": "calculated", "value": uniform(0)}
            T[n] = {"type": "fixedValue", "value": uniform((scalar or {}).get("inlet", 0.0))}
        elif p.kind == "outlet":
            U[n] = {"type": "inletOutlet", "inletValue": uniform((0, 0, 0)), "value": uniform(u)}
            P[n] = {"type": "fixedValue", "value": uniform(p_outlet)}
            for f, v in ((K, turb.k), (W, turb.omega), (NT, turb.nu_tilda), (T, (scalar or {}).get("inlet", 0.0))):
                f[n] = {"type": "inletOutlet", "inletValue": uniform(v), "value": uniform(v)}
            NUT[n] = {"type": "calculated", "value": uniform(0)}
        elif p.kind == "freestream":
            U[n] = {"type": "freestreamVelocity", "freestreamValue": uniform(u), "value": uniform(u)}
            P[n] = {"type": "freestreamPressure", "freestreamValue": uniform(0), "value": uniform(0)}
            for f, v in ((K, turb.k), (W, turb.omega), (NT, turb.nu_tilda), (T, (scalar or {}).get("inlet", 0.0))):
                f[n] = {"type": "inletOutlet", "inletValue": uniform(v), "value": uniform(v)}
            NUT[n] = {"type": "calculated", "value": uniform(0)}
        elif p.kind == "wall":
            U[n] = {"type": "noSlip"}
            P[n] = {"type": "zeroGradient"}
            K[n] = {"type": "kqRWallFunction", "value": uniform(turb.k)}
            W[n] = {"type": "omegaWallFunction", "value": uniform(turb.omega)}
            NT[n] = {"type": "fixedValue", "value": uniform(0)}
            NUT[n] = {"type": "nutUSpaldingWallFunction" if turb.model in ("rans", "les") else "zeroGradient",
                      "value": uniform(0)}
            if turb.model == "des":
                NUT[n] = {"type": "nutUSpaldingWallFunction", "value": uniform(0)}
            T[n] = {"type": "fixedValue", "value": uniform((scalar or {}).get("wall", 1.0))}
    inside = uniform(u) if u_cells is None else (f"nonuniform List<vector> {len(u_cells)}\n(\n" + "\n".join(
        f"({x:.6g} {y:.6g} {z:.6g})" for x, y, z in u_cells) + "\n)")
    case.write("0/U", "U", f"dimensions [0 1 -1 0 0 0 0];\n\ninternalField {inside};\n\nboundaryField\n{{\n"
                           f"{entries(U, 1)}}}\n", "volVectorField")
    case.write("0/p", "p", f"dimensions [0 2 -2 0 0 0 0];\n\ninternalField uniform 0;\n\nboundaryField\n{{\n"
                           f"{entries(P, 1)}}}\n", "volScalarField")
    for name, f, dims, v in (("k", K, "[0 2 -2 0 0 0 0]", turb.k), ("omega", W, "[0 0 -1 0 0 0 0]", turb.omega),
                             ("nut", NUT, "[0 2 -1 0 0 0 0]", 0.0), ("nuTilda", NT, "[0 2 -1 0 0 0 0]",
                                                                      turb.nu_tilda)):
        if name in turb.fields:
            case.write(f"0/{name}", name, f"dimensions {dims};\n\ninternalField {uniform(v)};\n\nboundaryField\n"
                                          f"{{\n{entries(f, 1)}}}\n", "volScalarField")
    if scalar is not None:
        case.write("0/T", "T", f"dimensions [0 0 0 0 0 0 0];\n\ninternalField {uniform(scalar.get('inlet', 0.0))};"
                               f"\n\nboundaryField\n{{\n{entries(T, 1)}}}\n", "volScalarField")


# --- settings ---------------------------------------------------------------------------------------------------
def write_physics(case: Case, nu: float, turb: Turbulence):
    case.write("constant/transportProperties", "transportProperties",
               {"transportModel": "Newtonian", "nu": nu})
    if turb.model == "laminar":
        body = {"simulationType": "laminar"}
    elif turb.model == "rans":
        body = {"simulationType": "RAS", "RAS": {"RASModel": "kOmegaSST", "turbulence": "on", "printCoeffs": "off"}}
    elif turb.model == "des":
        body = {"simulationType": "LES", "LES": {"LESModel": "SpalartAllmarasIDDES", "turbulence": "on",
                                                 "printCoeffs": "off", "delta": "IDDESDelta",
                                                 "IDDESDeltaCoeffs": {"hmax": "maxDeltaxyzCubeRoot",
                                                                      "maxDeltaxyzCubeRootCoeffs": {}}}}
    else:
        body = {"simulationType": "LES", "LES": {"LESModel": "WALE", "turbulence": "on", "printCoeffs": "off",
                                                 "delta": "cubeRootVol", "cubeRootVolCoeffs": {"deltaCoeff": 1}}}
    case.write("constant/turbulenceProperties", "turbulenceProperties", body)


def write_steady(case: Case, turb: Turbulence, iterations: int, functions: str, scalar: bool = False,
                 tol: float = 1e-5, app: str = "simpleFoam", relax: tuple[float, float] = (0.9, 0.8),
                 small: bool = False):
    """simpleFoam with SIMPLEC, stopped when every residual is under tol. relax is the under relaxation of the
    velocity and of the turbulence and temperature equations. small grids solve the pressure with conjugate
    gradients, which set up faster than multigrid below a few thousand cells."""
    div_t = {"div(phi,T)": "bounded Gauss limitedLinear 1"} if scalar else {}
    case.write("system/controlDict", "controlDict", entries(
        {"application": app, "startFrom": "startTime", "startTime": 0, "stopAt": "endTime", "endTime": iterations,
         "deltaT": 1, "writeControl": "timeStep", "writeInterval": iterations, "purgeWrite": 1,
         "writeFormat": "ascii", "writePrecision": 8, "writeCompression": "off", "timeFormat": "general",
         "timePrecision": 8, "runTimeModifiable": "false"}) + f"\nfunctions\n{{\n{functions}}}\n")
    case.write("system/fvSchemes", "fvSchemes", {
        "ddtSchemes": {"default": "steadyState"},
        "gradSchemes": {"default": "Gauss linear", "grad(U)": "cellLimited Gauss linear 1"},
        "divSchemes": {"default": "none", "div(phi,U)": "bounded Gauss linearUpwind grad(U)",
                       "div(phi,k)": "bounded Gauss limitedLinear 1", "div(phi,omega)": "bounded Gauss limitedLinear 1",
                       "div((nuEff*dev2(T(grad(U)))))": "Gauss linear", **div_t},
        "laplacianSchemes": {"default": "Gauss linear limited corrected 0.5"},
        "interpolationSchemes": {"default": "linear"}, "snGradSchemes": {"default": "limited corrected 0.5"},
        "wallDist": {"method": "meshWave"}})
    resid = {"p": tol, "U": tol / 10}
    if turb.model == "rans":
        resid['"(k|omega)"'] = tol
    if scalar:
        resid["T"] = tol / 10
    case.write("system/fvSolution", "fvSolution", {
        "solvers": {"p": ({"solver": "PCG", "preconditioner": "DIC"} if small else
                          {"solver": "GAMG", "smoother": "GaussSeidel"}) | {"tolerance": 1e-9, "relTol": 0.05},
                    '"(U|k|omega|T)"': {"solver": "smoothSolver", "smoother": "symGaussSeidel", "tolerance": 1e-10,
                                        "relTol": 0.1}},
        "SIMPLE": {"consistent": "yes", "nNonOrthogonalCorrectors": 0, "residualControl": resid, **_p_ref(case)},
        "relaxationFactors": {"equations": {"U": relax[0], '"(k|omega)"': relax[1], "T": 1.0}}})


def write_transient(case: Case, turb: Turbulence, end: float, dt: float, functions: str, write_every: float,
                    max_co: float = 0.9, keep: int = 1):
    """pimpleFoam in PISO mode with an adjustable time step held under max_co, keeping the last keep writes."""
    case.write("system/controlDict", "controlDict", entries(
        {"application": "pimpleFoam", "startFrom": "startTime", "startTime": 0, "stopAt": "endTime", "endTime": end,
         "deltaT": dt, "writeControl": "adjustable", "writeInterval": write_every, "purgeWrite": keep,
         "writeFormat": "ascii", "writePrecision": 8, "writeCompression": "off", "timeFormat": "general",
         "timePrecision": 8, "runTimeModifiable": "false", "adjustTimeStep": "yes", "maxCo": max_co}) +
               f"\nfunctions\n{{\n{functions}}}\n")
    case.write("system/fvSchemes", "fvSchemes", {
        "ddtSchemes": {"default": "backward"},
        "gradSchemes": {"default": "Gauss linear"},
        "divSchemes": {"default": "none", "div(phi,U)": "Gauss LUST grad(U)",
                       "div(phi,nuTilda)": "Gauss limitedLinear 1", "div(phi,k)": "Gauss limitedLinear 1",
                       "div(phi,omega)": "Gauss limitedLinear 1", "div((nuEff*dev2(T(grad(U)))))": "Gauss linear"},
        "laplacianSchemes": {"default": "Gauss linear limited corrected 0.33"},
        "interpolationSchemes": {"default": "linear"}, "snGradSchemes": {"default": "limited corrected 0.33"},
        "wallDist": {"method": "meshWave"}})
    case.write("system/fvSolution", "fvSolution", {
        "solvers": {"p": {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": 1e-7, "relTol": 0.01},
                    "pFinal": {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": 1e-7, "relTol": 0},
                    '"(U|k|omega|nuTilda)(Final)?"': {"solver": "smoothSolver", "smoother": "symGaussSeidel",
                                                      "tolerance": 1e-8, "relTol": 0}},
        "PIMPLE": {"momentumPredictor": "yes", "nOuterCorrectors": 1, "nCorrectors": 2,
                   "nNonOrthogonalCorrectors": 0, **_p_ref(case)}})


def _p_ref(case: Case) -> dict:
    """A closed or periodic domain has no boundary that fixes the pressure level, so one cell does."""
    open_ = any(p.kind in ("outlet", "freestream") for p in getattr(case, "patches", []))
    return {} if open_ else {"pRefCell": 0, "pRefValue": 0}


# --- function objects -------------------------------------------------------------------------------------------
def fo_patch_sum(name: str, patch: str, field_: str = "phi", operation: str = "sum", weight: str = "") -> str:
    w = f" weightField {weight};" if weight else ""
    return (f"    {name}\n    {{\n        type surfaceFieldValue; libs (fieldFunctionObjects); log false;\n"
            f"        writeControl timeStep; writeInterval 1; writeFields false; regionType patch; name {patch};\n"
            f"        operation {operation};{w} fields ({field_});\n    }}\n")


def fo_forces(patch: str, u: float, lref: float, aref: float, lift=(0, 1, 0), drag=(1, 0, 0)) -> str:
    return (f"    forces\n    {{\n        type forceCoeffs; libs (forces); log false; writeControl timeStep;\n"
            f"        writeInterval 1; patches ({patch}); rho rhoInf; rhoInf 1; CofR (0 0 0);\n"
            f"        liftDir {val(lift)}; dragDir {val(drag)}; pitchAxis (0 0 1); magUInf {val(u)};\n"
            f"        lRef {val(lref)}; Aref {val(aref)};\n    }}\n")


def fo_yplus() -> str:
    return ("    yPlus\n    {\n        type yPlus; libs (fieldFunctionObjects); log false; writeControl writeTime;\n"
            "    }\n")


def fo_scalar(pr: float) -> str:
    """Temperature as a passive scalar carried by the flow: laminar diffusivity nu / Pr, turbulent nut / Prt."""
    return (f"    T\n    {{\n        type scalarTransport; libs (solverFunctionObjects); field T;\n"
            f"        alphaD {val(1 / pr)}; alphaDt {val(1 / PRANDTL_T)}; nCorr 3; bounded01 false;\n"
            f"        writeControl writeTime;\n    }}\n")


def fo_average(start: float, nut: bool = False) -> str:
    """Time averages from start on: the mean velocity and its fluctuations, and the mean eddy viscosity."""
    extra = "            nut { mean on; prime2Mean off; base time; }\n" if nut else ""
    return (f"    average\n    {{\n        type fieldAverage; libs (fieldFunctionObjects); writeControl writeTime;\n"
            f"        timeStart {val(start)};\n        fields\n        (\n"
            f"            U {{ mean on; prime2Mean on; base time; }}\n{extra}        );\n    }}\n")


def fo_cell_volumes() -> str:
    return ("    volumes\n    {\n        type writeCellVolumes; libs (fieldFunctionObjects); writeControl writeTime;\n"
            "    }\n")


def fo_cell_centres() -> str:
    return ("    centres\n    {\n        type writeCellCentres; libs (fieldFunctionObjects); writeControl writeTime;\n"
            "    }\n")


# --- reading results --------------------------------------------------------------------------------------------
def table(case: Case, name: str, file: str | None = None) -> tuple[list[str], np.ndarray]:
    """The columns and rows of a function object's table, joining restarts, with vectors flattened."""
    root = case.root / "postProcessing" / name
    files = sorted(root.glob(f"*/{file}" if file else "*/*.dat"), key=lambda p: float(p.parent.name))
    header, rows = [], []
    for f in files:
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                cols = line[1:].split()
                if cols and cols[0].lower() == "time":
                    header = cols
                continue
            parts = line.replace("(", " ").replace(")", " ").split()
            if parts:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    rows.append([float(x) if _isnum(x) else np.nan for x in parts])
    width = max((len(r) for r in rows), default=0)
    arr = np.array([r + [np.nan] * (width - len(r)) for r in rows]) if rows else np.zeros((0, 0))
    return header, arr


def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def column(case: Case, name: str, col: str, file: str | None = None) -> np.ndarray:
    head, arr = table(case, name, file)
    if not len(arr):
        raise KeyError(f"no data from {name}")
    i = head.index(col) if col in head else (1 if col == "value" else None)
    if i is None:
        raise KeyError(f"{name} has no column {col}; it has {head}")
    return arr[:, i]


def last_time(case: Case) -> Path:
    times = [p for p in case.root.iterdir() if p.is_dir() and _isnum(p.name) and float(p.name) > 0]
    if not times:
        raise FileNotFoundError("the run wrote no results")
    return max(times, key=lambda p: float(p.name))


def read_field(path: Path) -> np.ndarray:
    """The internal values of a volScalarField or volVectorField written as ASCII."""
    text = path.read_text()
    m = re.search(r"internalField\s+(uniform|nonuniform)\s+", text)
    if m.group(1) == "uniform":
        v = text[m.end():text.index(";", m.end())].replace("(", " ").replace(")", " ")
        return np.array([float(x) for x in v.split()])
    n_m = re.search(r"List<(\w+)>\s*(\d+)\s*\(", text[m.end():])
    start = m.end() + n_m.end()
    end = text.index("\n)", start)
    vals = np.array(text[start:end].replace("(", " ").replace(")", " ").split(), dtype=float)
    n = int(n_m.group(2))
    return vals.reshape(n, -1) if n_m.group(1) == "vector" else vals
