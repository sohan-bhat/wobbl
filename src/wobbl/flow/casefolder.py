"""Run an OpenFOAM case folder someone already has, as it is, and check what came out.

The case is copied first, so the user's folder is never touched. A case without a mesh is meshed by blockMesh, and
by snappyHexMesh when it has a snappyHexMeshDict. Its application runs single core; icoFoam cases, like the lid
driven cavity, run on pimpleFoam in PISO mode, which solves the same laminar equations, since the lean install
keeps simpleFoam and pimpleFoam only. The answer reports how far the run got, how far the residuals fell, the
continuity error, the mesh quality and the last value of every function object it wrote.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..units import fmt
from ..verify.checks import Check, confidence
from . import foam
from .verify import residual_check, residuals

APPS = ("simpleFoam", "pimpleFoam")


def _dict_value(text: str, key: str) -> str:
    m = re.search(rf"^\s*{key}\s+([^;]+);", text, re.M)
    return m.group(1).strip() if m else ""


def _laminar_pimple(case: Path) -> list[str]:
    """What turning an icoFoam case into a laminar pimpleFoam one changes."""
    done = []
    tp = case / "constant" / "turbulenceProperties"
    if not tp.exists() and not (case / "constant" / "momentumTransport").exists():
        tp.write_text(foam.HEADER.format(cls="dictionary", obj="turbulenceProperties") + "simulationType laminar;\n")
        done.append("laminar turbulenceProperties")
    tr = case / "constant" / "transportProperties"
    if tr.exists() and "transportModel" not in tr.read_text():
        tr.write_text(tr.read_text().rstrip() + "\ntransportModel Newtonian;\n")
    sol = case / "system" / "fvSolution"
    text = sol.read_text()
    if "PIMPLE" not in text:
        m = re.search(r"PISO\s*\{([^}]*)\}", text)
        inner = m.group(1) if m else "nCorrectors 2; nNonOrthogonalCorrectors 0;"
        text += f"\nPIMPLE\n{{\n    momentumPredictor no;\n    nOuterCorrectors 1;\n{inner}\n}}\n"
        done.append("a PIMPLE dictionary from its PISO settings")
    if "Final" not in text:
        text = re.sub(r"(\n\s*)U\s*\n(\s*)\{", r"\1\"(U|UFinal)\"\n\2{", text, count=1)
        text = re.sub(r"(\n\s*)p\s*\n(\s*)\{", r"\1\"(p|pFinal)\"\n\2{", text, count=1)
    sol.write_text(text)
    sch = case / "system" / "fvSchemes"
    text = sch.read_text()
    if "dev2(T(grad(U)))" not in text:      # the viscous term pimpleFoam writes in full
        text = re.sub(r"divSchemes\s*\{", "divSchemes\n{\n    div((nuEff*dev2(T(grad(U))))) Gauss linear;", text,
                      count=1)
        sch.write_text(text)
        done.append("the full viscous term's scheme")
    cd = case / "system" / "controlDict"
    cd.write_text(re.sub(r"application\s+icoFoam;", "application pimpleFoam;", cd.read_text()))
    return done


def run_case(src: Path, out_dir: str | Path, progress=None, quick: bool = False):
    from ..outcome_generic import GenericOutcome
    from ..solvers.base import get
    from .install import ensure

    t0 = time.time()
    say = progress or (lambda a, b: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case_dir = out / "case"
    shutil.rmtree(case_dir, ignore_errors=True)
    shutil.copytree(src, case_dir, ignore=shutil.ignore_patterns("processor*", "postProcessing", "log.*", "*.foam"))
    for d in case_dir.iterdir():     # start from the initial conditions, not from someone's earlier results
        if d.is_dir() and foam._isnum(d.name) and float(d.name) > 0:
            shutil.rmtree(d)
    ensure(lambda m: say("setup", m))
    cd_text = (case_dir / "system" / "controlDict").read_text()
    app = _dict_value(cd_text, "application")
    notes = []
    if app == "icoFoam":
        notes = _laminar_pimple(case_dir)
        notes.insert(0, "icoFoam ran as pimpleFoam in PISO mode, the same laminar equations")
        app = "pimpleFoam"
    if app not in APPS:
        raise ValueError(f"this case runs {app or 'no application'}, and Wobbl's OpenFOAM has {', '.join(APPS)} "
                         "(plus icoFoam cases, which run on pimpleFoam)")
    c = foam.Case.__new__(foam.Case)
    c.root = case_dir
    if not (case_dir / "constant" / "polyMesh" / "faces").exists() and \
            not (case_dir / "constant" / "polyMesh" / "faces.gz").exists():
        if not (case_dir / "system" / "blockMeshDict").exists():
            raise ValueError("the case has no mesh and no system/blockMeshDict to make one")
        say("mesh", "blockMesh")
        c.run("blockMesh")
        if (case_dir / "system" / "snappyHexMeshDict").exists():
            if (case_dir / "system" / "surfaceFeatureExtractDict").exists():
                c.run("surfaceFeatureExtract")
            say("mesh", "snappyHexMesh")
            c.run("snappyHexMesh", "-overwrite", timeout=7200)
    chk = c.run("checkMesh", check=False)
    say("solve", app)
    log = c.run(app, timeout=7200)
    steady = app == "simpleFoam"
    times = [float(x) for x in re.findall(r"^Time = ([\d.eE+-]+)", log, re.M)]
    end = float(_dict_value(cd_text, "endTime") or (times[-1] if times else 0))
    checks = []
    cells = re.search(r"cells:\s+(\d+)", chk)
    failed = re.search(r"Failed (\d+) mesh checks", chk)
    no = re.search(r"Mesh non-orthogonality Max:\s*([\d.eE+-]+)", chk)
    checks.append(Check("mesh quality", "warn" if failed else "pass",
                        f"{int(cells.group(1)) if cells else 0:,} cells, worst non orthogonality "
                        f"{float(no.group(1)) if no else 0:.0f} degrees" +
                        (f", {failed.group(1)} checkMesh checks failed" if failed else
                         ", every checkMesh check passed")))
    reached = times[-1] if times else 0.0
    checks.append(Check("completion", "pass" if reached >= 0.999 * end or steady else "warn",
                        f"ran to {fmt(reached)} of {fmt(end)} in {len(times)} steps"))
    checks.append(residual_check(log, 1e-4, steady=steady))
    cont = re.findall(r"time step continuity errors : sum local = ([\d.eE+-]+)", log)
    if cont:
        worst = max(float(x) for x in cont[-max(1, len(cont) // 5):])
        checks.append(Check("continuity", "pass" if worst < 1e-5 else ("warn" if worst < 1e-3 else "fail"),
                            f"the local continuity error stays at or below {worst:.2g} over the last fifth of the run"))
    level, reasons = confidence(checks, [("medium", "the case's own setup decides its accuracy, which Wobbl ran as "
                                                    "given without a grid convergence study")])
    values = {"application": app, "end_time": reached, "steps": len(times),
              "residuals": {k: v[1] for k, v in residuals(log).items()}, "monitors": _monitors(case_dir)}
    res = residuals(log)
    worst_field = max(res, key=lambda k: res[k][1]) if res else ""
    head = (f"The case ran {app} to {fmt(reached)} in {len(times)} steps on {int(cells.group(1)) if cells else 0:,} "
            f"cells" + (f", with the largest final residual {res[worst_field][1]:.2g} ({worst_field})." if res else
                        "."))
    metrics = [("application", app), ("steps", str(len(times))), ("end time", fmt(reached))] + \
        [(k, fmt(v)) for k, v in list(values["monitors"].items())[:4]]
    o = GenericOutcome("flow", str(src), "", "ok" if all(ch.status != "fail" for ch in checks) else "fails", head,
                       metrics, level, reasons, checks, notes + ["The case ran single core as it was set up."],
                       values, {}, [], solver=f"OpenFOAM {get('openfoam').version()} {app}",
                       t_total=time.time() - t0)
    try:
        _playback(c, out, app)
    except Exception as e:  # noqa: BLE001
        o.notes.append(f"No playback file: {e}")
    if quick:
        shutil.rmtree(case_dir, ignore_errors=True)
    return o.write(out)


def _monitors(case_dir: Path) -> dict:
    """The last value of every column every function object wrote."""
    out = {}
    pp = case_dir / "postProcessing"
    if not pp.exists():
        return out
    c = foam.Case.__new__(foam.Case)
    c.root = case_dir
    for d in sorted(p for p in pp.iterdir() if p.is_dir()):
        try:
            head, arr = foam.table(c, d.name)
        except Exception:  # noqa: BLE001
            continue
        if len(arr) and head:
            for i, col in enumerate(head[1:arr.shape[1]], start=1):
                if np.isfinite(arr[-1, i]):
                    out[f"{d.name} {col}"] = float(arr[-1, i])
    return out


def _playback(c, out: Path, app: str):
    """The last velocity field in the plane through the middle of the domain, across its thinnest extent."""
    from . import playback

    c.run("postProcess", "-func", "writeCellCentres", "-latestTime", check=False)
    last = foam.last_time(c)
    C, U = foam.read_field(last / "C"), foam.read_field(last / "U")
    if C.ndim != 2 or U.ndim != 2:
        raise ValueError("the velocity is uniform, so there is nothing to show")
    span = C.max(0) - C.min(0)
    k = int(np.argmin(span))
    a, b = [i for i in range(3) if i != k]
    mid = 0.5 * (C[:, k].max() + C[:, k].min())
    from scipy.spatial import cKDTree
    near, _ = cKDTree(C).query(C, k=2)
    sel = np.abs(C[:, k] - mid) <= 0.55 * near[:, 1] + 1e-12
    f = playback.grid_field("flow", C[sel, a], C[sel, b], {"u": U[sel, a], "v": U[sel, b]},
                            {"u": "m/s", "v": "m/s"}, nx=128, ny=max(16, int(128 * span[b] / max(span[a], 1e-12))),
                            unit="m", spacing=near[sel, 1],
                            plane={"origin": [mid if i == k else 0.0 for i in range(3)],
                                   "u": np.eye(3)[a].tolist(), "v": np.eye(3)[b].tolist()})
    playback.write(out, "case", f"OpenFOAM case run by {app}", "The final velocity in the plane through the middle "
                   "of the domain, across its thinnest extent, in the case's own units.", [f])
