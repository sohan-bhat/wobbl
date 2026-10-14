"""One call from a strength question to a verified, rendered, reported answer."""
from __future__ import annotations

import json
from pathlib import Path

from ..report.render import convergence_plot, render_structural
from ..verify.checks import rel_up
from .outcome import Outcome, evaluate
from .report import write_outputs
from .run import Progress, run_structural
from .spec import StructuralStudy


def analyze_strength(study: StructuralStudy, out_dir: str | Path, progress: Progress | None = None,
                     render: bool = True, fixes: bool = True, pre: tuple | None = None) -> Outcome:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = run_structural(study, out / "work", progress, pre=pre)
    sd = study.to_dict()
    sd["part"] = rel_up(sd["part"], out)
    gd = run.prep.to_dict()
    gd["source"], gd["path"] = rel_up(gd["source"], out), rel_up(gd["path"], out)
    (out / "study.json").write_text(json.dumps(sd, indent=1))
    (out / "geometry.json").write_text(json.dumps(gd, indent=1))
    o = evaluate(run)
    o.workdir = str(out)
    if fixes:
        from ..fix.suggest import suggest_fixes
        o.fixes = suggest_fixes(run, o)
    if render:
        if progress:
            progress("render", "drawing heatmaps")
        img = out / "images"
        o.images = render_structural(img, run.mesh.X, run.mesh.tets, run.mesh.tris, run.measure, run.result.U,
                                     o.governing_xyz, o.governing_MPa, "von Mises" if "Mises" in o.measure
                                     else "max principal", o.allowable_MPa, o.target_sf, o.hotspots)
        o.images["convergence"] = convergence_plot(img / "convergence.png", o.levels, o.hotspots)
    _save_vtu(run, out / "result.vtu")
    return write_outputs(o, out)


def _save_vtu(run, path: Path):
    """Full field results for ParaView or any VTK viewer."""
    try:
        from ..report.vtu import write_vtu

        res = run.result
        write_vtu(path, run.mesh.X, run.mesh.tets, {
            "displacement_mm": res.U,
            # xx, yy, zz, xy, yz, zx: VTK's order for a symmetric tensor, so ParaView reads it as one
            "stress_MPa": res.S,
            "von_mises_MPa": res.von_mises(),
            "failure_measure_MPa": run.measure,
            "max_principal_MPa": res.principal()[:, 2],
            "excluded_zone": run.excluded.astype(float)})
    except Exception:  # noqa: BLE001  the VTU is a convenience, never a reason to fail
        pass
