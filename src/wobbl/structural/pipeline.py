"""One call from a strength question to a verified, rendered, reported answer."""
from __future__ import annotations

import json
from pathlib import Path

from ..report.render import convergence_plot, render_structural
from .outcome import Outcome, evaluate
from .report import write_outputs
from .run import Progress, run_structural
from .spec import StructuralStudy


def analyze_strength(study: StructuralStudy, out_dir: str | Path, progress: Progress | None = None,
                     render: bool = True, fixes: bool = True, pre: tuple | None = None) -> Outcome:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = run_structural(study, out / "work", progress, pre=pre)
    (out / "study.json").write_text(json.dumps(study.to_dict(), indent=1))
    (out / "geometry.json").write_text(json.dumps(run.prep.to_dict(), indent=1))
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
        from ..report.render import _grid

        g = _grid(run.mesh.X, run.mesh.tets)
        g.point_data["displacement_mm"] = run.result.U
        g.point_data["von_mises_MPa"] = run.result.von_mises()
        g.point_data["failure_measure_MPa"] = run.measure
        g.point_data["max_principal_MPa"] = run.result.principal()[:, 2]
        g.point_data["excluded_zone"] = run.excluded.astype(float)
        g.save(str(path))
    except Exception:  # noqa: BLE001  the VTU is a convenience, never a reason to fail
        pass
