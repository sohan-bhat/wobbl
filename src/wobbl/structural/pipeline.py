"""One call from a strength question to a verified, rendered, reported answer."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from ..report.render import convergence_plot, render_structural
from ..verify.checks import rel_up
from .outcome import Outcome, evaluate
from .report import _WRITING, write_outputs
from .run import Progress, run_structural
from .spec import StructuralStudy


def analyze_strength(study: StructuralStudy, out_dir: str | Path, progress: Progress | None = None,
                     render: bool = True, fixes: bool = True, pre: tuple | None = None, wait: bool = True) -> Outcome:
    """Solve, check and report a strength study. The pictures are drawn in the background while the fixes, the
    VTU and the 3D view are worked out. wait=False returns as soon as the answer is written, with the drawing
    left in o.pictures, which rewrites the report with the pictures when it finishes."""
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
    pictures = None
    if render:
        if progress:
            progress("render", "drawing heatmaps alongside the rest")
        pictures = Pictures(o, out, run.mesh.X, run.mesh.tets, run.mesh.tris, run.measure, run.result.U,
                            rewrite=not wait)
        pictures.start()
    if fixes:
        from ..fix.suggest import suggest_fixes
        o.fixes = suggest_fixes(run, o)
    _save_vtu(run, out / "result.vtu")
    _save_scene(run, o, out)
    if pictures is not None:
        if wait:
            pictures.wait()
        else:
            o.pictures = pictures
    return write_outputs(o, out)


def draw_pictures(o: Outcome, out: Path, X, tets, tris, measure, U) -> dict:
    """The heatmaps and the convergence chart of a strength result, as name -> path."""
    img = Path(out) / "images"
    images = render_structural(img, X, tets, tris, measure, U, o.governing_xyz, o.governing_MPa,
                               "von Mises" if "Mises" in o.measure else "max principal", o.allowable_MPa, o.target_sf,
                               o.hotspots)
    images["convergence"] = convergence_plot(img / "convergence.png", o.levels, o.hotspots)
    return images


def pictures_from_fields(o: Outcome, out: Path) -> dict | None:
    """Draw the pictures of a finished run from its saved fields, for a run made without them; None when the fields
    were not kept."""
    import numpy as np

    path = Path(out) / "work" / "fields.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        if "tets" not in f:
            return None
        return draw_pictures(o, Path(out), f["X"], f["tets"], f["tris"], f["measure"], f["U"])


class Pictures(threading.Thread):
    """Draws a result's pictures beside the rest of the run. rewrite=True writes the outputs again once they are
    done, for a caller that has already shown the answer."""

    def __init__(self, o: Outcome, out: Path, X, tets, tris, measure, U, rewrite: bool = False):
        super().__init__(name="wobbl-pictures")
        self.o, self.out, self.fields, self.rewrite = o, Path(out), (X, tets, tris, measure, U), rewrite
        self.error: BaseException | None = None

    def run(self):
        try:
            images = draw_pictures(self.o, self.out, *self.fields)
        except BaseException as e:  # noqa: BLE001  a picture is never a reason to lose a finished study
            self.error = e
            images = {"error": f"pictures failed: {str(e)[:200]}"}
        with _WRITING:   # never while the outcome is being saved
            self.o.images.update(images)
        if self.rewrite:
            write_outputs(self.o, self.out)

    def wait(self):
        self.join()
        if self.error is not None and os.environ.get("WOBBL_DEBUG"):
            raise self.error


def _save_scene(run, o, out: Path):
    """The run's interactive 3D view (wobbl.viz). A view that cannot be built never fails the run."""
    try:
        from ..viz.adapters.strength import write_scene
        write_scene(run, o, out)
    except Exception:  # noqa: BLE001
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise


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
