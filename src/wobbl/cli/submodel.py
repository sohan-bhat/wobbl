"""wobbl submodel: a fine local model of the peak of any finished result whose domain registered one."""
from __future__ import annotations

import json

import click

from . import ui


def _xyz(text: str):
    try:
        v = [float(x) for x in text.replace(" ", "").split(",")]
    except ValueError:
        v = []
    if len(v) != 3:
        raise click.BadParameter("give a point as x,y,z in mm, for example 10,-4,2.5")
    return v


@click.command()
@click.argument("result", type=click.Path(exists=True, file_okay=False))
@click.option("--at", "at", default="", help="Center of the local model as x,y,z in mm (default the global peak).")
@click.option("--radius", type=float, default=None, help="Radius of the local model in mm (default 2.5 feature sizes).")
@click.option("--size", type=float, default=None, help="Element size in the local model in mm.")
@click.option("--out", default=None, help="Folder for the local model (default RESULT/submodel).")
@click.option("--json", "as_json", is_flag=True)
def submodel(result, at, radius, size, out, as_json):
    """Refine only around the peak of a finished result.

    The result's own domain does the refining. For a strength result the part is cut down to a ball around the
    peak, meshed finely, and held on its cut where the global solution moved. The peak comes back with two checks
    on the cut.

    \b
    wobbl submodel result
    wobbl submodel result --at 12,0,4 --radius 20
    """
    from ..jobs.workload import refine

    point = _xyz(at) if at else None
    try:
        with ui.progress(not as_json) as cb:
            r = refine(result, point, radius, size, out, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(r, indent=1))
        return
    word, key = ("TRUSTED", "ok") if r["trusted"] else ("CHECK THE CUT", "marginal")
    sign = "+" if r["change_pct"] >= 0 else ""
    ui.verdict_panel(word, key, f"local peak {r['submodel_peak_MPa']:.4g} MPa against {r['global_peak_MPa']:.4g} MPa "
                     f"in the global model ({sign}{r['change_pct']:.3g}%)",
                     [("local peak", f"{r['submodel_peak_MPa']:.4g} MPa"), ("radius", f"{r['radius_mm']:.3g} mm"),
                      ("element", f"{r['element_size_mm']:.3g} mm"), ("nodes", f"{r['nodes']:,}"),
                      ("time", f"{r['time_s']:.0f} s")],
                     "high" if r["trusted"] else "low",
                     [f"{r['measure']} at ({', '.join(f'{v:.4g}' for v in r['where_mm'])}) mm",
                      f"feature size from {r['feature']}"])
    ui.checks_table(r["checks"])
    if r["notes"]:
        ui.section("notes")
        ui.bullets(r["notes"])
    ui.console.print(f"\n  local model in {r['folder']}\n")
