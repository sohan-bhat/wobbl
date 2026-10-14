"""wobbl submodel, kept as a hidden alias of wobbl simulate RESULT "refine the stress around the peak"."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui


def run(result: str, at=None, radius: float | None = None, size: float | None = None, out: str | None = None,
        progress=None) -> dict:
    """The part.submodel analysis of wobbl simulate on a finished strength result, returned as its outcome."""
    from ..simulate import route, run_plan

    xyz = ",".join(str(v) for v in at) if isinstance(at, (list, tuple)) else at
    options = {k: [str(v)] for k, v in (("at", xyz), ("radius", radius), ("size", size)) if v}
    plan = route(str(result), "", ["submodel"], "part")
    steps = run_plan(plan, options, Path(out) if out else Path(result) / "submodel", progress)
    o = steps[0].outcome
    return o if isinstance(o, dict) else o.to_dict()


@click.command(hidden=True)
@click.argument("result", type=click.Path(exists=True, file_okay=False))
@click.option("--at", "at", default="", help="Center of the local model as x,y,z in mm (default the global peak).")
@click.option("--radius", type=float, default=None, help="Radius of the local model in mm (default 2.5 feature sizes).")
@click.option("--size", type=float, default=None, help="Element size in the local model in mm.")
@click.option("--out", default=None, help="Folder for the local model (default RESULT/submodel).")
@click.option("--json", "as_json", is_flag=True)
def submodel(result, at, radius, size, out, as_json):
    """Refine only around the peak of a finished strength result, as wobbl simulate RESULT --physics submodel."""
    try:
        with ui.progress(not as_json) as cb:
            d = run(result, at or None, radius, size, out, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(d, indent=1, default=str))
        return
    ui.verdict_panel(d["verdict"].upper(), d["verdict"], d["headline"], [tuple(m) for m in d["metrics"]],
                     d["confidence"], d["confidence_reasons"])
    ui.checks_table(d["checks"])
    ui.console.print(f"\n  report {d.get('report', '')}\n")
