"""wobbl cast, wobbl mold and wobbl warp: can the part be made the way it is drawn?"""
from __future__ import annotations

import json

import click

from . import ui
from .thermal import show_generic


def _finish(o, as_json):
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


def _params(params) -> dict:
    return dict(p.split("=", 1) for p in params if "=" in p)


@click.command()
@click.argument("part")
@click.option("--alloy", "-m", default="a356", show_default=True, help="Casting alloy: a356 or pure-al.")
@click.option("--riser", "risers", multiple=True,
              help="A blind cylindrical riser, e.g. 'd=40 h=60 at 50,0,30' (mm, the center of its foot on the "
                   "casting; axis=x or y to turn it). Repeatable. Without one Wobbl places risers over the hot "
                   "spots; 'none' freezes the bare casting.")
@click.option("--mesh-size", type=float, default=None, help="Metal element size in mm.")
@click.option("--chill", "chills", multiple=True,
              help="A chill block against the casting, e.g. 'steel 30x30x15 at 50,0,-10' (steel or copper, mm). "
                   "Repeatable.")
@click.option("--mold", default="sand", show_default=True, help="Mold material.")
@click.option("--pour", default="", help="Pouring temperature, e.g. '720 C' (default the alloy's usual).")
@click.option("--mold-temp", default="", help="Starting mold temperature (default 25 C).")
@click.option("--param", "params", multiple=True, help="Parameter of a .py part, e.g. thickness=12.")
@click.option("--quality", type=click.Choice(["fast", "normal"]), default="normal")
@click.option("--keep", is_flag=True, help="Keep the meshes and scratch files in work/.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def cast(part, alloy, risers, mesh_size, chills, mold, pour, mold_temp, params, quality, keep, out, question,
         as_json):
    """Casting: where the metal freezes last, whether risers feed it, and the shrinkage porosity left behind.

    \b
    wobbl cast bracket.step -m a356
    wobbl cast bracket.step -m a356 --riser "d=40 h=60 at 60,30,20" --chill "steel 30x30x10 at 10,30,-5"

    The part is surrounded by sand, filled at once with metal at the pouring
    temperature and frozen with an enthalpy solver. With no riser given, Wobbl
    places risers over the hot spots by the modulus rule and checks them.
    """
    from ..process.casting import run_casting
    from ..process.tools import casting_study
    from .main import _out_dir

    try:
        st = casting_study(part, alloy, risers, chills, mold, pour, mold_temp, question, quality, _params(params),
                           mesh_size)
        out_dir = _out_dir(part, out)
        with ui.progress(not as_json) as cb:
            o = run_casting(st, out_dir, cb, keep=keep)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    _finish(o, as_json)
