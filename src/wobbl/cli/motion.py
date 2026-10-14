"""wobbl motion: falling, thrown and rocket flights."""
from __future__ import annotations

import json

import click

from . import ui
from .thermal import show_generic


@click.command()
@click.argument("spec", required=False, default="")
@click.option("--name", default="", help="Name for the report and the output folder, the spec's name if left out.")
@click.option("--mass", default="", help="Mass of the body, e.g. '1 kg' or '145 g'.")
@click.option("--density", default="", help="Density of a solid round body, e.g. '2700 kg/m3', instead of --mass.")
@click.option("--diameter", "-d", default="", help="Diameter that meets the air, e.g. '9 cm'.")
@click.option("--cd", type=float, default=None, help="Drag coefficient on the frontal area (default 0.47, a sphere).")
@click.option("--height", default="", help="Starting height above the ground, e.g. '300 m'.")
@click.option("--speed", default="", help="Launch speed, e.g. '30 m/s' or '60 mph'.")
@click.option("--angle", default="", help="Launch angle above level, e.g. '35 deg' (default 45 when thrown).")
@click.option("--wind", default="", help="Wind blowing back toward the start, e.g. '5 m/s' (negative from behind).")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def motion(spec, name, mass, density, diameter, cd, height, speed, angle, wind, out, question, as_json):
    """Drops, throws and rocket flights under gravity, air drag, wind and thrust.

    \b
    wobbl motion --name rock --diameter "9 cm" --density "2700 kg/m3" --cd 0.6 --height "300 m"
    wobbl motion --name ball --mass "145 g" --diameter "7.4 cm" --speed "40 m/s" --angle 35
    wobbl motion rocket.toml
    """
    from ..motion.study import run_motion
    from ..motion.tools import save_request, study_from
    from .main import _out_dir

    try:
        st = study_from(name, mass, density, diameter, cd, height, speed, angle, wind, question, spec)
        out_dir = _out_dir(spec or st.name, out)
        with ui.progress(not as_json) as cb:
            o = run_motion(st, out_dir, cb)
        save_request(out_dir, spec, name, mass, density, diameter, cd, height, speed, angle, wind, question)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
        return
    show_generic(o)
