"""How each domain runs a request again, for domains.rerun, until the shared simulate registry runs them all.

Each runner takes a Request (its domain, entry point and spec) and an empty folder, runs the study there without
images or fixes where the domain allows it, and returns the outcome as a dict. A spec is the plain JSON the domain
was run from, the same that domains.save_request wrote into request.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .domains import Request, legacy, register


def _part(spec: dict, base: Path | None) -> dict:
    s = dict(spec)
    p = s.get("part")
    if p and base is not None and not Path(p).is_absolute():
        s["part"] = str((base / p).resolve())
    return s


# --- strength and heat, whose result folders kept study.json before request.json existed ----------------------
@legacy("strength")
def _strength_legacy(d: Path, o: dict) -> Request | None:
    p = d / "study.json"
    if not p.exists():
        return None
    return Request("strength", _part(json.loads(p.read_text()), d), "study")


@legacy("thermal")
def _thermal_legacy(d: Path, o: dict) -> Request | None:
    p = d / "study.json"
    if not p.exists():
        return None
    return Request("thermal", _part(json.loads(p.read_text()), d), "study")


@register("strength")
def _strength(req: Request, out: Path) -> dict:
    from ..structural.pipeline import analyze_strength
    from ..structural.spec import StructuralStudy

    o = analyze_strength(StructuralStudy.from_dict(req.spec), out, render=False, fixes=False)
    return o.to_dict()


@register("thermal")
def _thermal(req: Request, out: Path) -> dict:
    from ..thermal.study import Film, ThermalStudy, run_thermal

    d = dict(req.spec)
    d["films"] = [Film(**f) if isinstance(f, dict) else f for f in d.get("films", [])]
    d["fixed"] = [tuple(x) for x in d.get("fixed", [])]
    st = ThermalStudy(**{k: v for k, v in d.items() if k in ThermalStudy.__dataclass_fields__})
    o = run_thermal(st, out)
    return o.to_dict() if hasattr(o, "to_dict") else o


@register("circuit")
def _circuit(req: Request, out: Path) -> dict:
    from ..circuits.run import run_circuit

    s = req.spec
    return run_circuit(s["template"], dict(s.get("params", {})), out, s.get("question", "")).to_dict()


@register("motion")
def _motion(req: Request, out: Path) -> dict:
    from ..motion.spec import quantity, read_spec
    from ..motion.study import run_motion
    from ..motion.tools import study_from

    s = req.spec
    if "toml" in s:
        st = read_spec(s["toml"], s.get("question", ""), s.get("base"))
        if s.get("name"):
            st.name = s["name"]
        if s.get("wind"):
            st.wind = quantity(s["wind"], "speed", "--wind")
    else:
        st = study_from(**{k: v for k, v in s.items() if k in ("name", "mass", "density", "diameter", "cd", "height",
                                                              "speed", "angle", "wind", "question")})
    return run_motion(st, out).to_dict()


@register("flow")
def _flow(req: Request, out: Path) -> dict:
    from ..flow.tools import build

    params = dict(req.spec)
    if req.kind == "pipe":
        params["cfd"] = "off"      # the CFD friction check cross checks a correlation, so reruns skip it
    study, run = build(req.kind, params)
    return run(study, out).to_dict()


@register("battery")
def _battery(req: Request, out: Path) -> dict:
    from ..chem.runaway import cell_from, run_runaway
    from ..units import parse_quantity

    s = req.spec
    c = cell_from(s.get("cell", "18650"), s.get("film", ""), float(s.get("emissivity", 0.8)))
    ta = parse_quantity(str(s.get("ambient", "25 C")), "temperature").value + 273.15
    return run_runaway(c, ta, out, s.get("question", ""), render=False).to_dict()


@register("motor")
def _motor(req: Request, out: Path) -> dict:
    from ..chem.motor import run_motor
    from ..chem.spec import read_spec

    ms = read_spec(req.spec["toml"], req.spec.get("question", ""))
    return run_motor(ms, out, render=False).to_dict()
