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
