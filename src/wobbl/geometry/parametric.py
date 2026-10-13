"""Parametric parts: a Python file that builds the shape from named parameters.

A part file defines ``PARAMS`` (name -> default value in mm) and ``build(occ, **params)``,
where ``occ`` is gmsh's OpenCASCADE API (addBox, addCylinder, fuse, cut, fillet, ...).
Wobbl builds the STEP itself, so fixes like "thicken to 7.5 mm" or "fillet 3 mm" can
change a parameter and re-run the study:

    PARAMS = {"thickness": 5.0, "fillet": 0.5}

    def build(occ, thickness, fillet):
        a = occ.addBox(0, 0, 0, 80, thickness, 25)
        ...
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

THICK = re.compile(r"(thick|wall|^t$|^t_)", re.I)
RADIUS = re.compile(r"(fillet|radius|^r$|^r_)", re.I)


def is_parametric(path: str | Path) -> bool:
    return str(path).lower().endswith(".py")


def load_params(path: str | Path) -> dict:
    ns = _exec(path)
    return dict(ns.get("PARAMS", {}))


def _exec(path: str | Path) -> dict:
    src = Path(path).read_text()
    ns: dict = {"__file__": str(path), "__name__": "wobbl_part"}
    exec(compile(src, str(path), "exec"), ns)  # noqa: S102  the user's own part file, like running it locally
    if "build" not in ns:
        raise ValueError(f"{Path(path).name} needs a build(occ, **params) function")
    return ns


def _build(path: str, params: dict, out: str) -> str:
    import gmsh

    from ..validation.geometry import quiet_stdout
    from .cad import _deterministic

    ns = _exec(path)
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    _deterministic(gmsh)
    try:
        gmsh.model.add("part")
        ns["build"](gmsh.model.occ, **params)
        gmsh.model.occ.synchronize()
        if not gmsh.model.getEntities(3):
            raise ValueError("build() made no solid")
        with quiet_stdout():
            gmsh.write(out)
    finally:
        gmsh.finalize()
    return out


def build_step(path: str | Path, overrides: dict | None, out_dir: str | Path) -> tuple[str, dict]:
    """Build the part with defaults plus overrides; returns (STEP path, parameters used)."""
    from ..mesh.isolate import run_isolated

    params = load_params(path)
    for k, v in (overrides or {}).items():
        if k not in params:
            raise ValueError(f"{Path(path).name} has no parameter '{k}' (has {', '.join(params) or 'none'})")
        params[k] = type(params[k])(v) if not isinstance(params[k], bool) else str(v).lower() in ("1", "true")
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = out / f"{Path(path).stem}-{key}.step"
    if not step.exists():
        run_isolated(_build, str(Path(path).resolve()), params, str(step), timeout=180)
    (out / f"{step.stem}.json").write_text(json.dumps(params, indent=1))
    return str(step), params


def thickness_param(params: dict) -> str | None:
    return next((k for k in params if THICK.search(k)), None)


def radius_param(params: dict) -> str | None:
    return next((k for k in params if RADIUS.search(k)), None)
