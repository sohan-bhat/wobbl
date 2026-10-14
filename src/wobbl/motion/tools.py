"""Build motion studies from flags, spec files and MCP calls."""
from __future__ import annotations

import math
import uuid
from pathlib import Path

from .spec import SPHERE_CD, MotionStudy, SpecError, load_toml, quantity, read_spec
from .study import run_motion


def study_from(name: str = "", mass: str = "", density: str = "", diameter: str = "", cd: float | None = None,
               height: str = "", speed: str = "", angle: str = "", wind: str = "", question: str = "",
               spec: str = "") -> MotionStudy:
    """A study from a spec file or TOML text, or a round body from flags. With a spec, only the wind can be
    changed from the outside, since it changes from one day to the next."""
    if spec:
        rest = {"--mass": mass, "--density": density, "--diameter": diameter, "--height": height,
                "--speed": speed, "--angle": angle, "--cd": cd}
        given = [k for k, v in rest.items() if v not in ("", None)]
        if given:
            raise SpecError(f"{', '.join(given)} cannot be combined with a spec file",
                            "Change the value in the spec, and pass only --name, --wind, --question and --out with it.")
        st = read_spec(spec, question)
        if name:
            st.name = name
        if wind:
            st.wind = quantity(wind, "speed", "--wind")
        return st
    st = MotionStudy(name=name or "body", question=question)
    if not diameter:
        raise SpecError("give the body's diameter", "For example --diameter '9 cm'.")
    st.diameter = quantity(diameter, "length", "--diameter")
    if mass and density:
        raise SpecError("give --mass or --density, not both")
    if mass:
        st.mass = quantity(mass, "mass", "--mass")
    elif density:
        st.mass = quantity(density, "density", "--density") * math.pi / 6 * st.diameter ** 3
        st.notes.append(f"A solid sphere {st.diameter * 100:.3g} cm across, so {st.mass:.4g} kg.")
    else:
        raise SpecError("give the body's --mass, or its --density for a solid sphere")
    st.cd = SPHERE_CD if cd is None else float(cd)
    if cd is None:
        st.notes.append(f"No drag coefficient given, so {SPHERE_CD} for a sphere (override with --cd).")
    if height:
        st.height = quantity(height, "length", "--height")
    if speed:
        st.speed = quantity(speed, "speed", "--speed")
    if angle:
        st.angle = quantity(angle if any(c.isalpha() for c in angle) else f"{angle} deg", "angle", "--angle")
    elif st.speed:
        st.angle = 45.0
        st.notes.append("No launch angle given, so 45 deg above level (override with --angle).")
    if wind:
        st.wind = quantity(wind, "speed", "--wind")
    st.validate()
    return st


def footprint() -> dict:
    """What motion adds to an install. It runs on scipy, which Wobbl needs anyway, so it downloads nothing and
    leaves nothing to prune, and its own code is a few dozen kilobytes."""
    import scipy

    code = sum(p.stat().st_size for p in Path(__file__).parent.glob("*.py"))
    return {"download_bytes": 0, "code_bytes": code, "scipy": scipy.__version__,
            "line": f"motion runs on scipy {scipy.__version__}, which came with Wobbl, so it downloads nothing and "
                    f"its own code is {code / 1000:.0f} kB"}


def solve_motion_tool(spec: str = "", name: str = "", mass: str = "", density: str = "", diameter: str = "",
                      cd: float | None = None, height: str = "", speed: str = "", angle: str = "", wind: str = "",
                      question: str = "") -> dict:
    from ..solvers.locate import wobbl_home

    try:
        st = study_from(name, mass, density, diameter, cd, height, speed, angle, wind, question, spec)
        out = wobbl_home() / "results" / f"{st.name}-motion-{uuid.uuid4().hex[:6]}"
        o = run_motion(st, out)
        save_request(out, spec, name, mass, density, diameter, cd, height, speed, angle, wind, question)
    except Exception as e:  # noqa: BLE001
        hint = getattr(e, "hint", "")
        return {"error": str(e), **({"hint": hint} if hint else {})}
    return {"verdict": o.verdict, "headline": o.headline, "metrics": dict(o.metrics),
            "values": {k: v for k, v in o.values.items() if not k.startswith("_")}, "events": o.events,
            "confidence": o.confidence, "confidence_reasons": o.confidence_reasons,
            "checks": [{"name": c.name, "status": c.status, "summary": c.summary} for c in o.checks],
            "assumptions": o.assumptions, "fixes": o.fixes, "images": o.images, "report": o.report,
            "trajectory": str(out / "trajectory.csv"), "result_dir": o.workdir}


def save_request(out, spec: str = "", name: str = "", mass: str = "", density: str = "", diameter: str = "",
                 cd: float | None = None, height: str = "", speed: str = "", angle: str = "", wind: str = "",
                 question: str = ""):
    """request.json for the run: the flags as given, or the spec parsed with the folder its files are read from,
    so uncertainty and credibility can fly it again with other values."""
    from pathlib import Path

    from ..credibility.domains import save_request as save

    if spec:
        path = Path(spec) if "\n" not in spec else None
        d = load_toml(path.read_text() if path else spec)
        body = {"toml": d, "base": str((path.parent if path else Path.cwd()).resolve()), "name": name,
                "wind": wind, "question": question}
        save(out, "motion", body, "spec")
    else:
        flags = {"name": name, "mass": mass, "density": density, "diameter": diameter, "cd": cd, "height": height,
                 "speed": speed, "angle": angle, "wind": wind, "question": question}
        save(out, "motion", {k: v for k, v in flags.items() if v not in ("", None)}, "flags")


__all__ = ["study_from", "solve_motion_tool", "footprint", "save_request"]
