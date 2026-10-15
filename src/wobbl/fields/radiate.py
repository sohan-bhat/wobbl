"""Which solver an antenna question needs: the cheap one where it is valid, the 3D full wave solve where it is not.

Wobbl has three ways to answer an antenna question.

A straight wire dipole or a monopole on an infinite ground plane is a body of revolution, so the axisymmetric solve
in antenna.py is exact for it, in seconds, and the induced EMF theory checks it. Nothing 3D can add to that answer.

A patch sized from the transmission line model (Balanis, chapter 14) takes no solve at all. It is good to 2 to 4%
in resonance, which is enough to size a patch for a frequency and not enough to say whether a given patch lands
within a 2% tolerance, and it knows nothing about the feed, the ground plane's edge, S11 or the pattern.

Everything else is 3D: a loop, an inverted F or meander trace, a patch whose size or feed was given, a dipole over a
plate of finite size, a monopole on a small ground, any antenna that arrives as a STEP file or a KiCad board. Those
go to wave3d. The decision comes from the antenna's own dimensions and from what the question asks for, never from
a scenario's name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# what a question asks for that only a field solve over the real geometry answers
RESPONSE = re.compile(r"\bs11\b|return loss|\bvswr\b|bandwidth|impedance|\bmatch(?:ed|ing)?\b|\bgain\b|\bpattern\b|"
                      r"efficien|directiv|beamwidth|radiation|\blobes?\b|\btuned?\b|detun", re.I)
# a question that puts the antenna somewhere the symmetric models do not describe
SETTING = re.compile(r"finite ground|small ground|ground (?:plane|plate) of|on a (?:pcb|board|case|box|laptop|phone|"
                     r"car|roof|chassis)|near (?:a|the)\b|next to|\bbent\b|\btilted?\b|enclosure|housing|"
                     r"\bhuman\b|\bhand\b|\bbody\b", re.I)

SYMMETRIC = ("dipole", "monopole")
THREE_D = ("loop", "invf")


@dataclass
class Route:
    mode: str                 # axisymmetric, closed form or 3d
    reasons: list[str] = field(default_factory=list)

    @property
    def full_wave(self) -> bool:
        return self.mode == "3d"


def route(kind: str, params: dict, question: str = "") -> Route:
    """The solver for an antenna described by its dimensions."""
    q = question or ""
    asks = RESPONSE.search(q)
    where = SETTING.search(q)
    if kind in THREE_D:
        return Route("3d", [f"a {_name(kind)} has no rotational symmetry and no closed form that holds its feed and "
                            "its pattern, so it is solved in 3D"])
    if kind == "monopole" and params.get("ground"):
        return Route("3d", ["the ground plate has a finite size, which the axisymmetric model (an infinite plane) "
                            "cannot hold, so it is solved in 3D"])
    if kind in SYMMETRIC:
        if where:
            return Route("3d", [f"the question puts the antenna in a setting ('{where.group(0)}') that breaks its "
                                "rotational symmetry, so it is solved in 3D"])
        return Route("axisymmetric", ["a straight wire is a body of revolution, so the axisymmetric full wave solve "
                                      "is exact for it"])
    if kind == "patch":
        given = [k for k in ("width", "length", "feed", "ground") if params.get(k)]
        if given:
            return Route("3d", [f"the patch's {', '.join(given)} {'was' if len(given) == 1 else 'were'} given, so it "
                                "is a design to check, and the transmission line model is good to 2 to 4% in "
                                "resonance, no better than the tolerance"])
        if asks or where:
            return Route("3d", [f"the question asks about {(asks or where).group(0)!r}, which the transmission line "
                                "model cannot answer, so it is solved in 3D"])
        return Route("closed form", ["only the patch's size is wanted, and the transmission line model gives it "
                                     "without a solve"])
    return Route("3d", ["solved in 3D"])


def _name(kind: str) -> str:
    return {"invf": "printed inverted F trace"}.get(kind, kind)


# --- the 3D study ---------------------------------------------------------------------------------------
def _budget(params: dict) -> float:
    from .params import value
    from .wavecost import DEFAULT_BUDGET

    v = params.get("budget")
    return DEFAULT_BUDGET if v is None else value(params, "budget", DEFAULT_BUDGET, "time")


def compare_with_models(built):
    """Checks of the solve against what a simple model says of the same antenna, with that model's stated error."""
    from ..units import fmt
    from ..verify.checks import Check

    def patch(found: dict) -> list:
        res, ref = found["resonance"], built.ref
        fr = ref["transmission_line_Hz"]
        if not res:
            return [Check("transmission line model", "info", "no resonance inside the sweep to compare with the "
                          f"transmission line model's {fmt(fr / 1e6)} MHz")]
        d = (res[0] - fr) / fr
        st = "pass" if abs(d) <= 0.04 else "warn"
        return [Check("transmission line model", st,
                      f"the field solve resonates at {fmt(res[0] / 1e6)} MHz and the transmission line model "
                      f"(Balanis chapter 14, good to 2 to 4% on thin substrates) gives {fmt(fr / 1e6)} MHz, "
                      f"{100 * d:+.1f}% apart",
                      {"solve_Hz": res[0], "model_Hz": fr, "relative": d})]

    def invf(found: dict) -> list:
        res = found["resonance"]
        if not res:
            return []
        return [Check("quarter wave estimate", "info",
                      f"the arm is {fmt(built.dims['arm'] * 1e3)} mm, a quarter of a wavelength in a medium of "
                      f"permittivity about {(2.99792458e8 / (4 * built.dims['arm'] * res[0])) ** -2:.2g} at the solved "
                      f"resonance of {fmt(res[0] / 1e6)} MHz")]

    return {"patch": patch, "invf": invf}.get(built.kind)


def run_radiate(kind: str, params: dict, out_dir, question: str = "", keep: bool = False,
                log=lambda stage, detail="": None, why: list | None = None, quality: str = ""):
    """An antenna from its dimensions, solved in 3D. The geometry is built from the dimensions (the defaults are
    first guesses from the frequency), the study is planned against its time budget, and the report holds the
    solver's own checks and a comparison with the simple model of the same antenna."""
    import shutil
    from pathlib import Path

    from . import antenna3d, wave3d
    from .params import text

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = dict(params)
    q = quality or text(p, "quality", "normal")
    log("mesh", f"building the {_name(kind)}")
    shutil.rmtree(out / "geometry", ignore_errors=True)
    built = antenna3d.build(kind, p, out / "geometry")
    notes = [f"Solved in 3D. {r[:1].upper() + r[1:]}." for r in (why or [])]
    notes += built.notes
    try:
        o = wave3d.run(built.scene, built.setup, out, question, q, log=log, title=built.title, keep=keep,
                       budget=_budget(p), compare=compare_with_models(built), notes=notes)
    finally:
        if not keep:
            shutil.rmtree(out / "geometry", ignore_errors=True)
    return o
