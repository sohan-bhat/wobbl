"""Entry points for buckling, vibration and fatigue on any part, with any supports, loads and duty cycle.

Each takes plain values, the same strings the command line and agents use, and returns a LifeOutcome:

    buckling(part, material, fixtures, loads)          the load factor at which the part buckles, knocked down
    vibration(part, material, fixtures, ...)           modes, resonance against rotors and tones, and the response
                                                       to a sine sweep of the base or of the loads, or a random input
    fatigue(source or part, ...)                       cycles before a crack starts, from a finished strength or
                                                       vibration result, from a part and loads (solved first), or
                                                       from stresses given directly, under one load, load blocks or
                                                       a measured load history

DOMAINS describes them for a router: each entry's name, function, what it answers and its required inputs.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..structural.spec import StructuralStudy, parse_fixture, parse_load


def _study(part, material, fixtures=(), loads=(), *, quality="normal", scale=1.0, body=None, params=None,
           mesh_size=None, question="", print_direction=None) -> StructuralStudy:
    p = Path(str(part)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"file not found: {part}")
    st = StructuralStudy(part=str(p.resolve()), material=material, quality=quality, question=question, scale=scale,
                         body=body, mesh_size=mesh_size)
    st.params = dict(params or {})
    st.print_direction = print_direction
    st.fixtures = [parse_fixture(x) for x in fixtures]
    st.loads = [parse_load(x) for x in loads]
    return st


def _out(out, part, what: str) -> Path:
    if out:
        return Path(out)
    return Path("wobbl-out") / f"{Path(str(part)).stem or what}-{what}"


def buckling(part, material: str, fixtures, loads, *, target_sf: float | None = None, shape: str = "auto",
             modes: int = 4, out=None, progress=None, keep: bool = False, **part_kw):
    """Buckling of any part under its loads. Fixtures and loads are written as for a strength study."""
    from .buckling import BucklingStudy, run_buckling

    if not loads:
        raise ValueError("buckling needs the loads on the part, for example '1000 N -z @ z=max'")
    st = BucklingStudy(_study(part, material, fixtures, loads, **part_kw), target_sf, shape, modes, keep)
    return run_buckling(st, _out(out, part, "buckling"), progress)


def vibration(part, material: str, fixtures, *, loads=(), masses=(), rpm=(), blades: int = 2, frequencies=(),
              margin: float = 0.2, sine: str = "", psd: str = "", axis: str = "z", duration: str = "",
              damping: float | None = None, modes: int = 15, target_sf: float | None = None, out=None,
              progress=None, keep: bool = False, **part_kw):
    """Natural frequencies of any part as it is held, judged against what shakes it. A sine of '2 g z 10-500 Hz'
    shakes the base, and a band alone, '10-500 Hz', sweeps the loads as harmonic forces."""
    from .vibration import run_vibration
    from .vibspec import VibrationStudy

    st = VibrationStudy(_study(part, material, fixtures, loads, **part_kw), list(masses), modes, list(rpm), blades,
                        list(frequencies), margin, sine, psd, axis, duration, damping, target_sf, keep)
    st.excitations()
    return run_vibration(st, _out(out, part, "vibration"), progress)


def fatigue(source: str = "", *, part=None, material: str = "", fixtures=(), loads=(), amplitude: str = "",
            mean: str = "", kt: float | None = None, radius: str = "", ratio: float = 0.0, cycles: str = "1e7",
            blocks=(), history: str = "", repeats: float = 1.0, duration: str = "", surface: str = "auto",
            size: str = "", loading: str = "auto", reliability: float = 0.9, mean_stress: str = "goodman",
            target_sf: float | None = None, question: str = "", out=None, progress=None, **part_kw):
    """Fatigue life. Give a finished strength or vibration result as source, or a part with its fixtures and loads,
    which runs the strength study first, or a material with stresses. The load cycles at ratio R, through blocks
    like '100% x 2e4', or through a load history (a file or numbers) counted by rainflow."""
    from .fatigue_study import FatigueStudy, run_fatigue

    out_dir = _out(out, part or source or "fatigue", "fatigue")
    if part is not None:
        if source:
            raise ValueError("give either a result folder or a part to solve, not both")
        if not fixtures or not loads:
            raise ValueError("fatigue on a part needs its fixtures and loads, which the strength study solves first")
        from ..structural.pipeline import analyze_strength

        t0 = time.time()
        st = _study(part, material, fixtures, loads, question=question, **part_kw)
        if progress:
            progress("fatigue", "solving the part under its loads first")
        analyze_strength(st, out_dir / "strength", progress, render=False, fixes=False)
        source, material = str(out_dir / "strength"), ""
        if progress:
            progress("fatigue", f"strength study done in {time.time() - t0:.0f} s")
    fs = FatigueStudy(str(source), material, amplitude, mean, kt, radius, ratio, str(cycles), list(blocks), repeats,
                      duration, surface, size, loading, reliability, mean_stress, target_sf, question, history)
    return run_fatigue(fs, out_dir, progress)


DOMAINS = [
    {"name": "buckling", "function": buckling,
     "answers": "whether a part buckles under its loads, the load factor and the design factor after imperfections",
     "needs": ["part", "material", "fixtures", "loads"]},
    {"name": "vibration", "function": vibration,
     "answers": "natural frequencies against motor, propeller or other speeds, and the stress of a sine sweep of the "
                "base or of the loads, or of random vibration",
     "needs": ["part", "material", "fixtures"]},
    {"name": "fatigue", "function": fatigue,
     "answers": "how many load cycles a part lasts before a crack starts, under one load, blocks or a load history",
     "needs": ["a strength or vibration result, a part with fixtures and loads, or a material with stresses"]},
]
