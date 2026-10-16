"""What a run hands the drawings: its vectors as an instrumented run names them, and the one live reading that
answers the question (the part closest to a rating), for draw.write_circuit's bench and schematic and for the
template runs' playback."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .netlist import Circuit
from .solve import Wave
from .stress import Stress, _Pins


def wave_data(c: Circuit, w: Wave) -> dict:
    """The run as vectors: time, v(node) and i(ammeter) for every pin of Circuit.spice(instrument=True)."""
    d: dict = {"time": w.t} if w.kind == "tran" else {}
    for n, v in w.v.items():
        d[f"v({n})"] = v
    for e, k, vn in c.probes():
        cur = w.i.get(e.name) or w.i.get("X" + e.name)
        if cur and k < len(cur) and cur[k] is not None:
            d[f"i({vn})"] = cur[k]
    return d


def key_reading(c: Circuit, w: Wave, stress: list[Stress]) -> dict | None:
    """The reading to show live: the most used rating of the board, in the units a meter would show it, with
    25 samples of it over the run."""
    best = max(((s, s.worst) for s in stress if s.worst is not None and s.worst.use is not None),
               key=lambda x: x[1].use, default=None)
    if best is None:
        return None
    s, it = best
    e = c.element(s.ref) or next((x for x in c.elements if x.ref == s.ref), None)
    unit, scale = it.unit, 1.0
    if unit in ("A", "W", "V") and 0 < abs(it.value) < 1:
        unit, scale = "m" + unit, 1000.0
    series = None
    if e is not None and w.kind == "tran":
        try:
            P = _Pins(e, w)
            if it.unit == "W":
                x = P.power()
            elif it.unit == "A":
                x = np.abs(P.i(e.pins[0]))
            elif it.unit == "C":
                th = (e.part.get("ratings") or {}).get("theta_ja") or 0
                x = (it.value - th * s.p) + th * P.power()
            else:
                x = np.abs(P.v(e.pins[0]) - P.v(e.pins[1])) if len(e.pins) > 1 else None
            if x is not None:
                t = np.linspace(w.t[0], w.t[-1], 25)
                series = [round(float(v) * (1 if it.unit == "C" else scale), 4) for v in np.interp(t, w.t, x)]
        except (KeyError, IndexError, TypeError):
            series = None
    return {"label": f"{s.ref} {it.what}".upper(), "value": it.value * (1 if it.unit == "C" else scale),
            "unit": unit, "series": series, "limit": (it.limit or 0) * (1 if it.unit == "C" else scale) or None,
            "note": it.note or f"{100 * it.use:.0f}% of its rating"}


def draw(out: Path, c: Circuit, w: Wave, stress: list[Stress], verdict: str, **sweep) -> dict[str, str]:
    """setup.svg and schematic.svg from draw.write_circuit; a drawing that cannot be made is left out. sweep
    passes the question and ambient on to the 3D board's input sweep."""
    board(out, c, w, stress, **sweep)
    try:
        from .draw import write_circuit
        return write_circuit(Path(out) / "images", c, wave_data(c, w), verdict, key_reading(c, w, stress))
    except Exception:  # noqa: BLE001  pictures never cost the result
        return {}


def board(out: Path, c: Circuit, w: Wave, stress: list[Stress], **sweep) -> None:
    """board.json, the board plan the interactive 3D view (wobbl.viz) is built from, with the circuit solved
    across the input its question points at (sweep.py). It never costs the result."""
    try:
        from ..viz.adapters.circuits import write_board
        write_board(out, c, w, stress, **sweep)
    except Exception:  # noqa: BLE001
        import os
        if os.environ.get("WOBBL_DEBUG"):
            raise


def template_parts(netlist: str, analysis: str, tran, out: Path, ambient: float, verdict: str,
                   template: str = "", params: dict | None = None, question: str = "") -> dict:
    """A template's own netlist run instrumented, so its result carries every part against its ratings and a
    playback.json like any other circuit. The template's verdict and numbers are its own and stay as they are."""
    from .bind import bind
    from .netlist import parse_spice
    from .playback import write as write_playback
    from .solve import materialize, run, run_op
    from .stress import evaluate

    c = parse_spice(netlist)
    bind(c)
    work = Path(out) / "work"
    def solve(cc, tag="parts", longer=1):
        """The template's own run of a circuit; longer stretches a transient so a changed part settles too."""
        if analysis == "tran" and tran:
            step, stop = tran[0], tran[1] * longer
            start = max(0.0, stop - 20000 * step)        # a long switching run keeps only its settled end
            wv = run(materialize(cc, "tran"), [f"tran {step:.6e} {stop:.6e} {start:.6e} {step:.6e} uic"], work,
                     tag)[0]
            wv.window = wv.t >= wv.t[0] + 0.5 * (wv.t[-1] - wv.t[0])
            return wv
        return run_op(cc, work, tag)

    def build(p):
        from .templates import TEMPLATES
        cc = parse_spice(TEMPLATES[template](p).netlist)
        bind(cc)
        return cc

    w = solve(c)
    stress = evaluate(c, w, ambient)
    write_playback(out, c, w, stress, verdict)
    board(out, c, w, stress, question=question, ambient=ambient,
          template=(template, dict(params or {}), build, solve) if template else None)
    return {s.ref: s.to_dict() for s in stress}
