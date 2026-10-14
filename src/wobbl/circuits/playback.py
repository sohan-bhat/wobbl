"""playback.json for a circuit run: the netlist as a graph and what every part did, for the site to animate.

    version, kind "circuit", analysis ("op" or "tran"), verdict, units
    nets       [{id, name, ground}]                       every net, with the name the user gave it
    parts      [{ref, kind, part, value, pins: [{name, net}], status, use, worst, tj, margins: [...]}]
               margins are each rating: {what, value, limit, unit, use, status, note}
    frames     how many samples follow; time is their time in seconds ([0] for an operating point)
    net_v      {net: series}                               node voltages
    part_v     {ref: series}, part_i {ref: series}, part_p {ref: series}
               each part's main voltage and current (across and through a two pin part, collector or drain
               for a transistor, output for a regulator, op amp or timer) and its power
    pin_i      {ref: {pin: series}}                        every pin's current for parts with more than two pins

Series use the shared playback encoding (wobbl.life.playback): base64 little endian integers with a scale,
value = scale * stored. Current into a pin is positive. The run is resampled to at most 240 frames, evenly in
time, so a viewer can play it at any speed; the file stays small and is the same for the same run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..life.playback import encode, quantize, write as _write
from .netlist import Circuit
from .solve import Wave
from .stress import Stress, _Pins

FRAMES = 240
MAIN = {"npn": ("c", "e", "c"), "pnp": ("e", "c", "e"), "nmos": ("d", "s", "d"), "pmos": ("s", "d", "s"),
        "regulator": ("in", "out", "out"), "opamp": ("out", "vee", "out"), "comparator": ("out", "vee", "out"),
        "timer": ("out", "gnd", "out"), "charger": ("vin", "bat", "bat"), "mcu": (None, None, None)}


def _resample(w: Wave, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, w.t, x) if w.kind == "tran" else np.asarray(x[-1:], float)


def write(out: Path, c: Circuit, w: Wave, stress: list[Stress], verdict: str) -> str:
    t = np.linspace(w.t[0], w.t[-1], min(FRAMES, max(len(w.t), 2))) if w.kind == "tran" else np.zeros(1)
    by = {s.ref: s for s in stress}
    nets = [{"id": n, "name": c.label(n), "ground": n == "0"} for n in c.nodes()]
    parts, pv, pi, pp, pins = [], {}, {}, {}, {}
    for e in c.elements:
        if not e.nodes or e.kind in ("coupling", "xspice"):
            continue
        s = by.get(e.ref)
        d = {"ref": e.ref, "kind": e.kind or e.letter, "part": e.part.get("key", ""),
             "value": " ".join(t_ for t_ in e.rest if "=" not in t_)[:40] if e.letter in "rclvi" else "",
             "pins": [{"name": p, "net": n} for p, n in zip(e.pins, e.nodes)]}
        if s is not None:
            d.update({"status": s.status, "use": s.worst.use if s.worst else None,
                      "worst": s.worst.what if s.worst else None, "tj": s.tj,
                      "margins": [{"what": it.what, "value": it.value, "limit": it.limit, "unit": it.unit,
                                   "use": it.use, "status": it.status, "note": it.note} for it in s.items]})
        parts.append(d)
        try:
            P = _Pins(e, w)
        except (KeyError, IndexError):
            continue
        a, b, ip = MAIN.get(e.kind, (e.pins[0], e.pins[1] if len(e.pins) > 1 else None, e.pins[0]))
        if a is None:
            a, b, ip = e.pins[0], None, e.pins[0]
        try:
            v = P.v(a) - (P.v(b) if b else 0)
            i = P.i(ip) * (-1 if e.kind in ("regulator", "charger") else 1)
            p = P.power()
        except (KeyError, IndexError, TypeError):
            continue
        pv[e.ref] = quantize(_resample(w, v, t))
        pi[e.ref] = quantize(_resample(w, i, t))
        pp[e.ref] = quantize(_resample(w, p, t))
        if len(e.pins) > 2:
            pins[e.ref] = {pn: quantize(_resample(w, P.i(pn), t)) for pn in e.pins}
    doc = {"version": 1, "kind": "circuit", "analysis": w.kind, "verdict": verdict,
           "units": {"time": "s", "voltage": "V", "current": "A", "power": "W", "temperature": "C"},
           "nets": nets, "parts": parts, "frames": int(len(t)), "time": encode(t.astype("float32"), "float32"),
           "net_v": {n: quantize(_resample(w, w.volt(n), t)) for n in c.nodes() if n != "0"},
           "part_v": pv, "part_i": pi, "part_p": pp, "pin_i": pins}
    return _write(Path(out) / "playback.json", _round(doc))


def _round(x):
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x
