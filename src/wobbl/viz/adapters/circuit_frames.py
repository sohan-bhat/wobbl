"""What a circuit does across the input its question points at, frame by frame, for the 3D bench.

A run solves its circuit across one input the designer controls (circuits/sweep.py): the series resistor, a
load current, a part's value or the supply. Each frame is the whole circuit solved at one value of it, with
every part's share of its limit and the readouts that answer the question. A run from before the sweep existed
falls back on its transient's own samples ("k" in the board plan, powers in playback.json).
"""
from __future__ import annotations

import base64

import numpy as np


def decode(s: dict) -> np.ndarray:
    """A playback series back to numbers (value = scale * stored)."""
    a = np.frombuffer(base64.b64decode(s["data"]), dtype="<" + np.dtype(s["dtype"]).str[1:]).astype(float)
    return a * float(s.get("scale", 1.0)) + float(s.get("offset", 0.0))


def match(name: str, keys) -> str | None:
    """The playback ref for a board part's name, which may carry a SPICE letter in front (XU1 for U1)."""
    keys = list(keys)
    for k in (name, name[1:], name.split("_")[-1]):
        if k in keys:
            return k
    low = {k.lower().replace(" ", "_"): k for k in keys}
    return low.get(name.lower()) or next((low[k] for k in low if k.endswith(name.lower()[1:])), None)


def frames(b: dict, pb: dict | None) -> dict:
    """The timeline and per frame currents and heat of every part, wire and lead, from board.json (its k samples
    or its sweep) and playback.json. Frame count is 1 when the run is steady."""
    sw = b.get("sweep")
    pl = sw["plan"] if sw else b["plan"]
    parts, wires, ins = pl.get("parts", []), pl.get("wires", []), pl.get("instruments", [])
    if sw:
        t = np.asarray(sw["values"], float)
        inp = sw.get("input") or {}
        out = {"t": t, "unit": sw.get("unit", "V"), "label": sw.get("label") or f"{sw['source']} supply",
               "design": float(inp.get("design", t[-1])), "input": inp, "between": "step", "kind": "sweep",
               "parts": [np.asarray(x, float) for x in sw["parts"]],
               "wires": [np.asarray(x, float) for x in sw["wires"]],
               "leads": [[np.asarray(x, float) for x in i] for i in sw["leads"]],
               "readouts": sw.get("readouts") or []}
        if "heat" in sw:
            out["heat"] = [None if h is None else np.asarray(h, float) for h in sw["heat"]]
        else:
            pw = {k: np.asarray(v, float) for k, v in sw.get("power", {}).items()}
            out["heat"] = [_heat(p, pw.get(match(p["name"], pw) or "", None), out["parts"][j]) for j, p in
                           enumerate(parts)]
        return out
    peak = float(pl.get("peak") or 0) or max([abs(w.get("current") or 0) for w in wires] + [1e-12])
    live = any(x.get("k") for x in parts + wires)
    n = 25 if live else 1

    def ser(x):
        k, cur = x.get("k"), float(x.get("current") or 0.0)
        if k is None or n == 1:
            return np.full(n, cur)
        return np.sign(cur or 1) * np.asarray(k, float) * peak

    span = b.get("span") or pl.get("span")
    t = np.linspace(span[0], span[1], n) if span and n > 1 else np.zeros(n)
    out = {"t": t, "unit": "s", "label": "time", "between": "interpolated", "kind": "tran" if n > 1 else "op",
           "parts": [ser(p) for p in parts], "wires": [ser(w) for w in wires],
           "leads": [[ser(ld) for ld in i.get("leads", [])] for i in ins]}
    pw = {}
    if pb and n > 1 and pb.get("part_p"):
        tt = decode(pb["time"]) if isinstance(pb.get("time"), dict) else None
        for ref, s in pb["part_p"].items():
            x = decode(s)
            if tt is not None and len(tt) == len(x) and len(x) > 1:
                pw[ref] = np.interp(t, tt, x)
    out["heat"] = [_heat(p, pw.get(match(p["name"], pw) or "", None), out["parts"][j]) for j, p in enumerate(parts)]
    return out


def _heat(p: dict, power, cur) -> np.ndarray | None:
    """A part's share of its rating per frame: heat scales with current for an LED and with power otherwise."""
    h = p.get("heat")
    if h is None:
        return None
    n = len(cur)
    if p.get("kind") == "led":
        ref = abs(p.get("current") or 0)
        return np.full(n, h) if ref <= 0 else h * np.abs(cur) / ref
    ref = abs(p.get("power") or 0)
    if power is not None and ref > 0:
        return h * np.abs(power) / ref
    ref = abs(p.get("current") or 0)
    return np.full(n, h) if ref <= 0 else h * (np.asarray(cur) / ref) ** 2
