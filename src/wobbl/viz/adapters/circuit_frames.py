"""What a circuit does over its timeline, frame by frame, for the 3D bench.

A transient run already has its shape in the board plan: every wire and part carries 25 samples of its current
("k"), and playback.json holds each part's power over time. An operating point has no time, so the board is
swept instead. The main supply is turned up from zero to its value in 25 solved steps, which shows the LED
coming on and the current crossing a rating at the voltage where it really does.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

STEPS = 25


def sweep(c, pl: dict, work: Path) -> dict | None:
    """The board plan solved at 25 supply voltages from zero to the run's own, as absolute currents per frame for
    every part, wire and lead, and each part's power per frame. The sweep's own plan goes with it, as the layout
may differ a little when every step's currents are known."""
    from ...circuits.bind import dc_value
    from ...circuits.board import plan, source_role
    from ...circuits.solve import OPTIONS, materialize, ngspice

    m = materialize(c, "op")
    src = next((e for e in m.elements if e.letter == "v" and "0" in e.nodes and e.nodes[0] != e.nodes[1]
                and source_role(e)[0] == "supply"), None)
    v = dc_value(src) if src is not None else None
    if not v:
        return None
    vals = np.linspace(0.0, v, STEPS)
    data, _ = ngspice().run_raw(m.spice(instrument=True, extra=[OPTIONS]), work,
                                [f"dc {src.name} 0 {v:.6g} {v / (STEPS - 1):.6g}"], "sweep")
    d = {k.lower(): np.real(np.asarray(x, float)) for k, x in data[0].items()}
    n = len(next(iter(d.values())))
    if n != STEPS:
        vals = np.linspace(0.0, v, n)
    d["time"] = vals
    sp = plan(m, d)
    peak = sp.get("peak")
    out = {"source": src.name, "unit": "V", "values": [round(float(x), 5) for x in np.interp(
        np.linspace(0, n - 1, STEPS), np.arange(n), vals)]}

    def series(item, peak_):
        k, cur = item.get("k"), item.get("current") or 0.0
        if k is None or not peak_:
            return [cur] * STEPS
        return [round(float(np.sign(cur) * x * peak_), 9) for x in k]

    out["parts"] = [series(p, peak) for p in sp["parts"]]
    final = {p["name"]: p for p in pl["parts"]}
    for p in sp["parts"]:              # the run's own operating point stands for the sweep's last step
        f = final.get(p["name"]) or final.get(p["name"][1:]) or {}
        p.update({k: f.get(k) for k in ("current", "power", "heat")})
    inst = {i["name"]: i for i in pl.get("instruments", [])}
    for i in sp["instruments"]:
        i["readings"] = (inst.get(i["name"]) or i).get("readings", [])
    out["plan"] = sp
    out["wires"] = [series(w, peak) for w in sp["wires"]]
    out["leads"] = [[series(ld, peak) for ld in i.get("leads", [])] for i in sp["instruments"]]
    volts = {k[2:-1]: x for k, x in d.items() if k.startswith("v(") and k.endswith(")")}
    power = {}
    for e in m.elements:
        cur = [d.get(f"i({vn})") for ee, _, vn in m.probes() if ee is e]
        if not cur or any(x is None for x in cur):
            continue
        if len(e.nodes) == 2:
            cur = [cur[0], -cur[0]]
        vs = [np.zeros(n) if nd == "0" else volts.get(nd) for nd in e.nodes]
        if any(x is None for x in vs):
            continue
        p = sum(a * b for a, b in zip(vs, cur))
        power[e.name] = [round(float(x), 9) for x in np.interp(np.linspace(0, n - 1, STEPS), np.arange(n), p)]
    out["power"] = power
    return out


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
        n = len(sw["values"])
        out = {"t": np.asarray(sw["values"], float), "unit": "V", "label": f"{sw['source']} supply",
               "between": "interpolated", "kind": "sweep",
               "parts": [np.asarray(x, float) for x in sw["parts"]],
               "wires": [np.asarray(x, float) for x in sw["wires"]],
               "leads": [[np.asarray(x, float) for x in i] for i in sw["leads"]]}
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
