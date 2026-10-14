"""The checks a general circuit run makes on itself before Wobbl stands behind it.

KCL at every node from the pin currents, the power account (what the supplies give equals what the parts
dissipate plus what the capacitors and inductors store, with the stored energy taken from 1/2 C V^2 and
1/2 L I^2 rather than from the same integral), convergence (half the time step for a transient, a hundred
times tighter tolerances for an operating point) and where every model came from.
"""
from __future__ import annotations

import numpy as np

from ..units import fmt
from ..verify.checks import Check
from .netlist import Circuit
from .solve import Wave
from .stress import Stress, _Pins


def kcl(c: Circuit, w: Wave) -> Check:
    """Currents into each node, from every pin on it, must sum to zero."""
    worst, where, scale = 0.0, "", 0.0
    sums: dict[str, np.ndarray] = {}
    mags: dict[str, float] = {}
    for e in c.elements:
        cur = w.i.get(e.name)
        if not cur:
            continue
        for n, x in zip(e.nodes, cur):
            if x is None or n == "0":
                continue
            sums[n] = sums.get(n, 0) + x
            mags[n] = max(mags.get(n, 0.0), float(np.max(np.abs(x))))
    for n, s in sums.items():
        r = float(np.max(np.abs(s)))
        scale = max(scale, mags[n])
        if r > worst:
            worst, where = r, n
    tol = max(1e-8, 1e-6 * scale)
    st = "pass" if worst <= tol else "warn" if worst <= 100 * tol else "fail"
    return Check("KCL", st, f"currents into every node sum to zero within {fmt(worst)} A (largest at "
                            f"{c.label(where) or 'none'}, against {fmt(scale)} A flowing)",
                 {"residual_A": worst, "node": where})


def power_balance(c: Circuit, w: Wave, stress: list[Stress]) -> Check:
    """Supplied power equals dissipated plus stored. For a transient the energies over the whole run are
    compared, with the stored energy from the capacitor voltages and inductor currents at the ends."""
    src_kinds = ("supply", "signal", "battery", "isource", "source")
    if w.kind == "op":
        given = sum(s.p for s in stress if s.kind in src_kinds)
        used = sum(s.p for s in stress if s.kind not in src_kinds)
        rel = abs(given - used) / max(abs(given), 1e-12)
        st = "pass" if rel < 1e-3 or abs(given - used) < 1e-9 else "warn" if rel < 0.01 else "fail"
        return Check("power balance", st, f"supplies give {fmt(given)} W and the parts dissipate {fmt(used)} W "
                                          f"({100 * rel:.2g}% apart)", {"supplied_W": given, "dissipated_W": used})
    t = w.t
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    e_src = e_diss = e_store = e_store_int = 0.0
    for e in c.elements:
        if not w.i.get(e.name):
            continue
        p = _Pins(e, w).power()
        energy = float(trap(p, t))
        if e.kind in src_kinds:
            e_src -= energy
        elif e.letter in ("c", "l") and e.value:
            e_store_int += energy
            if e.letter == "c":
                v = w.volt(e.nodes[0]) - w.volt(e.nodes[1])
                e_store += 0.5 * e.value * (v[-1] ** 2 - v[0] ** 2)
            else:
                i = w.i[e.name][0]
                e_store += 0.5 * e.value * (i[-1] ** 2 - i[0] ** 2)
        else:
            e_diss += energy
    rel = abs(e_src - e_diss - e_store) / max(abs(e_src), 1e-15)
    st = "pass" if rel < 0.005 else "warn" if rel < 0.03 else "fail"
    return Check("power balance", st, f"over the run the supplies gave {fmt(e_src)} J, the parts dissipated "
                                      f"{fmt(e_diss)} J and the capacitors and inductors stored {fmt(e_store)} J "
                                      f"(1/2 C V^2 and 1/2 L I^2), {100 * rel:.2g}% apart",
                 {"supplied_J": e_src, "dissipated_J": e_diss, "stored_J": e_store, "stored_integral_J": e_store_int})


def key_values(stress: list[Stress]) -> dict[str, float]:
    """The numbers convergence is judged on: each part's power and its most used rating."""
    out = {}
    for s in stress:
        out[f"{s.ref} power"] = s.p
        w = s.worst
        if w is not None:
            out[f"{s.ref} {w.what}"] = w.value
    return out


def compare(a: dict[str, float], b: dict[str, float], name: str, how: str) -> Check:
    """The largest relative change between two runs' key values, ignoring values too small to matter."""
    scale = max([abs(v) for v in a.values()] + [1e-12])
    worst, which = 0.0, ""
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        ref = max(abs(va), abs(vb), 1e-3 * scale if "power" in k else 1e-9)
        d = abs(va - vb) / ref
        if d > worst:
            worst, which = d, k
    st = "pass" if worst < 0.01 else "warn" if worst < 0.05 else "fail"
    shown = f"; the most moved is {which}: {fmt(a[which])} and {fmt(b[which])}" if which else ""
    return Check(name, st, f"every part's power and most used rating agree to {100 * worst:.2g}% {how}{shown}",
                 {"change": worst, "value": which})


def provenance(c: Circuit, stress: list[Stress]) -> tuple[Check, list[tuple[str, str]]]:
    """Where each model came from, and what that does to the confidence."""
    lib = [e.ref for e in c.elements if e.part and not e.part.get("generic") and "netlist" not in
           e.part.get("source", "")]
    net = [e.ref for e in c.elements if "as the netlist defines" in e.part.get("source", "")]
    gen = [e.ref for e in c.elements if e.part.get("generic") and e.kind not in ("resistor", "capacitor",
                                                                                  "inductor")]
    passive = [e.ref for e in c.elements if e.part.get("generic") and e.kind in ("resistor", "capacitor",
                                                                                 "inductor")]
    st = {s.ref: s.status for s in stress}
    hot = [r for r in gen if st.get(r) in ("near", "over", "warn", "fails")]
    bits = []
    if lib:
        bits.append(f"{len(lib)} from the parts library")
    if net:
        bits.append(f"{len(net)} as the netlist defines them")
    if gen:
        bits.append(f"{len(gen)} generic ({', '.join(gen[:6])})")
    if passive:
        bits.append(f"{len(passive)} passives with assumed ratings")
    extra = []
    if hot:
        extra.append(("medium", f"{', '.join(hot)} run near a limit on a generic model; check the real part's "
                                "datasheet"))
    status = "warn" if hot else "pass" if not gen else "info"
    return Check("model provenance", status, "models: " + ", ".join(bits) if bits else "no semiconductor models",
                 {"library": lib, "netlist": net, "generic": gen}), extra
