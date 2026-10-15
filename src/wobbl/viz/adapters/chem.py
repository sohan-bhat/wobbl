"""Chemistry runs as scenes: a solid motor burning on a test stand, a flame's burned gas as molecules, a cell
heating in an oven, and the flow down a nozzle.

Every number comes from the run's playback.json. Where a picture needs something the run does not decide, such as
where a gas molecule happens to sit or the flame's shape, the view says it is drawn for the eye.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import props as P
from ..scene import Scene
from .common import lathe, read, verdict

# small molecules as atoms at bond lengths in angstrom, laid along x and in the xy plane
MOLECULES = {
    "N2": [("N", -0.55, 0), ("N", 0.55, 0)], "O2": [("O", -0.6, 0), ("O", 0.6, 0)], "H2": [("H", -0.37, 0), ("H", 0.37, 0)],
    "CO": [("C", -0.56, 0), ("O", 0.56, 0)], "NO": [("N", -0.58, 0), ("O", 0.58, 0)], "OH": [("O", -0.48, 0), ("H", 0.48, 0)],
    "CO2": [("O", -1.16, 0), ("C", 0, 0), ("O", 1.16, 0)], "H2O": [("O", 0, 0), ("H", 0.757, 0.586), ("H", -0.757, 0.586)],
    "H": [("H", 0, 0)], "O": [("O", 0, 0)], "N": [("N", 0, 0)], "Ar": [("Ar", 0, 0)],
}


def build(out: Path, pb: dict) -> Path | None:
    kind = pb.get("manifest", {}).get("kind") or pb.get("kind")
    o = read(out, "outcome.json") or {}
    fn = {"flame": flame, "cell-runaway": runaway, "nozzle": nozzle}.get(kind)
    return fn(out, pb, o) if fn else None


def _place(counts: dict, order: list, slots: np.ndarray, rng) -> dict:
    """Slot index of every molecule of every species in a frame, species after species."""
    at, k = {}, 0
    for sp in order:
        n = counts.get(sp, 0)
        at[sp] = list(range(k, k + n))
        k += n
    return at


def flame(out, pb, o):
    fr = pb["frames"]
    phi = np.array([f["phi"] for f in fr], float)
    TK = np.array([f["T_K"] for f in fr], float)
    order = [sp for sp in sorted({k for f in fr for k in f["x"]}, key=lambda k: -max(f["x"].get(k, 0) for f in fr))
             if sp in MOLECULES]
    N = 80
    counts = []
    for f in fr:
        x = np.array([f["x"].get(sp, 0) for sp in order])
        raw = x / max(x.sum(), 1e-12) * N
        c = np.floor(raw).astype(int)
        c[np.argsort(-(raw - c))[: N - c.sum()]] += 1
        counts.append(dict(zip(order, c)))
    side = int(np.ceil(N ** (1 / 3)))
    g = np.array([(i, j, k) for k in range(side) for j in range(side) for i in range(side)], float)[:N] * 4.2
    rng = np.random.default_rng(7)
    turn = rng.uniform(0, np.pi, N)
    mx = {sp: max(c[sp] for c in counts) for sp in order}
    els, slots = [], []
    for sp in order:
        for j in range(mx[sp]):
            for a in MOLECULES[sp]:
                els.append(a[0])
                slots.append((sp, j, a[1], a[2]))
    F = len(fr)
    pos = np.full((F, len(els), 3), np.nan)
    for f in range(F):
        at = _place(counts[f], order, g, rng)
        for i, (sp, j, dx, dy) in enumerate(slots):
            if j < len(at[sp]):
                k = at[sp][j]
                c, s_ = np.cos(turn[k]), np.sin(turn[k])
                pos[f, i] = g[k] + [dx * c - dy * s_, dx * s_ + dy * c, 0]
    s = Scene("The burned gas as molecules", domain="combustion", question=o.get("question", ""), length="angstrom",
              source="Each frame is its own equilibrium solve. The share of each molecule is the run's mole fraction, "
                     "rounded to 80 molecules, and where they sit is drawn for the eye.")
    s.timeline(phi, unit="", label="equivalence ratio", seconds=10, between="step")
    s.atoms(els, pos, radius=0.75, name="burned gas")
    box = g.max(0) + 2.5
    P.wire_box(s, [-2.5, -2.5, -2.5], box, name="sample box")
    s.readout("flame temperature", TK, unit="K", digits=4)
    for sp in [x for x in ("CO2", "H2O", "CO", "NO", "O2") if x in order][:4]:
        s.readout(sp, [f["x"].get(sp, 0) * 100 for f in fr], unit="%", digits=3)
    here = pb["manifest"].get("operating_frame")
    if here is not None:
        s.event(float(phi[here]), "your mixture", f"{TK[here]:.4g} K")
    verdict(s, o)
    s.camera(target=box / 2 - 1, direction=(0.8, -1, 0.6), distance=float(np.linalg.norm(box)) * 1.7)
    return s.save(out)


def runaway(out, pb, o):
    c, fr = pb["cell"], pb["frames"]
    t = np.array([f["t_min"] for f in fr], float)
    D, L = c["diameter_mm"], c["length_mm"]
    s = Scene(f"An {c.get('label', 'cell')} in two ovens either side of the onset", domain="battery",
              question=o.get("question", ""), length="mm",
              source="Each cell is one lumped temperature, as the model has it, so it heats evenly.")
    s.timeline(t, unit="min", label="time", seconds=10, between="interpolated")
    for k, (side, x) in enumerate((("below", -L * 0.9), ("above", L * 0.9))):
        T = np.array([f[side]["T_C"] for f in fr], float)
        b = s.body("cylinder", [D / 2, L / 2], pos=[x, 0, L / 2 + 6], color="#3c6e9e", name=f"cell {side}")
        b.color_by("cell temperature", T, unit="C", colormap="inferno", range=(min(20, T.min()), 300),
                   limit=c.get("critical_C"), limit_label="onset")
        b.light(np.clip((T - 150) / 350, 0, 1))
        oven = c.get(f"{side}_oven_C")
        P.wire_box(s, [x - L * 0.6, -L * 0.6, 0], [x + L * 0.6, L * 0.6, L * 1.4], name="oven")
        if oven is not None:
            s.label(f"oven {oven:.4g} C", [x, 0, L * 1.5], kind="note")
        s.readout(f"cell {side}", T, unit="C", digits=4, limit=c.get("critical_C"))
    s.ground(0, size=L * 5, grid=10, color="#1d1f23")
    for key in ("sei", "electrolyte"):
        s.readout(f"{key} left, hot oven", [max(f["above"][key], 0) * 100 for f in fr], unit="%", digits=3)
    if c.get("trigger_min"):
        s.event(c["trigger_min"], "runaway starts", c.get("trigger", ""), kind="fail")
        s.label("runs away", [L * 0.9, 0, L + 14], t0=c["trigger_min"], kind="fail")
    verdict(s, o)
    s.camera(target=[0, 0, L * 0.6], direction=(0.2, -1, 0.45), distance=L * 4.2)
    return s.save(out)


def nozzle(out, pb, o):
    fr = pb["frames"]
    ar = np.array([f["area_ratio"] for f in fr], float)
    rt = 10.0
    r = rt * np.sqrt(np.maximum(ar, 1.0))
    z = -(r - rt) / np.tan(np.radians(15))
    X, T = lathe([(rt * 2.2, rt * 3), (rt * 2.2, rt * 1.2), (rt, 0)] + [(ri, zi) for ri, zi in zip(r[1:], z[1:])
                                                                          if zi < 0][::4] + [(r[-1], z[-1])], 48)
    s = Scene("Down the nozzle", domain="combustion", question=o.get("question", ""), length="mm",
              source="Each frame is an equilibrium expansion at that area ratio, drawn on a 15 degree cone.")
    s.timeline(ar, unit="", label="area ratio", seconds=7, between="interpolated")
    s.mesh(X, T, name="nozzle", color="#8d939b", opacity=0.45)
    TK = np.array([f.get("T_K", np.nan) for f in fr], float)
    disc = s.body("cylinder", [1.0, 0.6], pos=np.c_[np.zeros(len(ar)), np.zeros(len(ar)), z], color="#ff9a3c",
                  name="gas")
    disc.color_by("gas temperature", TK, unit="K", colormap="inferno")
    for k, name, unit in (("P_MPa", "pressure", "MPa"), ("T_K", "temperature", "K"), ("v_m_s", "speed", "m/s"),
                          ("isp_s", "Isp", "s")):
        if k in fr[0]:
            s.readout(name, [f.get(k) for f in fr], unit=unit, digits=4)
    verdict(s, o)
    return s.save(out)
