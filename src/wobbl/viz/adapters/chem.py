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


def flame(out, pb, o):
    """A burner with its flame through the sweep of equivalence ratio. The burned gas is colored by the run's
    equilibrium temperature at each mixture and the unburned cone by the reactants'. Above the flame its products
    hang as molecules in their solved shares, CO and NO among them."""
    fr = pb["frames"]
    fl = pb.get("flame") or {}
    phi = np.array([f["phi"] for f in fr], float)
    TK = np.array([f["T_K"] for f in fr], float)
    T0 = float(fl.get("reactant_T_K") or 298.15)
    n = len(fr)
    s = Scene(f"A {fl.get('fuel', 'fuel')} flame in {fl.get('oxidizer', 'air')}", domain="combustion",
              question=o.get("question", ""), length="mm",
              source="Each frame is its own equilibrium solve at that equivalence ratio. The flame's shape is drawn "
                     "for the eye, its colors and the molecules' shares are the run's.")
    s.timeline(phi, unit="", label="equivalence ratio", seconds=12, between="interpolated")
    # the burner
    s.mesh(*lathe([(0, 0), (16, 0), (16, 6), (6, 8), (6, 90), (4.5, 90), (4.5, 10), (0, 10)], 32), name="burner",
           color=P.STEEL, smooth=False)
    s.body("cylinder", [3.5, 3], pos=[-13, 0, 18], quat=P.z_to([1, 0, 0]), color=P.DARK, name="air inlet", prop=True)
    # the flame: an unburned inner cone in a burned envelope, taller and lazier as the mixture runs rich
    tall = 1 + 0.9 * np.clip(phi - 1, 0, None) + 0.4 * np.clip(1 - phi, 0, None)
    Xo, To = lathe([(0, 90), (6.2, 90)] + [((6.2 + 6 * np.sin(np.pi * z * 0.8)) * (1 - z) ** 0.55 + 0.2, 90 + 55 * z)
                                           for z in np.linspace(0.05, 1, 14)] + [(0, 145)], 32)
    Xi, Ti = lathe([(0, 90), (4.6, 90)] + [(4.6 * (1 - z), 90 + 16 * z) for z in np.linspace(0.1, 1, 6)], 32)
    for X, T, name, temp, op in ((Xo, To, "flame", TK, 0.55), (Xi, Ti, "unburned cone", np.full(n, T0), 0.75)):
        m = s.mesh(X, T, name=name, color="#ff9a3c", opacity=op)
        m.color_by("gas temperature", np.outer(temp, np.ones(len(X))), unit="K", colormap="inferno",
                   range=(T0, max(TK.max(), T0 + 1)))
        m.deform(np.c_[np.zeros(len(X)), np.zeros(len(X)), X[:, 2] - 90], scale=tall - 1)
        m.glow = np.clip((temp - 1000) / 1500, 0, 1)
    s.notes = [x for x in s.notes if not x.startswith("Deformation")]
    # the products above the flame
    rng = np.random.default_rng(7)
    sp_all = [k for k in MOLECULES if any(k in f["x"] for f in fr)]
    trace = [k for k in ("CO", "NO") if k in sp_all]
    main = [k for k in sp_all if k not in trace and max(f["x"].get(k, 0) for f in fr) > 1e-3]
    N, K = 46, 10
    count = {}
    for k in main:
        count[k] = np.array([round(f["x"].get(k, 0) / max(sum(f["x"].get(m, 0) for m in main), 1e-12) * N)
                             for f in fr])
    for k in trace:
        x = np.array([f["x"].get(k, 0) for f in fr])
        count[k] = np.ceil(x / max(x.max(), 1e-12) * K - 1e-9).astype(int)
    els, pos = [], []
    scale = 3.0
    for k in main + trace:
        for j in range(int(count[k].max())):
            c = [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(160, 215)]
            a, b = rng.uniform(0, 2 * np.pi, 2)
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ \
                np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            on = (count[k] > j)
            for e, dx, dy in MOLECULES[k]:
                p = np.asarray(c) + R @ np.array([dx, dy, 0.0]) * scale
                els.append(e)
                pos.append(np.where(on[:, None], p, np.nan))
    if els:
        s.atoms(els, np.stack(pos, axis=1), radius=0.35 * scale, name="burned gas")
        s.label("CO and NO molecules drawn to their own peak", [30, 0, 150], kind="note")
    s.readout("flame temperature", TK, unit="K", digits=4)
    for k in trace:
        s.readout(f"{k}", [f["x"].get(k, 0) * 1e6 for f in fr], unit="ppm", digits=3)
    for k in [x for x in ("CO2", "H2O", "O2") if x in main]:
        s.readout(k, [f["x"].get(k, 0) * 100 for f in fr], unit="%", digits=3)
    here = pb["manifest"].get("operating_frame")
    if here is not None:
        s.event(float(phi[here]), "your mixture", f"{TK[here]:.4g} K")
    s.ground(0, size=300, grid=20)
    verdict(s, o)
    s.camera(target=[0, 0, 150], direction=(0.5, -1, 0.25), distance=330)
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
