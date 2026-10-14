"""Make the synthetic strain gauge readings in readings.csv. They are not from a real test.

The readings come from Wobbl's own converged solution of the mount, at the gauge spots `wobbl uncertainty` picked,
read the way a gauge would (averaged over a 3 mm grid), with Young's modulus set to 70.6 GPa instead of the
nominal 72.4 GPa and random gauge noise added. Since they come from the model itself, comparing the model with
them shows the workflow, how calibration finds the modulus back and what the outputs look like. It says nothing
about how well the model matches a real mount, which only a real test can.

    python make_readings.py output > readings.csv
"""
import json
import sys
from pathlib import Path

import numpy as np

from wobbl.credibility.probes import gauge, load_fields, surface

E_NOMINAL, E_TRUE, NU = 72400.0, 70600.0, 0.33
LOADS = (500.0, 1000.0, 1500.0, 2000.0)

out = Path(sys.argv[1])
u = json.loads((out / "uncertainty.json").read_text())
f = load_fields(out)
surf = surface(f)
rng = np.random.default_rng(2026)
lines = ["# Synthetic readings, not from a real test. Made by make_readings.py from Wobbl's own solution with",
         "# E = 70.6 GPa and gauge noise, to show the workflow. They say nothing about how well the model matches",
         "# a real mount.",
         "name,quantity,x,y,z,direction,load,value,uncertainty,unit"]
spots = {p["kind"]: p for p in u["probes"]}
o = json.loads((out / "outcome.json").read_text())
lp = next(c["detail"] for c in o["checks"] if c["name"] == "load introduction")
# SG3 sits where the rigid attachment would put its peak, the spot the test plan picks to tell the two apart
spots["discriminating"] = {"xyz": lp["alternative_xyz"], "direction": None}
for name, kind in (("SG1", "peak"), ("SG2", "far field"), ("SG3", "discriminating")):
    p = spots[kind]
    g = gauge(f, p["xyz"], E_NOMINAL, NU, p["direction"], surf=surf)
    xyz = ",".join(f"{v:.2f}" for v in g["xyz"])
    axis = ",".join(f"{v:.4f}" for v in g["direction"])
    for load in LOADS:
        eps = g["strain"] * load / 2000.0 * E_NOMINAL / E_TRUE * 1e6
        sd = max(5.0, 0.02 * abs(eps))
        lines.append(f'{name},strain,{xyz},"{axis}",{load:g},{eps + sd * rng.standard_normal():.1f},{sd:.1f},'
                     "microstrain")
p = spots["deflection"]
i = int(np.argmin(np.linalg.norm(f["X"] - np.asarray(p["xyz"]), axis=1)))
d = np.asarray(p["direction"])
for load in LOADS:
    w = float(f["U"][i] @ d) * load / 2000.0 * E_NOMINAL / E_TRUE
    xyz = ",".join(f"{v:.2f}" for v in f["X"][i])
    axis = ",".join(f"{v:.4f}" for v in d)
    lines.append(f'D1,displacement,{xyz},"{axis}",{load:g},{w + 0.005 * rng.standard_normal():.4f},0.005,mm')
print("\n".join(lines))
