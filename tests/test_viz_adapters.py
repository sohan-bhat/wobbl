"""Every playback shape the domains write turns into a scene through the adapters."""
import json
from pathlib import Path

import numpy as np
import pytest

from wobbl.viz.adapters import build_scene
from wobbl.viz.scene import load

EX = Path(__file__).resolve().parents[1] / "examples"


def grid_mesh(n=6):
    x, y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
    V = np.c_[x.ravel(), y.ravel(), np.zeros(n * n)]
    i = np.arange(n * n).reshape(n, n)
    F = np.concatenate([np.c_[i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel()],
                        np.c_[i[:-1, :-1].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()]])
    return V, F


def outcome(d, domain, verdict="ok"):
    (d / "outcome.json").write_text(json.dumps({"domain": domain, "verdict": verdict, "headline": "Done.",
                                                "question": "will it work?"}))


def kinds(d):
    h, _ = load(d / "scene.wobbl")
    return {o["kind"] for o in h["objects"]}, h


def test_fields_slice(tmp_path):
    from wobbl.fields import playback as fp
    gx, gy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 15))
    fp.write(tmp_path, "electric", {"e": "field, V/m", "v": "potential, V"},
             {"grid": {"x0": 0, "x1": 10, "y0": 0, "y1": 8, "nx": 20, "ny": 15, "mirror": False},
              "e": fp.pack(gx + gy), "v": fp.pack(gx * 5),
              "parts": [{"name": "plate", "kind": "rect", "data": [1, 1, 3, 2], "hole": True}], "numbers": {"C_pF": 2.0}})
    outcome(tmp_path, "electric")
    assert build_scene(tmp_path)
    k, h = kinds(tmp_path)
    assert {"mesh", "lines", "body"} <= k and h["readouts"][0]["name"] == "C pF"


def test_flow_particles(tmp_path):
    from wobbl.flow import playback as fl
    u = np.ones((12, 30))
    u[5:7, 10:14] = np.nan
    f = {"name": "air", "grid": {"x0": 0, "x1": 0.03, "y0": 0, "y1": 0.012, "nx": 30, "ny": 12, "unit": "m"},
         "components": {"u": fl.quantize(u, "m/s"), "v": fl.quantize(np.where(np.isnan(u), np.nan, 0.0), "m/s")}}
    fl.write(tmp_path, "channel", "Air in a gap", "test", [f], scalars={"dp": (3.0, "Pa")})
    outcome(tmp_path, "flow")
    assert build_scene(tmp_path)
    k, h = kinds(tmp_path)
    assert "points" in k and h["timeline"]["frames"] > 10


def test_molding_fill(tmp_path):
    V, F = grid_mesh()
    tf = V[:, 0] / 10
    pb = {"manifest": {"kind": "molding", "title": "Fill", "gates": [[0, 0, 1]]},
          "mesh": {"vertices": V.tolist(), "triangles": F.tolist()}, "fields": {"t_fill": tf.tolist()},
          "frames": {"time_s": list(np.linspace(0, 1, 10)), "gate_pressure_MPa": list(range(10)),
                     "share_full": list(np.linspace(0, 1, 10))}}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "process")
    assert build_scene(tmp_path)
    _, h = kinds(tmp_path)
    melt = next(o for o in h["objects"] if o.get("name") == "melt")
    assert "reveal" in melt and h["timeline"]["between"] == "exact"


def test_crack_fronts(tmp_path):
    V, F = grid_mesh()
    steps = [{"step": i, "size_mm": 1 + i, "cycles": 1000.0 * i, "k_lead": 5 + i,
              "fronts": [{"xyz": np.c_[np.linspace(0, 1 + i, 5), np.zeros(5), np.full(5, 1.0)].tolist(),
                          "k": np.full((5, 3), 5.0 + i).tolist()}]} for i in range(4)]
    from wobbl.fracture.playback import write
    write(tmp_path / "playback.json", part={"vertices": V.tolist(), "triangles": F.tolist(), "edge_vertices": [],
                                            "edges": []},
          origin=[0, 0, 1], normal=[0, 1, 0], steps=steps, starts=[[[0, 0, 1], [0.5, 0, 1]]],
          growth=([1, 2], [0, 1000]), lives=np.array([1e4, 2e4]), percentiles={"B10": 1e4}, target=None, pof=None,
          inspection={}, verdict="fails", headline="It cracks through.", material="6061", spectrum="", end="edge")
    outcome(tmp_path, "crack", "fails")
    assert build_scene(tmp_path)
    _, h = kinds(tmp_path)
    assert h["verdict"]["tone"] == "fail" and any(o.get("name") == "crack face" for o in h["objects"])


def test_crack_load_label_and_view(tmp_path):
    from types import SimpleNamespace
    from wobbl.structural.spec import Load
    from wobbl.viz.adapters import fracture
    V, F = grid_mesh()
    steps = [{"step": i, "size_mm": 1 + i, "cycles": 1000.0 * i, "k_lead": 5 + i,
              "fronts": [{"xyz": np.c_[np.linspace(0, 1 + i, 5), np.zeros(5), np.full(5, 1.0)].tolist(),
                          "k": np.full((5, 3), 5.0 + i).tolist()}]} for i in range(4)]
    from wobbl.fracture.playback import write
    write(tmp_path / "playback.json", part={"vertices": V.tolist(), "triangles": F.tolist(), "edge_vertices": [],
                                            "edges": []},
          origin=[5, 5, 1], normal=[0, 1, 0], steps=steps, starts=[[[0, 0, 1], [0.5, 0, 1]]],
          growth=([1, 2], [0, 1000]), lives=np.array([1e4, 2e4]), percentiles={"B10": 1e4}, target=None, pof=None,
          inspection={}, verdict="fails", headline="It cracks through.", material="6061", spectrum="", end="edge")
    outcome(tmp_path, "crack", "fails")
    run = SimpleNamespace(mesh=SimpleNamespace(X=V), study=SimpleNamespace(loads=[Load("force", magnitude=250.0)], fixtures=[]),
                          selections={})
    pb = json.loads((tmp_path / "playback.json").read_text())
    fracture.build(tmp_path, pb, run=run, outcome={"question": ""})
    h, _ = load(tmp_path / "scene.wobbl")
    names = {r["name"]: r for r in h["readouts"]}
    assert names["load"]["unit"] == "N" and names["load"]["values"][0] == 250.0
    assert "front outline" in {o.get("name") for o in h["objects"]}
    assert len(h["camera"]["target"]) == 3 and h["camera"]["dist"] > 0


def test_examples_with_playback(tmp_path):
    import shutil
    for name in ("wiper-linkage", "propane-flame", "strut-buckling", "cell-runaway"):
        d = tmp_path / name
        shutil.copytree(EX / name / "output", d, ignore=shutil.ignore_patterns("*.png", "*.jpg", "*.html",
                                                                              "scene.wobbl"))
        assert build_scene(d), name
        h, _ = load(d / "scene.wobbl")
        assert h["verdict"]["word"] and h["timeline"]["frames"] > 1, name


def _names(h):
    return {o.get("name") for o in h["objects"]}


def test_rocket_flies_its_own_airframe(tmp_path):
    from wobbl.motion.spec import read_spec
    from wobbl.motion.study import run_motion
    spec = {"name": "tiny", "rocket": {"mass": "0.6 kg", "diameter": "41 mm", "length": "0.9 m", "cg": "0.55 m",
                                       "cd": 0.5},
            "nose": {"shape": "cone", "length": "15 cm"},
            "fins": {"count": 4, "root": "9 cm", "tip": "4 cm", "span": "5 cm", "sweep": "4 cm"},
            "motor": {"name": "G", "loaded_mass": "0.12 kg", "propellant": "0.06 kg", "length": "12 cm",
                      "thrust": [[0, 0], [0.05, 120], [1.0, 90], [1.3, 0]]},
            "launch": {"rail_length": "1 m", "rail_angle": "3 deg", "wind": "3 m/s"},
            "drogue": {"diameter": "45 cm", "cd": 1.5}}
    st = read_spec(spec, "how high does it go in a 3 m/s wind?")
    run_motion(st, tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert {"rocket", "nose", "fins", "climb", "drogue descent", "landing spot", "launch rail", "wind"} <= names
    assert "exhaust" not in names and "follow" not in h["camera"]
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["label"] == "wind" and tl["unit"] == "m/s" and tl["design"] == 3.0
    assert 3.0 in tl["t"] and tl["t"][0] == 0 and 10 <= tl["frames"] <= 25
    rocket = next(o for o in h["objects"] if o.get("name") == "rocket")
    assert rocket["pose"]["pos"]["shape"][0] == tl["frames"]
    assert h["readouts"][0]["name"] == "apogee"
    land = next(f for f in h["fields"] if f["name"] == "landing speed")
    assert land["limit"] == pytest.approx(st.landing_limit) and land["limit_label"]
    # the asked wind gives exactly the run's own apogee
    o = json.loads((tmp_path / "outcome.json").read_text())
    i = tl["t"].index(3.0)
    assert h["readouts"][0]["values"][i] == pytest.approx(o["values"]["apogee_m"], rel=1e-4)


def test_sweep_inputs_follow_the_question():
    from wobbl.motion.spec import MotionStudy
    from wobbl.motion.sweep import choose
    drop = choose(MotionStudy(name="rock", mass=1, diameter=0.09, height=300, question="how fast from 300 m?"))
    assert drop.label == "drop height" and drop.design == 300 and 300 in drop.values
    assert drop.values[0] < 30 and drop.values[-1] >= 600
    feet = choose(MotionStudy(name="rock", mass=1, diameter=0.09, height=30.48, question="dropped from 100 ft"))
    assert feet.unit == "ft" and feet.design == pytest.approx(100)
    angle = choose(MotionStudy(name="ball", mass=0.4, diameter=0.2, speed=20, angle=35, question="how far?"))
    assert angle.key == "angle" and 35 in angle.values
    speed = choose(MotionStudy(name="ball", mass=0.4, diameter=0.2, speed=20, angle=35, question="how high?"))
    assert speed.key == "speed" and 20 in speed.values


def test_thrown_ball_is_drawn_at_its_size(tmp_path):
    from wobbl.motion.spec import MotionStudy
    from wobbl.motion.study import run_motion
    st = MotionStudy(name="ball", mass=0.43, diameter=0.22, cd=0.25, height=1.5, speed=20, angle=35,
                     question="how far does a kicked ball go?")
    run_motion(st, tmp_path)
    h, blob = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert {"ball", "weight", "drag", "path"} <= names and "cliff" not in names
    assert h["timeline"]["kind"] == "input" and h["timeline"]["label"] == "launch angle"
    speed = next(f for f in h["fields"] if f["name"] == "speed")
    assert speed["limit"] > 0 and speed["limit_label"] == "terminal speed"
    from wobbl.viz.scene import array
    ball = next(o for o in h["objects"] if o.get("name") == "ball")
    X = array(h, blob, ball["pos"])
    assert 0.15 < np.ptp(X[:, 0]) < 0.3


def test_dropped_rock_rises_its_cliff_with_the_slider(tmp_path):
    from wobbl.motion.spec import MotionStudy
    from wobbl.motion.study import run_motion
    st = MotionStudy(name="rock", mass=1.03, diameter=0.09, cd=0.6, height=300,
                     question="How fast does a rock hit the ground after falling off a 300 m cliff?")
    run_motion(st, tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["design"] == 300 and tl["unit"] == "m"
    cliff = next(o for o in h["objects"] if o.get("name") == "cliff")
    assert cliff["deform"]["frames"]["shape"][0] == tl["frames"]
    assert h["readouts"][0]["name"] == "impact speed"
    v = h["readouts"][0]["values"]
    assert all(b >= a for a, b in zip(v, v[1:]))        # a longer fall never lands slower
    o = json.loads((tmp_path / "outcome.json").read_text())
    assert v[tl["t"].index(300)] == pytest.approx(o["values"]["impact_speed_mps"], rel=1e-4)


def test_antenna_wave_plays_one_cycle(tmp_path):
    from wobbl.fields import playback as fp
    r, z = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 14))
    for mirror_z, gain in ((True, [-30.0, 0.0, 2.1, 0.0]), (False, [-5.0, 1.0, 3.0, 1.0])):
        fp.write(tmp_path, "antenna-wire", {"h_re, h_im": "phasor, A/m"},
                 {"grid": {"x0": 0, "x1": 60, "y0": 0, "y1": 60, "nx": 18, "ny": 14, "mirror": True},
                  "h_re": fp.pack(np.cos(6 * np.hypot(r, z))), "h_im": fp.pack(np.sin(6 * np.hypot(r, z))),
                  "wire": {"radius_mm": 0.5, "height_mm": 29.0, "gap_mm": 0.5, "mirror_z": mirror_z},
                  "pattern": {"theta_deg": [0, 45, 90, 135], "gain_dBi": gain}, "frequency_Hz": 2.44e9,
                  "sweep": {"x": [2300, 2500], "s11_dB": [-12, -8]}})
        outcome(tmp_path, "antenna")
        assert build_scene(tmp_path)
        ks, h = kinds(tmp_path)
        assert {"mesh", "body"} <= ks and h["timeline"]["frames"] == 24
        assert any(o.get("name") == "wire" for o in h["objects"])
        assert {r_["name"] for r_ in h["readouts"]} >= {"S11", "peak gain"}


def test_heat_pack_draws_its_cells(tmp_path):
    from wobbl.viz.adapters.common import box_mesh
    Xb, Tb = box_mesh([18.5, 18.5, 32.5], [18.5, 18.5, 32.5])
    g = np.stack(np.meshgrid(np.linspace(0, 37, 8), np.linspace(0, 37, 8), np.linspace(0, 65, 8)), -1).reshape(-1, 3)
    X = np.vstack([Xb, g])
    hist = np.array([35 + k * (1 + X[:, 0] / 37) for k in range(1, 6)], float)
    np.savez(tmp_path / "fields.npz", X=X, T=hist[-1], tris=Tb, tets=np.zeros((0, 4), int),
             hist_t=np.arange(1, 6) * 60.0, hist_T=hist)
    (tmp_path / "study.json").write_text(json.dumps({"material": "li-ion-cell", "ambient": 35}))
    outcome(tmp_path, "thermal")
    assert build_scene(tmp_path)
    _, h = kinds(tmp_path)
    cells = [o for o in h["objects"] if o.get("name", "").startswith("cell ")]
    assert len(cells) == 4 and all("field" in c for c in cells)
    assert h["title"] == "The 4 cell pack heating up"


MOTOR = """
name = "t"
propellant = "knsb"
[[grain]]
shape = "{shape}"
diameter = "38 mm"
length = "60 mm"
count = 2
{extra}
[nozzle]
throat = "8 mm"
exit = "20 mm"
[casing]
material = "6061-t6"
outer_diameter = "44 mm"
wall = "2 mm"
length = "130 mm"
target_sf = 2
"""


@pytest.mark.parametrize("shape,extra", [("star", 'points = 5\npoint_length = "6 mm"\npoint_width = "3 mm"'),
                                         ("tube", 'core = "12 mm"')])
def test_motor_cutaway_is_swept_over_its_throat(tmp_path, shape, extra):
    from wobbl.chem.motor import run_motor
    from wobbl.chem.spec import read_spec
    from wobbl.viz.scene import array
    run_motor(read_spec(MOTOR.format(shape=shape, extra=extra), "will the casing hold?"), tmp_path, render=False)
    assert build_scene(tmp_path)
    h, blob = load(tmp_path / "scene.wobbl")
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["label"] == "nozzle throat" and tl["design"] == pytest.approx(8.0)
    assert 10 <= tl["frames"] <= 25 and min(tl["t"]) < 8.0 < max(tl["t"])
    names = _names(h)
    assert {"casing", "grain", "grain 2", "burning port", "nozzle", "exhaust"} <= names
    hoop = next(f for f in h["fields"] if f["name"] == "hoop stress")
    assert hoop["limit"] > 0 and hoop["limit_label"]
    ro = {r["name"]: r["values"] for r in h["readouts"]}
    assert h["readouts"][0]["name"] == "peak pressure" and h["readouts"][0]["limit"] > 0
    assert ro["peak pressure"][0] > 1.5 * ro["peak pressure"][-1]     # a narrower throat raises the pressure
    assert {"casing safety factor", "hoop stress", "total impulse"} <= set(ro)
    assert "follow" not in h["camera"] and h["camera"]["presets"]
    nozzle = next(o for o in h["objects"] if o.get("name") == "nozzle")
    d = array(h, blob, nozzle["deform"]["frames"])
    r = np.hypot(*(array(h, blob, nozzle["pos"])[None, :, :2] + d[:, :, :2]).transpose(2, 0, 1))
    assert r[-1].min() > r[0].min() + 1                   # the throat opens along the slider
    if shape == "star":
        port = next(o for o in h["objects"] if o.get("name") == "burning port")
        pr = np.hypot(*array(h, blob, port["pos"])[:, :2].T)
        assert np.ptp(pr) > 2                             # its outline is not a circle


def test_motor_sweeps_what_the_question_turns_on():
    from wobbl.chem.motor_sweep import apply, choose
    from wobbl.chem.spec import read_spec
    sp = read_spec(MOTOR.format(shape="bates", extra='core = "12 mm"'))
    assert choose(sp, "will the casing hold?").key == "throat"
    core = choose(sp, "is the core too narrow for erosive burning?")
    assert core.key == "core" and core.design == pytest.approx(12.0) and 12.0 in core.values
    wall = choose(sp, "how thick a wall does it need?")
    assert wall.key == "wall" and apply(sp, "wall", 3.0).casing.wall == pytest.approx(0.003)


def test_flame_on_a_burner_for_any_fuel(tmp_path):
    frames = [{"phi": p, "T_K": 1500 + 600 * (1 - abs(p - 1.05)),
               "x": {"N2": 0.7, "CO2": 0.1, "H2O": 0.15, "CO": 0.01 * p, "NO": 0.001 / p, "O2": 0.03 * (2 - p)}}
              for p in np.linspace(0.6, 1.6, 9)]
    pb = {"manifest": {"kind": "flame", "operating_frame": 4}, "frames": frames,
          "flame": {"fuel": "CH4", "oxidizer": "air", "reactant_T_K": 300}}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "chemistry")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    assert {"burner", "flame", "unburned cone", "burned gas"} <= _names(h)
    assert {"flame temperature", "CO", "NO"} <= {r["name"] for r in h["readouts"]}
    gas = next(o for o in h["objects"] if o.get("name") == "burned gas")
    assert "C" in gas["el"] and "N" in gas["el"]


def _species(formula, phase, el):
    return {"formula": formula, "phase": phase, "coef": 1, "atoms": [{"element": e} for e in el],
            "positions": [[i * 1.2, 0, 0] for i in range(len(el))], "bonds": []}


@pytest.mark.parametrize("wet", [True, False])
def test_reaction_runs_in_its_vessel(tmp_path, wet):
    from wobbl.chem.scene import write
    if wet:
        re = [_species("HCl", "aq", "HCl"), _species("CaCO3", "s", ["Ca", "C", "O", "O", "O"])]
        pr = [_species("CO2", "g", ["C", "O", "O"]), _species("H2O", "l", ["H", "H", "O"])]
        extra = {"gas_L": 24.0, "adiabatic": {"rise_K": 3.0}}
    else:
        re = [_species("Mg", "s", ["Mg"]), _species("CuO", "s", ["Cu", "O"])]
        pr = [_species("MgO", "s", ["Mg", "O"]), _species("Cu", "s", ["Cu"])]
        extra = {"gas_L": 0.0, "adiabatic": {"T_K": 2600.0, "rise_K": 2300.0}}
    pb = {"manifest": {"kind": "reaction"}, "equation": "a -> b", "reactants": re, "products": pr,
          "energies": {"dH_kJ": -300.0}, "conditions": {"T_K": 298.15}, "equilibrium_extent": 1.0,
          "frames": [{"T_K": 300, "dG_kJ": -1, "log10K": 1}], **extra}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "chemistry")
    assert write(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    if wet:
        assert {"beaker", "solution", "CO2 bubbles"} <= names
    else:
        assert {"crucible", "charge", "reaction front", "sparks"} <= names
    assert h["timeline"]["unit"] == "%" and h["timeline"]["frames"] > 30
