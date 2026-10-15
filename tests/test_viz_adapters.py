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
    tl = h["timeline"]
    assert names["load"]["unit"] == "N" and tl["kind"] == "input" and tl["design"] == 250.0
    assert names["load"]["values"][tl["t"].index(250.0)] == 250.0
    assert "fronts" in {o.get("name") for o in h["objects"]}
    assert len(h["camera"]["target"]) == 3 and h["camera"]["dist"] > 0


def test_crack_load_sweep_judges_the_target_life(tmp_path):
    """With the run's lives at each load, the crack's K limit sits where one part in ten fails before the target
    and its target where one in a hundred does, so the color's verdict is the question's."""
    from wobbl.fracture.playback import write
    V, F = grid_mesh()
    steps = [{"step": i, "size_mm": 1 + i, "cycles": 1000.0 * i, "k_lead": 5 + i,
              "fronts": [{"xyz": np.c_[np.linspace(0, 1 + i, 5), np.zeros(5), np.full(5, 1.0)].tolist(),
                          "k": np.c_[np.full(5, 5.0 + i), np.zeros((5, 2))].tolist()}]} for i in range(4)]
    f = np.linspace(0.5, 1.5, 11)
    sweep = {"factor": f.tolist(), "nominal": (2e5 * f ** -4).tolist(), "B1": (8e4 * f ** -4).tolist(),
             "B10": (1.5e5 * f ** -4).tolist(), "pof": np.clip(0.01 * f ** 8, 0, 1).tolist(), "load_N": 4000.0,
             "kc": 29.0, "target": 1e5, "max_pof": 0.01}
    write(tmp_path / "playback.json", part={"vertices": V.tolist(), "triangles": F.tolist(), "edge_vertices": [],
                                            "edges": []},
          origin=[0, 0, 1], normal=[0, 1, 0], steps=steps, starts=[[[0, 0, 1], [0.5, 0, 1]]],
          growth=([1, 2], [0, 1000]), lives=np.array([1e4, 2e4]), percentiles={"B10": 1e4}, target=1e5, pof=0.5,
          inspection={}, verdict="fails", headline="It cracks.", material="7075", spectrum="", end="edge",
          sweep=sweep)
    outcome(tmp_path, "fracture", "fails")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl, fld = h["timeline"], h["fields"][0]
    assert tl["kind"] == "input" and tl["unit"] == "N" and tl["design"] == 4000.0
    assert h["readouts"][0]["name"] == "B1 life"
    # B10 = 1e5 at f = 1.5 ** 0.25 and B1 at 0.8 ** 0.25, against the largest K of 8
    assert fld["limit"] == pytest.approx(8 * 1.5 ** 0.25, rel=0.02)
    assert fld["target"] == pytest.approx(8 * 0.8 ** 0.25, rel=0.02)
    assert not any("reveal" in o for o in h["objects"])


def _life_pb(kind, X, F, **extra):
    from wobbl.life.playback import mesh_block
    return {"format": "wobbl-playback", "version": 1, "kind": kind, "mesh": mesh_block(X, F), **extra}


def test_vibration_sweeps_the_rotor_speed(tmp_path):
    from wobbl.life.playback import quantize
    V, F = grid_mesh()
    phi = np.c_[np.zeros((len(V), 2)), V[:, 0] / 10]
    modes = [{"mode": 1, "frequency_Hz": 100.0, "status": "resonant", "mass_share_xyz": [0, 0, 0.8],
              "shape": quantize(phi, "int8")}]
    pb = _life_pb("vibration", V, F, modes=modes, amplitude_mm=1.0,
                  excitations=[{"label": "once per turn", "lo_Hz": 50.0, "hi_Hz": 150.0}])
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "vibration", "verdict": "fails", "headline": "Rings.",
                                                       "question": "q", "values": {"damping": 0.05, "margin": 0.2}}))
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl, ro = h["timeline"], h["readouts"]
    assert tl["kind"] == "input" and tl["unit"] == "rpm" and tl["t"][0] == 3000 and tl["t"][-1] == 9000
    assert tl["design"] == pytest.approx(6000, abs=100)          # the worst speed, at resonance
    assert ro[0]["name"] == "amplification" and max(ro[0]["values"]) == pytest.approx(10.0, rel=0.02)
    assert h["fields"][0]["limit"] and h["fields"][0]["target"] < h["fields"][0]["limit"]
    assert any(e["kind"] == "fail" for e in h["events"])


def test_buckling_sweeps_the_load_to_the_buckling_load(tmp_path):
    from wobbl.life.playback import quantize
    V, F = grid_mesh()
    phi = np.c_[np.zeros(len(V)), np.sin(np.pi * V[:, 0] / 10), np.zeros(len(V))]
    pb = _life_pb("buckling", V, F, shape=quantize(phi, "int8"), buckling_factor=4.0, design_factor=3.0,
                  yield_factor=6.0, length_mm=10.0, stress_MPa=quantize(np.full(len(V), 20.0), "int16"),
                  frames={"load_factor": [0, 1], "bow_mm": [0.1, 0.133]})
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "buckling", "verdict": "holds", "headline": "OK.",
                                                       "question": "q", "values": {"applied_load_N": 1000.0,
                                                                                   "target_sf": 2.0}}))
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl, fld = h["timeline"], h["fields"][0]
    assert tl["kind"] == "input" and tl["design"] == 1.0 and tl["t"][-1] == pytest.approx(3.8)
    assert fld["name"] == "stress" and fld["limit"] == pytest.approx(60.0, rel=0.01)
    assert fld["target"] == pytest.approx(30.0, rel=0.01)
    assert h["readouts"][0]["name"] == "factor against buckling"


def test_drop_sweeps_the_height_and_leaves_the_contact_patch_out(tmp_path):
    from wobbl.life.playback import encode, quantize
    V, F = grid_mesh()
    V = V[:, [0, 2, 1]]                                     # a wall standing on its edge, landing along -z
    S = np.full(len(V), 30.0)
    S[V[:, 2] == 0] = 400.0                                 # the crush where it touches
    S[len(V) // 2 + 3] = 40.0
    pb = _life_pb("drop", V, F,
                  fall={"height_mm": 1000.0, "impact_speed_mm_s": 4429.0, "down": [0, 0, -1], "floor": "concrete",
                        "floor_name": "Concrete", "landing": "-z face down", "floor_level_mm": 0.0},
                  frames={"time_s": [0.0, 1e-4], "translation": quantize(np.zeros((2, 3))),
                          "coeff": quantize(np.array([[0.0], [1.0]])),
                          "stress": encode(np.round(np.vstack([0 * S, S]) / 400 * 255).astype("uint8"), "uint8",
                                           scale=400 / 255.0, offset=0.0)},
                  shapes=quantize(np.full((1, len(V), 3), 0.01)), history={"g_watch": [0, 900], "force_N": [0, 50]},
                  governing={"stress_MPa": 40.0, "at_mm": V[len(V) // 2 + 3].tolist(), "time_s": 1e-4})
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "drop", "verdict": "marginal", "headline": "M.",
                                                       "question": "q", "metrics": [["target", "2"]],
                                                       "values": {"allowable_MPa": 62.0}}))
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl, ro = h["timeline"], {r["name"]: r for r in h["readouts"]}
    assert tl["kind"] == "input" and tl["unit"] == "m" and tl["design"] == 1.0
    assert h["readouts"][0]["name"] == "peak stress" and h["fields"][0]["target"] == 31.0
    at = tl["t"].index(1.0)
    assert ro["peak stress"]["values"][at] == pytest.approx(40.0, rel=0.02)
    assert ro["peak stress"]["values"][-1] == pytest.approx(40.0 * np.sqrt(2), rel=0.02)


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


def test_antenna_field_follows_the_frequency_picked(tmp_path):
    from wobbl.fields import playback as fp
    r, z = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 14))
    freqs = [2300.0, 2440.0, 2580.0]
    grids = [np.exp(-3 * np.hypot(r, z)) * a for a in (0.4, 1.0, 0.6)]
    fp.write(tmp_path, "antenna-wire", {"h_rms": "rms field per frequency, A/m"},
             {"grid": {"x0": 0, "x1": 60, "y0": 0, "y1": 60, "nx": 18, "ny": 14, "mirror": True},
              "h_rms": [fp.pack(g, 0.0, 1.0) for g in grids], "tx_power_W": 0.01,
              "wire": {"radius_mm": 0.5, "height_mm": 29.0, "gap_mm": 0.5, "mirror_z": True},
              "pattern": {"theta_deg": [0, 45, 90, 135], "gain_dBi": [-30.0, 0.0, 2.1, 0.0]},
              "patterns": [[-30.0, 0.0, 1.8, 0.0], [-30.0, 0.0, 2.1, 0.0], [-30.0, 0.5, 2.3, 0.5]],
              "frequency_Hz": 2.44e9,
              "sweep": {"x": freqs, "s11_dB": [-6.0, -14.0, -8.0], "r_ohm": [40, 60, 80], "x_ohm": [-30, 0, 30]}})
    outcome(tmp_path, "antenna")
    assert build_scene(tmp_path)
    ks, h = kinds(tmp_path)
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["unit"] == "MHz" and tl["design"] == 2440.0 and tl["t"] == freqs
    assert h["readouts"][0]["name"] == "S11" and h["readouts"][0]["values"] == [-6.0, -14.0, -8.0]
    f = h["fields"][0]
    assert f["limit"] == pytest.approx(0.163, abs=1e-3) and f["limit_label"]
    pat = next(o for o in h["objects"] if o.get("name") == "gain pattern")
    assert pat["deform"]["frames"]["shape"][0] == 3
    assert not any(o.get("flow") or o.get("drawn") for o in h["objects"])


def test_antenna_run_with_only_its_working_phasor_shows_that_field(tmp_path):
    from wobbl.fields import playback as fp
    r, z = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 14))
    fp.write(tmp_path, "antenna-wire", {"h_re, h_im": "phasor, A/m"},
             {"grid": {"x0": 0, "x1": 60, "y0": 0, "y1": 60, "nx": 18, "ny": 14, "mirror": True},
              "h_re": fp.pack(np.cos(6 * np.hypot(r, z))), "h_im": fp.pack(np.sin(6 * np.hypot(r, z))),
              "wire": {"radius_mm": 0.5, "height_mm": 29.0, "gap_mm": 0.5, "mirror_z": False},
              "pattern": {"theta_deg": [0, 45, 90, 135], "gain_dBi": [-5.0, 1.0, 3.0, 1.0]}, "frequency_Hz": 2.44e9,
              "sweep": {"x": [2300, 2500], "s11_dB": [-12, -8]}})
    outcome(tmp_path, "antenna")
    assert build_scene(tmp_path)
    ks, h = kinds(tmp_path)
    assert {"mesh", "body"} <= ks and h["timeline"]["frames"] == 1
    assert h["readouts"][0]["name"] == "S11" and h["fields"][0]["limit"] > 0


def _pack_mesh():
    from wobbl.viz.adapters.common import box_mesh
    Xb, Tb = box_mesh([18.5, 18.5, 32.5], [18.5, 18.5, 32.5])
    g = np.stack(np.meshgrid(np.linspace(0, 37, 8), np.linspace(0, 37, 8), np.linspace(0, 65, 8)), -1).reshape(-1, 3)
    return np.vstack([Xb, g]), Tb


def _pack(tmp_path, study, **extra):
    X, Tb = _pack_mesh()
    hist = np.array([35 + k * (1 + X[:, 0] / 37) for k in range(1, 6)], float)
    np.savez(tmp_path / "fields.npz", X=X, T=hist[-1], tris=Tb, tets=np.zeros((0, 4), int),
             hist_t=np.arange(1, 6) * 60.0, hist_T=hist, **extra)
    (tmp_path / "study.json").write_text(json.dumps({"material": "li-ion-cell", "ambient": 35, "power": 6000,
                                                     "duration": 300, **study}))
    (tmp_path / "outcome.json").write_text(json.dumps({"domain": "thermal", "verdict": "hot", "headline": "Hot.",
                                                       "question": "too hot?", "values": {"limit_C": 42}}))
    assert build_scene(tmp_path)
    return X, hist, kinds(tmp_path)[1]


def test_heat_pack_draws_its_cells_at_any_heat_load(tmp_path):
    X, hist, h = _pack(tmp_path, {})
    cells = [o for o in h["objects"] if o.get("name", "").startswith("cell ")]
    assert len(cells) == 4 and all("field" in c for c in cells)
    assert h["title"] == "The 4 cell pack after 5 min at each heat load"
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["label"] == "heat load" and tl["unit"] == "W" and tl["design"] == 6.0
    assert tl["between"] == "exact" and tl["t"][0] == 0 and tl["t"][-1] >= 12
    hot = h["readouts"][0]
    assert hot["name"].startswith("hottest") and hot["limit"] == 42
    at = tl["t"].index(6.0)
    assert hot["values"][at] == pytest.approx(hist[-1].max(), abs=0.01)    # the asked power is the solved field
    assert hot["values"][0] == pytest.approx(35)                           # no heat, the air's temperature
    f = h["fields"][0]
    assert f["limit"] == 42 and f["limit_label"] == "limit"
    assert {"time over 42 C"} <= {r["name"] for r in h["readouts"]}
    assert any(e["kind"] == "fail" and 0 < e["t"] < 6 for e in h["events"])


def test_heat_sweep_with_radiation_blends_the_solved_powers(tmp_path):
    X, _ = _pack_mesh()
    T = 35 + 5 * (1 + X[:, 0] / 37)
    W = np.array([0.0, 3.0, 6.0, 12.0])
    F = np.array([35 + (w / 6) ** 0.8 * (T - 35) for w in W])
    _, _, h = _pack(tmp_path, {"emissivity": 0.9}, sweep_W=W, sweep_T=F.astype(np.float32),
                    sweep_hot=np.outer(W / 6, np.arange(1, 6)) + 35)
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["between"] == "interpolated" and tl["t"] == [0, 3, 6, 12]
    assert h["readouts"][0]["values"][2] == pytest.approx(F[2].max(), abs=0.01)


def test_heat_powers_reach_past_the_limit():
    from wobbl.viz.adapters.thermal import crossing, linear, powers
    k = powers(15, 70, 40, 85)
    assert k[0] == 0 and 1.0 in k and k[-1] == 2.0
    assert powers(150, 34, 25, 60)[-1] == pytest.approx(5.0)          # a cool part runs on to where it passes
    assert crossing(np.array([0, 10, 20]), np.array([40, 60, 80]), 70) == pytest.approx(15)
    assert linear({"ambient": 40, "films": [{"h": 0.006, "t_inf": 40}]})
    assert not linear({"ambient": 40, "emissivity": 0.85})
    assert not linear({"ambient": 40, "films": "[{'h': 0.03, 't_inf': 43.4}]"})


def test_cell_runaway_drags_the_oven(tmp_path):
    ovens = [60.0, 100.0, 150.0, 154.7, 154.9, 170.0]
    pb = {"manifest": {"kind": "cell-runaway"}, "frames": [],
          "cell": {"label": "18650", "diameter_mm": 18, "length_mm": 65, "critical_C": 154.9, "surroundings_C": 74.5},
          "sweep": {"oven_C": ovens, "peak_C": [60, 101, 160, 203, 205, 260], "peak_min": [150, 60, 120, 72, 70, 30],
                    "self_heating_C_min": [0, 0.3, 0.9, 9.9, 11.4, 30], "sei_left": [1, 0.5, 0, 0, 0, 0],
                    "electrolyte_left": [1, 1, 0.9, 0.5, 0.5, 0.2], "runaway": [False] * 4 + [True] * 2}}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "battery")
    assert build_scene(tmp_path)
    _, h = kinds(tmp_path)
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["unit"] == "C" and tl["design"] == 74.5 and tl["t"] == ovens
    assert h["readouts"][0]["name"] == "peak self heating" and h["readouts"][0]["limit"] == 10
    f = h["fields"][0]
    assert f["name"] == "self heating" and f["limit"] == 10 and f["limit_label"] == "runaway"
    assert any(e["name"] == "runaway starts" and e["t"] == 154.9 for e in h["events"])


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
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["design"] == 1.0 and "follow" not in h["camera"]
    f = next(f for f in h["fields"] if f["name"] == "gas temperature")
    assert f["limit"] == 1800 and f["limit_label"] and h["readouts"][0]["name"] == "flame temperature"


def _species(formula, phase, el):
    return {"formula": formula, "phase": phase, "coef": 1, "atoms": [{"element": e} for e in el],
            "positions": [[i * 1.2, 0, 0] for i in range(len(el))], "bonds": []}


def _sweep(wet):
    x = np.linspace(0, 168, 17) if wet else np.linspace(14, 40, 13)
    if wet:
        pts = [{"x": g, "gas_L": 24.5 * min(g / 84, 1), "heat_kJ": -31 * min(g / 84, 1), "T_K": 298 - 6 * min(g / 84, 1),
                "left_g": max(g - 84, 0)} for g in x]
        return {"input": {"what": "added", "name": "baking soda added", "unit": "g", "design": 84.0,
                          "solution_g": 1200.0}, "limit": {"T_K": 373.15, "label": "water boils"}, "points": pts}
    pts = [{"x": v, "heat_kJ": 850 * min(v / 25, 1), "T_K": 3000 - 20 * abs(v - 25), "charge_g": 200 + v,
            "heat_kJ_per_kg": 3900, "left": {"Al": max(v - 25, 0)} if v > 25 else {"Fe2O3": 25 - v}} for v in x]
    return {"input": {"what": "mixture", "name": "aluminum share of the charge", "unit": "% by mass", "design": 25.0},
            "limit": {"T_K": 3139.0, "label": "Fe boils"}, "points": pts}


@pytest.mark.parametrize("wet", [True, False])
def test_reaction_is_driven_by_its_input(tmp_path, wet):
    from wobbl.chem.scene import write
    if wet:
        re = [_species("HCl", "aq", "HCl"), _species("CaCO3", "s", ["Ca", "C", "O", "O", "O"])]
        pr = [_species("CO2", "g", ["C", "O", "O"]), _species("H2O", "l", ["H", "H", "O"])]
    else:
        re = [_species("Al", "s", ["Al"]), _species("Fe2O3", "s", ["Fe", "Fe", "O", "O", "O"])]
        pr = [_species("Al2O3", "s", ["Al", "Al", "O", "O", "O"]), _species("Fe", "s", ["Fe"])]
    pb = {"manifest": {"kind": "reaction"}, "equation": "a -> b", "reactants": re, "products": pr,
          "energies": {"dH_kJ": -300.0}, "conditions": {"T_K": 298.15}, "equilibrium_extent": 1.0,
          "frames": [{"T_K": 300, "dG_kJ": -1, "log10K": 1}], "sweep": _sweep(wet)}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "chemistry")
    assert write(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["design"] == pb["sweep"]["input"]["design"] and tl["frames"] >= 13
    if wet:
        assert {"beaker", "solution", "CO2 gathered at 1 atm"} <= names
        assert h["readouts"][0]["name"] == "CO2 given off" and h["readouts"][0]["values"][-1] == 24.5
    else:
        assert {"crucible", "burned charge"} <= names and h["readouts"][0]["name"] == "heat released"
        assert "sparks" not in names and "reaction front" not in names
    assert all("limit" in f and f["limit_label"] for f in h["fields"])
    assert "follow" not in h["camera"] and h["camera"]["presets"]


def test_fatigue_sweeps_the_duty_load_on_its_strength_scene(tmp_path):
    from wobbl.viz.scene import Scene
    V, F = grid_mesh()
    src = tmp_path / "strength"
    s = Scene("part", domain="strength")
    s.timeline(np.linspace(0, 1, 3), unit="N", label="load", between="exact")
    s.mesh(V, F, name="part").color_by("von Mises", V[:, 0] * 10, scale=np.linspace(0, 1, 3), unit="MPa", limit=200)
    s.save(src)
    run = tmp_path / "fatigue"
    run.mkdir()
    (run / "study.json").write_text(json.dumps({"source": "../strength"}))
    g = np.linspace(0.05, 3.0, 60)
    pb = {"format": "wobbl-playback", "kind": "fatigue", "where_mm": [10, 0, 0],
          "frames": {"cycles": [0, 1], "damage": [0, 0.5]},
          "sweep": {"stress_scale": g.tolist(), "damage": (0.5 * g ** 8).tolist(), "required": 10,
                    "unit": "repeats"}}
    (run / "playback.json").write_text(json.dumps(pb))
    (run / "outcome.json").write_text(json.dumps({"domain": "fatigue", "verdict": "marginal", "headline": "M.",
                                                  "question": "q", "values": {"sf": 0.5 ** -0.125,
                                                                              "target_sf": 2.0}}))
    assert build_scene(run)
    h, _ = load(run / "scene.wobbl")
    tl, fld, ro = h["timeline"], h["fields"][0], h["readouts"]
    assert tl["kind"] == "input" and tl["design"] == 100.0 and fld["limit"] == 1.0 and fld["target"] == 0.5
    assert ro[0]["name"] == "damage" and ro[0]["values"][tl["t"].index(100.0)] == pytest.approx(0.5, rel=0.02)



@pytest.mark.skipif(not __import__("wobbl.chem.install", fromlist=["x"]).available(), reason="Cantera not installed")
def test_reaction_sweeps_solve_each_input():
    from wobbl.chem import react, sweep
    from wobbl.chem.resolve import resolve
    q = resolve("baking soda and vinegar")
    react._phases(q, 298.15)
    th = react.thermo(q, 298.15)
    v = {"dH_kJ": th["dH"] / 1000, "gas_L": 24.47, "adiabatic": react._aqueous_rise(q, th, 298.15)}
    sw = sweep.reaction(q, v, 298.15, 101325.0)
    gas = [p["gas_L"] for p in sw["points"]]
    assert sw["input"]["unit"] == "g" and abs(sw["input"]["design"] - 84.0) < 0.1
    assert gas[0] == 0 and gas[-1] == gas[len(gas) // 2] == 24.47 and sw["points"][-1]["left_g"] > 80
    q = resolve("thermite")
    react._phases(q, 298.15)
    th = react.thermo(q, 298.15)
    v = {"dH_kJ": th["dH"] / 1000, "adiabatic": {"kind": "complete"}}
    sw = sweep.reaction(q, v, 298.15, 101325.0)
    T = {p["ratio"]: p["T_K"] for p in sw["points"]}
    assert sw["limit"]["label"] == "Fe boils" and abs(sw["limit"]["T_K"] - 3139) < 3
    assert T[1.0] == max(T.values()) and T[0.5] < T[1.0] - 500 and T[2.0] < T[1.0]


def test_molecule_steps_through_its_modes_by_hand(tmp_path):
    from wobbl.chem.scene import write
    pos = [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]
    vib = [{"kind": "vibration", "name": f"{f} cm-1", "frequency_cm": f, "intensity_km_mol": i,
            "vector": [[0, 0, 0], [0.25, 0, 0], [-0.25 * s, 0, 0]], "frames": [pos] * 16}
           for f, i, s in ((667, 50, 1), (1388, 0, 1), (2349, 600, -1))]
    pb = {"manifest": {"kind": "molecule"}, "molecule": {"name": "co2", "formula": "CO2"},
          "atoms": [{"element": e} for e in "COO"], "positions": pos, "bonds": [[0, 1, 2], [0, 2, 2]],
          "animations": vib, "charges": [0.6, -0.3, -0.3], "properties": {"dipole_D": 0.0}}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "chemistry")
    assert write(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["frames"] == 27 and abs(tl["design"] - 3.4) < 1e-6
    f = next(f for f in h["fields"] if f["name"] == "partial charge")
    assert f["limit"] == 0.5 and f["limit_label"]
    assert h["readouts"][0]["name"] == "frequency" and {"co2", "bonds"} <= _names(h)


def test_crystal_strains_by_hand(tmp_path):
    from wobbl.chem.scene import write
    x = np.linspace(-5, 5, 11)
    cells = [np.eye(3) * 3.6 * (1 + e / 100) for e in x]
    base = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    eos = {"kind": "equation of state", "energies_eV_per_atom": list(0.005 * x ** 2), "volumes_A3": [3.6 ** 3] * 11,
           "cells": [c.tolist() for c in cells], "frames": [(base @ c).tolist() for c in cells],
           "strain_pct": list(x), "pressure_GPa": list(-5 * x), "bulk_GPa": list(133 - 10 * x)}
    pb = {"manifest": {"kind": "crystal"}, "crystal": {"name": "copper"}, "cell": cells[5].tolist(),
          "atoms": [{"element": "Cu"}] * 4, "positions": (base @ cells[5]).tolist(), "animations": [eos],
          "properties": {"lattice_A": 3.6, "fusion_eV_atom": 0.136}}
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    outcome(tmp_path, "chemistry")
    assert write(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    assert h["timeline"]["kind"] == "input" and h["timeline"]["design"] == 0.0
    assert h["readouts"][0]["name"] == "bulk modulus" and h["readouts"][0]["values"][5] == 133
    f = next(f for f in h["fields"] if f["name"] == "strain energy")
    assert f["limit"] == 0.136 and {"copper", "unit cell"} <= _names(h)



def _strength_run(sf):
    from types import SimpleNamespace as NS
    V, F = grid_mesh()
    U = np.c_[np.zeros(len(V)), np.zeros(len(V)), 0.01 * V[:, 0]]
    vm = 2.0 * V[:, 0] + 1.0
    run = NS(mesh=NS(X=V, tris=F), measure=vm, result=NS(U=U), selections={},
             study=NS(loads=[NS(kind="force", magnitude=40.0, direction=None, point=None)], fixtures=[]))
    o = NS(question="will it hold 40 N?", governing_MPa=float(vm.max()), allowable_MPa=sf * float(vm.max()),
           governing_xyz=V[np.argmax(vm)].tolist(), measure="von Mises (MPa)", safety_factor=sf, part="arm.step",
           max_disp_mm=0.1, hotspots=[], verdict="holds" if sf >= 2 else "fails", target_sf=2.0,
           governing_where="at the tip")
    return run, o


@pytest.mark.parametrize("sf", [0.6, 2.5, 40.0])
def test_strength_scene_is_a_load_sweep_past_failure(tmp_path, sf):
    """The viewer sets the load: the frames sweep it from none to past where the part yields, open at the asked
    load, and every colored field carries its limit so the viewer can call the verdict."""
    from wobbl.viz.adapters.strength import write_scene
    run, o = _strength_run(sf)
    h, _ = load(write_scene(run, o, tmp_path))
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["unit"] == "N" and tl["design"] == 40.0 and tl["between"] == "exact"
    assert tl["t"][0] == 0 and 40.0 in tl["t"] and tl["t"][-1] >= 80.0
    if sf < 10:
        assert tl["t"][-1] > sf * 40.0           # the sweep reaches the load that yields it
        assert [e["name"] for e in h["events"]] == ["first yield"]
    assert all("limit" in f and f.get("limit_label") for f in h["fields"])
    first = h["readouts"][0]
    assert first["name"] == "peak stress" and first["limit"] == pytest.approx(sf * o.governing_MPa)
    i = tl["t"].index(40.0)
    assert first["values"][i] == pytest.approx(o.governing_MPa, rel=1e-3)


def test_shell_scene_sweeps_the_load_in_newtons(tmp_path):
    from types import SimpleNamespace as NS

    from wobbl.shells.scene import ramp_scene
    V, F = grid_mesh()
    forces = np.zeros((len(V), 3))
    forces[-1] = [0, 0, -120.0]
    model = NS(mesh=NS(X=V, tris=F, n_nodes=len(V)), forces=forces, rigid=[], constraints=[])
    vals = V[:, 0] + 1.0
    ramp_scene(tmp_path, title="Lid", domain="laminate", question="120 N?", model=model, values=vals / 11,
               field="failure index", unit="", U=np.c_[0 * V[:, :2], -0.01 * V[:, 0]], reserve=5.0, limit=1.0,
               limit_label="first ply failure", keep=len(V) - 1, peak_label="ply 1",
               readouts=[("worst ply index", 0.2, "", 1.0), ("deflection", 0.1, "mm", None)],
               events=[(5.0, "first ply failure", "ply 1", "fail")], verdict=("HOLDS", "It holds.", "pass"),
               source="linear")
    h, _ = load(tmp_path / "scene.wobbl")
    tl = h["timeline"]
    assert tl["kind"] == "input" and tl["design"] == 120.0 and tl["t"][-1] >= 600.0
    assert h["events"][0]["t"] == pytest.approx(600.0)
    assert h["fields"][0]["limit"] == 1.0 and h["readouts"][0]["name"] == "worst ply index"
    assert h["readouts"][-1]["name"] == "load"

