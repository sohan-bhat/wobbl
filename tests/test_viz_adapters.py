"""Every playback shape the domains write turns into a scene through the adapters."""
import json
from pathlib import Path

import numpy as np

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
    run_motion(read_spec(spec, "how high does it go?"), tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert {"rocket", "nose", "fins", "exhaust", "drogue parachute", "launch rail", "wind"} <= names
    rocket = next(o for o in h["objects"] if o.get("name") == "rocket")
    assert rocket["pose"]["pos"]["shape"][0] == h["timeline"]["frames"]
    assert {r["name"] for r in h["readouts"]} >= {"altitude", "speed", "Mach", "thrust"}


def test_thrown_ball_is_drawn_at_its_size(tmp_path):
    from wobbl.motion.spec import MotionStudy
    from wobbl.motion.study import run_motion
    st = MotionStudy(name="ball", mass=0.43, diameter=0.22, cd=0.25, height=1.5, speed=20, angle=35,
                     question="how far does a kicked ball go?")
    run_motion(st, tmp_path)
    h, blob = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert {"ball", "weight", "drag", "path"} <= names and "cliff" not in names
    from wobbl.viz.scene import array
    ball = next(o for o in h["objects"] if o.get("name") == "ball")
    X = array(h, blob, ball["pos"])
    assert 0.15 < np.ptp(X[:, 0]) < 0.3

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
