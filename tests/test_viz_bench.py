"""Circuit and mechanism runs of any shape turn into 3D scenes: a bench with real parts for a circuit, the links
with their joints, motors and setting for a mechanism."""
import json

import numpy as np

from wobbl.viz.adapters import build_scene
from wobbl.viz.props_bench import BAND, GOLD, bands
from wobbl.viz.scene import load


def _names(h):
    return [o.get("name", "") for o in h["objects"]]


def _outcome(d, domain, question, verdict="ok"):
    (d / "outcome.json").write_text(json.dumps({"domain": domain, "verdict": verdict, "headline": "Done.",
                                                "question": question}))


def test_resistor_bands_follow_the_value():
    assert bands(47) == [BAND[4], BAND[7], BAND[0], GOLD]
    assert bands(10_000) == [BAND[1], BAND[0], BAND[3], GOLD]
    assert bands(4.7) == [BAND[4], BAND[7], GOLD, GOLD]
    assert bands(None) == [] and bands(0) == []


def _pin(name, net, x, y, cur=0.0):
    return {"pin": name, "net": net, "hole": [int(x), "e"], "x": x, "y": y, "current": cur}


def _board(parts, wires, k=None, sweep=None):
    plan = {"board": {"cols": 30, "width": 33.4, "height": 18.2,
                      "rows": {"t+": 1.0, "t-": 2.0, "a": 3.6, "e": 7.6, "f": 10.6, "j": 14.6, "b-": 16.2,
                               "b+": 17.2}},
            "parts": parts, "wires": wires, "peak": 0.05,
            "instruments": [{"name": "V1", "kind": "supply", "readings": [["9.00", "V"]], "current": 0.05,
                             "leads": [{"post": "+", "net": "vin", "x": 3.2, "y": 1.0, "current": 0.05, "k": k},
                                       {"post": "-", "net": "0", "x": 2.2, "y": 2.0, "current": -0.05, "k": k}]}]}
    return {"plan": plan, "key": {"label": "R1 POWER", "value": 300, "unit": "mW", "limit": 250,
                                  "series": None}, "span": [0, 1e-3] if k else None, "sweep": sweep}


def _parts(k=None):
    return [
        {"name": "R1", "kind": "resistor", "body": "axial-resistor", "label": "120 Ω", "value": 120.0,
         "pins": [_pin("1", "vin", 4.2, 3.6, 0.05), _pin("2", "a", 9.2, 3.6, -0.05)], "current": 0.05, "k": k,
         "power": 0.3, "heat": 1.2},
        {"name": "D1", "kind": "led", "body": "led-5mm", "color": "green", "label": "LED_green_20",
         "pins": [_pin("a", "a", 10.2, 7.6, 0.02), _pin("k", "0", 10.2, 10.6, -0.02)], "current": 0.02, "k": k,
         "power": 0.04, "heat": 0.66},
        {"name": "C1", "kind": "electrolytic", "body": "radial-capacitor", "value": 1e-4,
         "pins": [_pin("1", "vin", 13.2, 7.6), _pin("2", "0", 13.2, 10.6)], "current": 0.0, "heat": None},
        {"name": "C2", "kind": "capacitor", "body": "radial-capacitor", "value": 1e-7,
         "pins": [_pin("1", "vin", 15.2, 7.6), _pin("2", "0", 15.2, 10.6)], "current": 0.0, "heat": None},
        {"name": "U1", "kind": "regulator", "body": "to220", "label": "7805",
         "pins": [_pin("in", "vin", 18.2, 3.6), _pin("g", "0", 19.2, 3.6), _pin("o", "b", 20.2, 3.6)],
         "current": 0.01, "power": 0.05, "heat": 0.1},
        {"name": "U2", "kind": "mcu", "body": "dip", "label": "ATtiny85",
         "pins": [_pin(str(i), "b", 22.2 + i % 4, 7.6 if i < 4 else 10.6) for i in range(8)], "current": 0.0},
        {"name": "L1", "kind": "inductor", "body": "inductor", "label": "10 uH",
         "pins": [_pin("1", "b", 26.2, 3.6), _pin("2", "c", 28.2, 3.6)], "current": 0.0},
    ]


def test_transient_bench_plays_heat_glow_and_a_failing_part(tmp_path):
    k = list(np.round(np.linspace(0.1, 1.0, 25), 3))
    wires = [{"kind": "jumper", "net": "a", "color": "#3fae5a", "points": [[9.2, 3.6], [10.2, 7.6]],
              "current": 0.05, "k": k},
             {"kind": "tap", "net": "0", "color": "#2b2e33", "points": [[10.2, 14.6], [10.2, 16.2]],
              "current": -0.02, "k": None}]
    (tmp_path / "board.json").write_text(json.dumps(_board(_parts(k), wires, k)))
    _outcome(tmp_path, "circuit", "does the resistor burn?", "fails")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert h["timeline"]["frames"] == 25 and h["timeline"]["unit"] == "ms"
    assert {"bench", "breadboard", "R1", "R1 band", "D1 dome", "C1", "C2", "U1 tab", "U2", "L1", "V1"} <= set(names)
    assert names.count("R1 band") == 4
    r1 = next(o for o in h["objects"] if o.get("name") == "R1")
    heat = r1["field"]["values"]
    assert heat[0] < 1 < heat[-1]                               # it warms with the current and passes its rating
    dome = next(o for o in h["objects"] if o.get("name") == "D1 dome")
    assert dome["glow"][-1] > dome["glow"][0]
    ring = next(o for o in h["objects"] if o.get("name") == "R1 over rating")
    assert ring["when"][0] > 0
    assert any(e["kind"] == "fail" and "R1" in e["name"] for e in h["events"])
    flow = next(o for o in h["objects"] if o.get("name") == "current")
    assert len(flow["flow"]["speed"]) == 25
    assert {"supply current", "R1 load"} <= {r["name"] for r in h["readouts"]}


def test_operating_point_without_a_sweep_still_stands_on_the_bench(tmp_path):
    (tmp_path / "board.json").write_text(json.dumps(_board(_parts(), [])))
    _outcome(tmp_path, "circuit", "is this fine?")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    assert h["timeline"]["frames"] == 1 and "R1" in _names(h)


def test_swept_supply_drives_the_timeline(tmp_path):
    parts = _parts()
    n = 25
    sw = {"source": "V1", "unit": "V", "values": list(np.linspace(0, 9, n)), "plan": _board(parts, [])["plan"],
          "parts": [list(np.linspace(0, p.get("current") or 0, n)) for p in parts], "wires": [],
          "leads": [[list(np.linspace(0, 0.05, n)), list(np.linspace(0, -0.05, n))]],
          "power": {"R1": list(np.linspace(0, 0.3, n) ** 2 / 0.3)}}
    (tmp_path / "board.json").write_text(json.dumps(_board(parts, [], sweep=sw)))
    _outcome(tmp_path, "circuit", "will a blue LED fry?", "fails")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    assert h["timeline"]["frames"] == n and h["timeline"]["unit"] == "V"
    sup = next(r for r in h["readouts"] if r["name"] == "supply")
    assert sup["values"][-1] == 9
    led = next(o for o in h["objects"] if o.get("name") == "D1")
    assert led["color"] == "#34c759"                           # a model that names its color beats the question


ARM = """<mujoco model="arm"><worldbody>
<geom name="floor" type="plane" size="0 0 0.05"/>
<body name="link1" pos="0 0 0.1"><joint name="shoulder" type="hinge" axis="0 -1 0"/>
<geom name="link1" type="capsule" fromto="0 0 0 0.2 0 0" size="0.01" mass="0.1"/>
<body name="link2" pos="0.2 0 0"><joint name="elbow" type="hinge" axis="0 -1 0"/>
<geom name="link2" type="capsule" fromto="0 0 0 0.15 0 0" size="0.008" mass="0.05"/>
<geom name="payload" type="sphere" pos="0.15 0 0" size="0.02" mass="0.25"/>
</body></body></worldbody></mujoco>"""


def _q_y(a):
    return np.c_[np.cos(a / 2), np.zeros_like(a), np.sin(a / 2), np.zeros_like(a)]


def _qmul(a, b):
    from wobbl.viz.adapters.mech_joints import qmul
    return qmul(a, b)


def _arm_playback(t):
    """A two link arm lifting, posed as MuJoCo records it: fromto capsules point their z from the end back."""
    a1, a2 = 0.6 * t, 0.4 * t                        # about -y, so a positive angle lifts
    q1 = _q_y(-a1)
    q2 = _qmul(q1, _q_y(-a2))
    d1 = np.c_[np.cos(a1), 0 * t, np.sin(a1)]
    d2 = np.c_[np.cos(a1 + a2), 0 * t, np.sin(a1 + a2)]
    base = np.array([0, 0, 0.1])
    elbow = base + 0.2 * d1
    tip = elbow + 0.15 * d2
    capq = _q_y(np.full_like(t, -np.pi / 2))          # turns z onto -x, from the end back to the start
    T = lambda q, p, extra: {"pos": p.tolist(), "quat": _qmul(q, extra).tolist()}  # noqa: E731
    n = len(t)
    need = 0.4 + 0.3 * t
    bodies = [{"name": "floor", "shape": "plane", "size": [0, 0, 0.05], "static": True, "pos": [0, 0, 0],
               "quat": [1, 0, 0, 0], "color": "#333333"},
              {"name": "link1", "shape": "capsule", "size": [0.01, 0.1, 0], "static": False, "color": "#3dcf7a",
               **T(q1, base + 0.1 * d1, capq)},
              {"name": "link2", "shape": "capsule", "size": [0.008, 0.075, 0], "static": False, "color": "#ff6b6b",
               **T(q2, elbow + 0.075 * d2, capq)},
              {"name": "payload", "shape": "sphere", "size": [0.02, 0, 0], "static": False, "color": "#d9dbe0",
               "pos": tip.tolist(), "quat": q2.tolist()}]
    joint = lambda v: {"angle": {"unit": "rad", "values": list(v)}, "speed": {"unit": "rad/s",  # noqa: E731
                                                                               "values": [0.5] * n},
                       "torque": {"unit": "N m", "values": list(np.minimum(need, 0.6))},
                       "needed": {"unit": "N m", "values": list(need)},
                       "limit": {"unit": "N m", "values": [0.6] * n}}
    return {"manifest": {"kind": "mechanism", "title": "robot arm", "suggested_seconds": 3},
            "t": list(t), "bodies": bodies, "joints": {"shoulder": joint(a1), "elbow": joint(a2)},
            "series": {"shoulder current": {"unit": "A", "values": list(need * 2)}},
            "events": [{"name": "pick", "t_s": 0.0}, {"name": "placed", "t_s": float(t[-1])},
                       {"name": "shoulder servo at its limit", "t_s": 0.67}]}


def test_joints_are_found_from_the_model_and_recorded_poses(tmp_path):
    from wobbl.viz.adapters.mech_joints import joints
    t = np.linspace(0, 1, 30)
    pb = _arm_playback(t)
    (tmp_path / "model.xml").write_text(ARM)
    J = joints(tmp_path, pb)
    assert set(J) == {"shoulder", "elbow"}
    assert np.allclose(J["shoulder"]["anchor"], [0, 0, 0.1], atol=1e-6)
    a1 = 0.6 * t
    assert np.allclose(J["elbow"]["anchor"], np.c_[0.2 * np.cos(a1), 0 * t, 0.1 + 0.2 * np.sin(a1)], atol=1e-6)
    assert np.allclose(J["elbow"]["axis"], [0, -1, 0], atol=1e-6)


def test_arm_on_a_desk_with_motors_and_payload(tmp_path):
    t = np.linspace(0, 1, 30)
    (tmp_path / "playback.json").write_text(json.dumps(_arm_playback(t)))
    (tmp_path / "model.xml").write_text(ARM)
    _outcome(tmp_path, "mechanism", "can this desktop arm lift 250 g?", "fails")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = _names(h)
    assert {"desk", "shoulder motor", "elbow motor", "pick spot", "placed spot"} <= set(names)
    motor = next(o for o in h["objects"] if o.get("name") == "shoulder motor")
    load_ = motor["field"]["values"]
    assert load_[0] < 1 < load_[-1]                            # it turns red once the joint asks for more
    assert any("250 g payload" in lb["text"] for lb in h["labels"])
    ro = {r["name"]: r for r in h["readouts"]}
    assert {"shoulder torque", "shoulder speed", "shoulder current"} <= set(ro)
    tq = ro["shoulder torque"]
    assert tq["values"][-1] > tq["limits"][-1] and tq["values"][0] < tq["limits"][0]


def test_wiper_gets_its_windshield_and_blade(tmp_path):
    t = np.linspace(0, 1, 30)
    pb = _arm_playback(t)
    pb["joints"]["elbow"].pop("needed")
    pb["joints"]["elbow"].pop("limit")
    pb["bodies"] = [b for b in pb["bodies"] if b["name"] != "floor"]
    # the elbow stands in for a grounded output link: pin link2 at a fixed pivot so it swings in place
    piv = np.array([0.3, 0.0, 0.1])
    a = 0.8 * t
    q = _q_y(-a)
    d = np.c_[np.cos(a), 0 * t, np.sin(a)]
    pb["bodies"][1].update({"pos": (piv + 0.075 * d).tolist(), "quat": _qmul(q, _q_y(np.full_like(t, -np.pi / 2))).tolist()})
    pb["bodies"] = pb["bodies"][:2]
    xml = ARM.replace('<body name="link2" pos="0.2 0 0">', '</body><body name="link2" pos="0.3 0 0.1">') \
             .replace("</body></body></worldbody>", "</body></worldbody>")
    (tmp_path / "model.xml").write_text(xml)
    (tmp_path / "playback.json").write_text(json.dumps(pb))
    _outcome(tmp_path, "mechanism", "does this wiper sweep the windshield?")
    assert build_scene(tmp_path)
    h, _ = load(tmp_path / "scene.wobbl")
    names = set(_names(h))
    assert {"windshield", "wiper arm", "wiper blade", "swept edge"} <= names
    blade = next(o for o in h["objects"] if o.get("name") == "wiper blade")
    assert blade["pos"]["shape"][0] == h["timeline"]["frames"]
