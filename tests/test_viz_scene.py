"""The scene file: arrays decode back to the run's values, the budget holds, decimation keeps real vertices."""
import numpy as np
import pytest

from wobbl.viz import Scene
from wobbl.viz.mesh import boundary, decimate
from wobbl.viz.scene import array, load


def sphere(n=40):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    X = np.c_[np.sin(th).ravel() * np.cos(ph).ravel(), np.sin(th).ravel() * np.sin(ph).ravel(), np.cos(th).ravel()]
    i = np.arange(n * 2 * n).reshape(n, 2 * n)
    j = np.roll(i, -1, axis=1)
    T = np.concatenate([np.c_[i[:-1].ravel(), i[1:].ravel(), j[1:].ravel()],
                        np.c_[i[:-1].ravel(), j[1:].ravel(), j[:-1].ravel()]])
    return X * 10, T


def test_round_trip():
    X, T = sphere()
    F = 30
    s = Scene("check", domain="test")
    s.timeline(np.linspace(0, 3, F), unit="s", between="interpolated")
    v = X[:, 2] * 4.0
    m = s.mesh(X, T, max_vertices=100000)
    m.color_by("height", v, unit="mm", scale=np.linspace(0, 1, F))
    s.readout("t", np.linspace(0, 3, F), unit="s")
    s.event(1.5, "half way")
    h, blob = load(s.to_bytes())
    o = h["objects"][0]
    assert h["format"] == "wobbl-scene" and h["timeline"]["frames"] == F
    P = array(h, blob, o["pos"])
    assert np.abs(P - X[np.unique(T)]).max() < 20 / 65535 * 1.01
    vals = array(h, blob, o["field"]["values"])
    assert np.abs(vals - v[np.unique(T)]).max() < 80 / 65535 * 1.01
    assert o["field"]["scale"][-1] == 1.0 and h["readouts"][0]["values"][-1] == 3.0
    assert h["events"][0]["t"] == 1.5


def test_frames_thin_to_fit_the_budget():
    rng = np.random.default_rng(0)
    F, n = 400, 2000
    s = Scene("busy")
    s.timeline(np.arange(F, dtype=float))
    s.points(rng.normal(size=(F, n, 3)), values=rng.normal(size=(F, n)), field="noise")
    s.readout("k", np.arange(F))
    data = s.to_bytes(budget=300_000)
    h, _ = load(data)
    assert len(data) <= 300_000
    assert 16 <= h["timeline"]["frames"] < F
    assert h["readouts"][0]["values"][-1] == F - 1      # the last frame always stays


def test_decimation_keeps_original_vertices_and_the_peak():
    X, T = sphere(80)
    peak = int(np.argmax(X[:, 0] + 0.3 * X[:, 1]))
    keep, T2 = decimate(X, T, 800, keep=[peak])
    assert len(keep) <= 800 and T2.max() < len(keep)
    assert peak in keep
    assert set(keep) <= set(np.unique(T))


def test_boundary_of_a_tet_is_outward():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    tris = boundary(X, np.array([[0, 1, 2, 3]]))
    c = X.mean(0)
    for t in tris:
        n = np.cross(X[t[1]] - X[t[0]], X[t[2]] - X[t[0]])
        assert np.dot(n, X[t].mean(0) - c) > 0


def test_per_frame_arrays_must_match_the_timeline():
    s = Scene("bad")
    s.timeline([0, 1, 2])
    with pytest.raises(ValueError):
        s.readout("x", [1, 2])


def test_rigid_bodies_keep_quaternions_continuous():
    s = Scene("spin")
    s.timeline([0, 1, 2])
    q = np.array([[1, 0, 0, 0], [-0.99, 0, 0, -0.1], [0.98, 0, 0, 0.2]], float)
    s.body("box", [1, 1, 1], pos=np.zeros((3, 3)), quat=q)
    h, blob = load(s.to_bytes())
    Q = array(h, blob, h["objects"][0]["quat"])
    assert (np.sum(Q[1:] * Q[:-1], axis=1) > 0).all()


def test_readout_target_marks_marginal_with_or_without_a_limit():
    s = Scene("check", domain="test")
    s.timeline([1.0, 2.0, 3.0], unit="N", label="load", kind="input", design=2.0)
    s.readout("stress", [10.0, 20.0, 30.0], unit="MPa", limit=40.0, target=15.0)
    s.readout("mass flux", [1.0, 2.0, 3.0], target=1.5)
    s.readout("plain", [1.0, 2.0, 3.0], limit=4.0)
    h, _ = load(s.to_bytes())
    ro = {r["name"]: r for r in h["readouts"]}
    assert ro["stress"]["limit"] == 40.0 and ro["stress"]["target"] == 15.0
    assert ro["mass flux"]["target"] == 1.5 and "limit" not in ro["mass flux"]
    assert "target" not in ro["plain"]


def test_a_motion_carries_its_own_readouts_a_body_value_and_a_fluid_inside():
    s = Scene("check", domain="test")
    s.timeline([1.0, 2.0], unit="N", label="load", kind="input", design=2.0)
    s.motion([0.0, 0.5, 1.0], [0.0, 1.0, 2.0])
    b = s.body("box", [1, 1, 1], pos=[0, 0, 0])
    b.color_by("share", [0.2, 0.4], unit="%", range=(0, 1))
    b.track(pos=np.zeros((3, 3)), field=[0.1, 0.5, 0.9])
    s.lines([np.array([[0, 0, 0], [1, 0, 0]], float)], name="coolant").inside()
    s.motion_readout("altitude", [0.0, 5.0, np.nan], unit="m")
    with pytest.raises(ValueError):
        s.motion_readout("speed", [1.0, 2.0])
    h, _ = load(s.to_bytes())
    assert h["motion"]["readouts"] == [{"name": "altitude", "unit": "m", "digits": 3, "values": [0.0, 5.0, None]}]
    assert h["objects"][0]["track"]["value"] == {"f": 0, "values": [0.1, 0.5, 0.9]}
    assert h["objects"][1]["inside"] is True


def test_steady_motions_keep_a_rate_per_input_frame_on_one_clock():
    X, T = sphere(10)
    s = Scene("rotor", domain="test")
    s.timeline([1000.0, 2000.0, 3000.0], unit="rpm", label="rotor speed", kind="input", design=2000.0)
    rpm = np.array([1000.0, 2000.0, 3000.0])
    part = s.mesh(X, T)
    part.deform(np.zeros((3, len(X), 3)) + [0, 0, 1])
    part.swing(2 * rpm / 60)
    rotor = s.body("box", [5, 0.5, 0.1], pos=[0, 0, 12], name="rotor").spin(rpm / 60, axis=(0, 0, 2)).ride(part)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    link = s.body("capsule", [0.1, 1], name="link")
    link.cycle(np.c_[np.cos(ang), np.sin(ang), 0 * ang], np.c_[np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)], 0.75)
    with pytest.raises(ValueError):
        s.to_bytes()                              # the objects move on a clock the scene has not set
    with pytest.raises(ValueError):
        rotor.spin(1.0)                           # one steady motion per object
    with pytest.raises(ValueError):
        s.body("box", [1, 1, 1]).spin([1.0, 2.0])  # a rate per frame or one for all
    with pytest.raises(ValueError):
        s.mesh(X, T).swing(1.0)                   # a swing needs a shape
    s.steady(200, note="shown 200 times slower")
    h, blob = load(s.to_bytes())
    assert h["steady"] == {"slow": [200.0], "note": "shown 200 times slower"}
    o = h["objects"]
    assert o[0]["steady"]["swing"] is True and o[0]["steady"]["rate"] == pytest.approx(2 * rpm / 60, rel=1e-5)
    assert o[1]["steady"]["spin"] == {"axis": [0.0, 0.0, 1.0], "center": [0.0, 0.0, 0.0]}
    assert o[1]["steady"]["rate"] == pytest.approx(rpm / 60, rel=1e-5) and o[1]["ride"] == 0
    cy = o[2]["steady"]
    assert cy["rate"] == [0.75] and np.abs(array(h, blob, cy["cycle"]["pos"])[:, 0] - np.cos(ang)).max() < 1e-4
    Q = array(h, blob, cy["cycle"]["quat"])
    assert (np.sum(Q[1:] * Q[:-1], axis=1) > 0).all()


def test_a_sweep_keeps_the_event_at_each_value_it_solved():
    s = Scene("drops", domain="test")
    s.timeline([0.5, 1.0, 2.0], unit="m", label="drop height", kind="input", design=1.0)
    s.readout("impact speed", [3.1, 4.4, 6.3], unit="m/s")
    b = s.body("sphere", [0.1], name="ball")
    for h in (0.5, 1.0, 2.0):
        tf = np.sqrt(2 * h / 9.81)
        t = np.linspace(0, tf, 5)
        s.motion(t, t, at=h)
        b.track(pos=np.c_[0 * t, 0 * t, h - 0.5 * 9.81 * t * t])
        s.readout_motion("impact speed", 9.81 * t)
        s.motion_readout("height", h - 0.5 * 9.81 * t * t, unit="m")
    with pytest.raises(ValueError):
        s.motion([0.0, 1.0], [0.0, 1.0], at=2.0)         # one motion a value
    h, blob = load(s.to_bytes())
    mo = h["motion"]
    assert [v["at"] for v in mo["variants"]] == [0.5, 1.0, 2.0] and mo["asked"] == 1 and mo["at"] == 1.0
    assert mo["t"][-1] == pytest.approx(np.sqrt(2 / 9.81), rel=1e-6)
    o = h["objects"][0]
    assert len(o["tracks"]) == 3 and o["track"] == o["tracks"][1]
    top = [array(h, blob, tr["pos"])[0, 2] for tr in o["tracks"]]
    assert top == pytest.approx([0.5, 1.0, 2.0], abs=1e-4)
    r = h["readouts"][0]
    assert len(r["motions"]) == 3 and r["motion"] == r["motions"][1]
    assert [v["readouts"][0]["values"][0] for v in mo["variants"]] == [0.5, 1.0, 2.0]
