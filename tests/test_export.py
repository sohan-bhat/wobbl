"""A finished run exported as a file: a GIF, an MP4 when ffmpeg is there, a PNG or the report, drawn from the run's
own scene with Wobbl's numpy rasterizer and saved where the user asked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wobbl.export import ExportError, export, save
from wobbl.export.clip import design_position, render, shots, still
from wobbl.export.sample import Scene as Posable
from wobbl.viz import Scene

ROOT = Path(__file__).resolve().parents[1]
MOTOR = ROOT / "examples" / "knsb-motor" / "output"


def _run(tmp: Path) -> Path:
    """A small finished run: a plate bent by a load swept up to 200 N, standing on the ground."""
    out = tmp / "run"
    s = Scene("Plate under its load", domain="strength")
    load = np.linspace(0, 250, 11)
    s.timeline(load, unit="N", label="load", kind="input", design=200, between="exact")
    X = np.array([[x, y, 5.0] for y in (0, 20) for x in np.linspace(0, 100, 11)])
    T = np.array([[i, i + 1, i + 12] for i in range(10)] + [[i, i + 12, i + 11] for i in range(10)])
    part = s.mesh(X, T)
    part.color_by("von Mises", X[:, 0] * 0.5, scale=load / 250, unit="MPa", limit=60)
    part.deform(np.column_stack([np.zeros(22), np.zeros(22), -X[:, 0] ** 2 / 2000]), scale=load / 250)
    s.body("box", [5, 10, 5], pos=(0, 10, 0), prop=True)
    s.arrows([[100, 10, 30]], [[0, 0, -20]], scale=load / 250)
    s.ground(0, size=300, grid=20, center=(50, 10))
    s.readout("load", load, unit="N")
    s.save(out)
    (out / "outcome.json").write_text(json.dumps({"verdict": "holds", "headline": "Peak stress 40 MPa against a "
                                                  "60 MPa limit, a safety factor of 1.5. Fine."}))
    (out / "report.html").write_text("<html>report</html>")
    return out


def test_a_sweep_ramps_to_its_design_value_while_the_camera_swings(tmp_path):
    sc = Posable(_run(tmp_path) / "scene.wobbl")
    assert design_position(sc) == pytest.approx(8.0)
    plan = shots(sc, 20)
    assert len(plan) == 120 and plan[0].u == 0 and max(s.u for s in plan) == pytest.approx(8.0)
    assert plan[-1].u == pytest.approx(8.0) and max(abs(s.angle) for s in plan) > 20
    assert still(sc).u == pytest.approx(8.0)


def test_frames_are_drawn_with_a_caption(tmp_path):
    run = _run(tmp_path)
    sc = Posable(run / "scene.wobbl")
    pic = render(run / "scene.wobbl", [still(sc)], (640, 360))[0]
    assert pic.shape == (360, 640, 3)
    bg = np.array([14, 15, 17])
    assert (np.abs(pic[5, 5].astype(int) - bg) <= 3).all()            # the dark background in a corner
    assert (pic[150:250, 200:450].std(axis=(0, 1)) > 5).any()          # something drawn in the middle
    yellow = (pic[:, :, 0] > 200) & (pic[:, :, 1] > 170) & (pic[:, :, 2] < 80)
    assert yellow[250:, :200].sum() > 20                               # the caption's accent in the corner


def test_a_png_and_a_gif_land_where_asked(tmp_path):
    run = _run(tmp_path)
    done = export(run, "png", out=str(tmp_path / "pics"), name="plate")
    assert done.path == tmp_path / "pics" / "plate.png" and Image.open(done.path).size == (1920, 1080)
    again = export(run, "png", out=str(tmp_path / "pics"), name="plate")
    assert again.path.name == "plate-2.png"                             # never overwrites
    sc = Posable(run / "scene.wobbl")
    frames = render(run / "scene.wobbl", shots(sc, 4)[:6], (320, 180))
    gif = save.gif(frames, 4, tmp_path / "clip.gif")
    im = Image.open(gif)
    assert im.n_frames == 6 and im.info["loop"] == 0 and im.size == (320, 180)


def test_mp4_without_ffmpeg_falls_back_to_a_gif_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(save, "ffmpeg", lambda: None)
    import wobbl.export.clip as clip
    monkeypatch.setattr(clip, "seconds", lambda sc: 0.5)
    done = export(_run(tmp_path), "mp4", out=str(tmp_path / "v.mp4"))
    assert done.format == "gif" and done.path == tmp_path / "v.gif" and "ffmpeg" in done.note


def test_folders_named_in_words_resolve_per_system(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    if sys.platform.startswith("linux"):
        (tmp_path / "cfg").mkdir()
        (tmp_path / "cfg" / "user-dirs.dirs").write_text('XDG_DOWNLOAD_DIR="$HOME/Telechargements"\n')
        assert save.known_folder("downloads") == tmp_path / "Telechargements"
    else:
        assert save.known_folder("downloads") == tmp_path / "Downloads"
    assert save.known_folder("desktop") == tmp_path / "Desktop"
    p = save.destination("desktop", "png", "led", tmp_path / "results")
    assert p == tmp_path / "Desktop" / "led.png" and p.parent.is_dir()


def test_the_report_is_copied_out_and_a_run_without_a_scene_says_why(tmp_path):
    run = _run(tmp_path)
    done = export(run, "report", out=str(tmp_path / "docs"), name="plate")
    assert done.path.read_text() == "<html>report</html>"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExportError, match="no 3D scene"):
        export(empty, "gif")


@pytest.mark.skipif(not (MOTOR / "scene.wobbl").exists(), reason="the motor example has no scene")
def test_the_motor_example_draws_on_its_test_stand(tmp_path):
    sc = Posable(MOTOR / "scene.wobbl")
    pic = render(MOTOR / "scene.wobbl", [still(sc)], (640, 360))[0]
    orange = (pic[:, :, 0] > 200) & (pic[:, :, 1] > 90) & (pic[:, :, 1] < 190) & (pic[:, :, 2] < 90)
    assert orange.sum() > 200                                          # the burning ports show through the cutaway
