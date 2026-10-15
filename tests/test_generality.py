"""The drop test and the thin wall detection on parts nothing was tuned for, and on every example part."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import _write

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(p for p in ROOT.glob("examples/*/*.py") if "def build" in p.read_text())


def _sphere_shell(o):
    a = o.addSphere(0, 0, 0, 30)
    b = o.addSphere(0, 0, 0, 28)
    o.cut([(3, a)], [(3, b)])


def _ibeam(o):
    f1 = o.addBox(0, -15, 0, 120, 30, 3)
    f2 = o.addBox(0, -15, 37, 120, 30, 3)
    w = o.addBox(0, -1.5, 3, 120, 3, 34)
    o.fuse([(3, f1)], [(3, f2), (3, w)])


def _torus(o):
    o.addTorus(0, 0, 0, 30, 9)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_thin_wall_detection_on_every_example(path, tmp_path):
    """Every example part reads and gets a verdict on whether its walls are thin, with the numbers behind it."""
    from wobbl.shells.mesh import thin_walled
    from wobbl.structural.run import describe_part

    from wobbl.geometry.cad import GeometryError

    try:
        _, feats = describe_part(str(path), tmp_path / "work")
    except GeometryError as e:
        # an assembly of separate bodies, such as the solenoid actuator, is a field example rather than one part
        if "separate bodies" in str(e) or "do not touch" in str(e):
            pytest.skip(f"{path.parent.name} is an assembly of separate bodies, not one part")
        raise
    thin, why = thin_walled(feats)
    assert why and ("walls about" in why)


@pytest.mark.slow
@pytest.mark.parametrize("build,thin", [(_sphere_shell, True), (_ibeam, True), (_torus, False)],
                         ids=["sphere shell", "i beam", "torus"])
def test_drop_on_parts_nothing_was_tuned_for(build, thin, tmp_path):
    """A hollow ball, an I beam and a solid ring each drop onto concrete, worst landing found, with momentum and
    energy in balance."""
    from wobbl.drop.study import DropStudy, run_drop
    from wobbl.shells.mesh import thin_walled
    from wobbl.structural.run import describe_part

    part = _write(tmp_path / "part.step", build)
    pre = describe_part(str(part), tmp_path / "work")
    assert thin_walled(pre[1])[0] == thin
    o = run_drop(DropStudy(str(part), "pc", "1 m", "worst", "concrete", quality="fast"), tmp_path / "out",
                 render=False, pre=pre)
    checks = {c.name: c.status for c in o.checks}
    assert checks["momentum"] == "pass" and checks["energy"] in ("pass", "warn")
    assert np.isfinite(o.values["governing_MPa"]) and o.values["governing_MPa"] > 0
    assert (tmp_path / "out" / "playback.json").exists()
