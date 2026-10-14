"""MuJoCo's install on first use: pruning the wheel and the registry hook, all offline."""
import pytest

from wobbl.mechanism import install
from wobbl.setup import installer
from wobbl.solvers.base import SolverNotFound


def test_prune_keeps_what_a_headless_simulation_loads(tmp_path):
    root = tmp_path / "w"
    for rel in ["mujoco/__init__.py", "mujoco/libmujoco.3.10.0.dylib", "mujoco/_structs.cpython-313-darwin.so",
                "mujoco/_functions.cpython-313-darwin.so", "mujoco/_specs.cpython-313-darwin.so",
                "mujoco/_enums.cpython-313-darwin.so", "mujoco/_callbacks.cpython-313-darwin.so",
                "mujoco/_constants.cpython-313-darwin.so", "mujoco/_errors.cpython-313-darwin.so",
                "mujoco/mujoco.dll", "mujoco/mujoco.lib", "mujoco/_simulate.cpython-313-darwin.so",
                "mujoco/_render.cpython-313-darwin.so", "mujoco/_rollout.cpython-313-darwin.so", "mujoco/viewer.py",
                "mujoco/rollout.py", "mujoco/bindings_test.py", "mujoco/include/mujoco/mujoco.h",
                "mujoco/plugin/libelasticity.dylib", "mujoco/sysid/a.py", "mujoco/usd/b.py",
                "mujoco/MuJoCo_(mjpython).app/Contents/Info.plist", "mujoco/rendering/classic/gl_context.py",
                "mujoco/__pycache__/x.pyc", "mujoco.libs/libGL.so"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x" * 10)
    removed = install.prune(root)
    left = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assert left == ["mujoco.libs/libGL.so", "mujoco/__init__.py", "mujoco/_callbacks.cpython-313-darwin.so",
                    "mujoco/_constants.cpython-313-darwin.so", "mujoco/_enums.cpython-313-darwin.so",
                    "mujoco/_errors.cpython-313-darwin.so", "mujoco/_functions.cpython-313-darwin.so",
                    "mujoco/_specs.cpython-313-darwin.so", "mujoco/_structs.cpython-313-darwin.so",
                    "mujoco/libmujoco.3.10.0.dylib", "mujoco/mujoco.dll"]
    assert removed == 140
    assert not (root / "mujoco" / "include").exists()           # emptied folders go too
    assert install.unused("mujoco-3.10.0.dist-info/RECORD") and not install.unused("mujoco.libs/libGL.so")


def test_mujoco_is_an_on_demand_recipe_with_its_own_installer(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "wobbl"))
    assert [r.name for r in installer.recipes_for("mechanism")] == ["mujoco"]
    assert not installer.RECIPES["mujoco"].core
    p = installer.plan("osx-64", ["mujoco"])
    assert p["custom"] == ["mujoco"] and p["groups"] == []
    assert installer.plan("linux-64", ["mechanism"])["custom"] == ["mujoco"]

    calls = []
    present = {"ok": False}
    monkeypatch.setattr(install, "available", lambda: present["ok"])
    monkeypatch.setattr(install, "load", lambda: None)

    def fake_install(log=print, force=False):
        calls.append(force)
        present["ok"] = True
        return {"installed_bytes": 1}

    monkeypatch.setattr(install, "install", fake_install)
    monkeypatch.setattr(installer, "verify", lambda log: {"solvers": {}, "unavailable": {}})
    monkeypatch.setenv("WOBBL_AUTO_INSTALL", "0")
    with pytest.raises(SolverNotFound, match="not installed"):
        install.require()
    assert not calls
    monkeypatch.setenv("WOBBL_AUTO_INSTALL", "1")
    logs = []
    assert installer.install_on_demand(install.DOMAINS, logs.append) is True
    assert calls == [False] and any("mujoco" in m for m in logs)
    assert installer.install_on_demand(install.DOMAINS, logs.append) is False      # already there


@pytest.mark.skipif(not install.available(), reason="MuJoCo is not installed")
def test_the_registry_check_swings_a_pendulum():
    r = install.check(lambda m: None)
    assert isinstance(r, dict) and r["works"], r
    assert install.WARNINGS == [] or all(isinstance(w, str) for w in install.WARNINGS)
