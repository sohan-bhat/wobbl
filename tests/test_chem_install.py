"""Cantera's install on first use: picking the wheel, pruning it and the registry hook, all offline."""
import pytest

from wobbl.chem import install
from wobbl.setup import installer
from wobbl.solvers.base import SolverNotFound

TAGS = ["cp312-cp312-macosx_11_0_arm64", "cp312-cp312-macosx_10_15_x86_64", "cp312-abi3-macosx_10_15_x86_64",
        "py3-none-any"]


def _file(name):
    return {"filename": name, "packagetype": "bdist_wheel", "url": "https://x/" + name, "size": 1}


def test_pick_wheel_follows_the_tag_order():
    files = [_file("cantera-3.2.0-cp311-cp311-macosx_10_15_x86_64.whl"),
             _file("cantera-3.2.0-cp312-cp312-macosx_10_15_x86_64.whl"),
             _file("cantera-3.2.0-cp312-cp312-win_amd64.whl"),
             {"filename": "cantera-3.2.0.tar.gz", "packagetype": "sdist"}]
    assert install.pick_wheel(files, TAGS)["filename"] == "cantera-3.2.0-cp312-cp312-macosx_10_15_x86_64.whl"
    files.append(_file("cantera-3.2.0-cp312-cp312-macosx_11_0_arm64.whl"))
    assert install.pick_wheel(files, TAGS)["filename"].endswith("arm64.whl")
    assert install.pick_wheel([_file("cantera-3.2.0-cp39-cp39-win32.whl")], TAGS) is None


def test_prune_keeps_the_library_and_the_three_data_files(tmp_path):
    root = tmp_path / "w"
    for rel in ["cantera/__init__.py", "cantera/_cantera.so", "cantera/composite.py", "cantera/composite.pyi",
                "cantera/ck2yaml.py", "cantera/py.typed", "cantera/test/test_x.py", "cantera/examples/a.py",
                "cantera/data/gri30.yaml", "cantera/data/nasa_gas.yaml", "cantera/data/nasa_condensed.yaml",
                "cantera/data/h2o2.yaml", "cantera/data/example_data/big.yaml", "cantera-3.2.0.data/scripts/ck2yaml",
                "cantera-3.2.0.dist-info/METADATA", "cantera.libs/libhdf5.so"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x" * 10)
    removed = install.prune(root)
    left = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    assert left == ["cantera-3.2.0.dist-info/METADATA", "cantera.libs/libhdf5.so", "cantera/__init__.py",
                    "cantera/_cantera.so", "cantera/composite.py", "cantera/data/gri30.yaml",
                    "cantera/data/nasa_condensed.yaml", "cantera/data/nasa_gas.yaml"]
    assert removed == 80


def test_cantera_is_an_on_demand_recipe_with_its_own_installer(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "wobbl"))
    assert [r.name for r in installer.recipes_for("combustion")] == ["cantera"]
    assert [r.name for r in installer.recipes_for("motor")] == ["cantera"]
    assert not installer.RECIPES["cantera"].core
    p = installer.plan("linux-64", ["cantera"])
    assert p["custom"] == ["cantera"] and p["env"] == [] and p["groups"] == []

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
    assert calls == [False] and any("cantera" in m for m in logs)
    assert installer.install_on_demand(install.DOMAINS, logs.append) is False      # already there
