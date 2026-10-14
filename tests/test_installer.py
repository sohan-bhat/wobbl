"""The install registry and installs on first use, all inside a scratch WOBBL_HOME."""
import json
import sys

import pytest

from wobbl.setup import installer
from wobbl.solvers import base


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "wobbl"))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(installer, "platform_tag", lambda: "linux-64")
    return tmp_path / "wobbl"


def test_registry_maps_domains_to_recipes():
    assert [r.name for r in installer.recipes_for("structural")] == ["calculix"]
    assert [r.name for r in installer.recipes_for("thermal")] == ["calculix"]
    assert [r.name for r in installer.recipes_for("circuit")] == ["ngspice"]
    assert installer.RECIPES["calculix"].package_names() == ["calculix"]
    assert installer.target(installer.RECIPES["ngspice"], "osx-arm64") == ("env-x86", "osx-64")
    assert installer.target(installer.RECIPES["ngspice"], "linux-aarch64") is None


def test_a_new_recipe_installs_on_first_use_not_in_setup(home, monkeypatch):
    calls = []

    def fake_mm(mm, args, log, logfile):   # stands in for micromamba: records the call, writes conda-meta
        calls.append(args)
        env = home / "env"
        (env / "conda-meta").mkdir(parents=True, exist_ok=True)
        (env / "bin").mkdir(exist_ok=True)
        (env / "bin" / "elmerf90").write_text("x")
        (env / "bin" / "ElmerSolver").write_text("x")
        (env / "conda-meta" / "elmer-9.0-h_0.json").write_text("{}")

    monkeypatch.setattr(installer, "_mm", fake_mm)
    monkeypatch.setattr(installer, "ensure_micromamba", lambda log, force=False: home / "bin" / "micromamba")
    monkeypatch.setattr(installer, "RECIPES", dict(installer.RECIPES))
    installer.register(installer.Recipe("elmer", ("multiphysics",), ("elmer>=9",), keep=("ElmerSolver*",),
                                        check=lambda log: {"path": "x", "version": "9", "works": True, "error": ""}))
    assert "elmer>=9" not in installer.plan("linux-64")["env"]          # setup installs only the core recipes
    assert installer.plan("linux-64", ["multiphysics"])["env"] == ["elmer>=9"]

    monkeypatch.setenv("WOBBL_AUTO_INSTALL", "0")
    assert installer.install_on_demand(("multiphysics",)) is False and not calls
    monkeypatch.setenv("WOBBL_AUTO_INSTALL", "1")
    logs = []
    assert installer.install_on_demand(("multiphysics",), logs.append) is True
    assert len(calls) == 1 and calls[0][0] == "create" and calls[0][-1] == "elmer>=9"
    assert installer.installed(installer.RECIPES["elmer"])
    assert (home / "env" / "bin" / "ElmerSolver").exists()
    assert json.loads((home / "config.json").read_text())["solvers"]["elmer"]["works"]
    assert installer.install_on_demand(("multiphysics",), logs.append) is False   # already there
    assert len(calls) == 1


def test_require_installs_a_missing_solver_on_first_use(home, monkeypatch):
    class Fake(base.SolverAdapter):
        name, domains = "fake", ("fakedomain",)
        path = None

        def locate(self):
            return self.path

    a = Fake()

    def on_demand(domains, log=None):
        a.path = home / "bin" / "fake"
        return True

    monkeypatch.setattr(installer, "install_on_demand", on_demand)
    assert a.require() == home / "bin" / "fake"

    def fails(domains, log=None):
        raise installer.SetupError("no network")

    a.path = None
    monkeypatch.setattr(installer, "install_on_demand", fails)
    with pytest.raises(base.SolverNotFound, match="no network"):
        a.require()
