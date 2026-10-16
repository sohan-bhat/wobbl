"""The install registry, pruning, cache clearing and installs on first use, all inside a scratch WOBBL_HOME."""
import json
import sys

import pytest

from wobbl.setup import installer
from wobbl.solvers import base


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "wobbl"))
    monkeypatch.delenv("WOBBL_KEEP_CACHE", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(installer, "platform_tag", lambda: "linux-64")
    return tmp_path / "wobbl"


def _fake_env(home, packages=("calculix-2.23-h1_0", "ngspice-41-h2_0", "gettext-tools-0.25.1-h3_2")):
    env = home / "env"
    for rel in ("bin/ccx_2.23", "bin/ngspice", "bin/xgettext", "bin/tic", "include/curses.h", "share/doc/a/README",
                "share/locale/de/x.mo", "share/terminfo/x/xterm", "lib/libc++.a", "lib/libopenblas.dylib",
                "lib/libgettextlib-0.25.1.dylib", "lib/pkgconfig/x.pc", "lib/ngspice/analog.cm"):
        (env / rel).parent.mkdir(parents=True, exist_ok=True)
        (env / rel).write_bytes(b"x" * 1000)
    (env / "conda-meta").mkdir()
    for p in packages:
        (env / "conda-meta" / f"{p}.json").write_text("{}")
    return env


def _works(*names):
    return {"solvers": {n: {"works": True} for n in names}, "unavailable": {}}


def test_registry_maps_domains_to_recipes():
    assert [r.name for r in installer.recipes_for("structural")] == ["calculix"]
    assert [r.name for r in installer.recipes_for("thermal")] == ["calculix"]
    assert [r.name for r in installer.recipes_for("circuit")] == ["ngspice"]
    assert installer.RECIPES["calculix"].package_names() == ["calculix", "libblas"]
    assert installer.target(installer.RECIPES["ngspice"], "osx-arm64") == ("env-x86", "osx-64")
    assert installer.target(installer.RECIPES["ngspice"], "linux-aarch64") is None


def test_a_new_recipe_installs_on_first_use_not_in_setup(home, monkeypatch):
    calls = []

    def fake_mm(mm, args, what, prefix, pkgs=None):   # stands in for micromamba: records the call, writes conda-meta
        calls.append(args)
        env = home / "env"
        (env / "conda-meta").mkdir(parents=True, exist_ok=True)
        (env / "bin").mkdir(exist_ok=True)
        (env / "bin" / "elmerf90").write_text("x")
        (env / "bin" / "ElmerSolver").write_text("x")
        (env / "conda-meta" / "elmer-9.0-h_0.json").write_text("{}")

    monkeypatch.setattr(installer, "run_micromamba", fake_mm)
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
    assert not (home / "env" / "bin" / "elmerf90").exists()             # an executable no recipe keeps
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


def test_prune_keeps_what_solvers_run(home, monkeypatch):
    env = _fake_env(home)
    monkeypatch.setattr(installer, "verify", lambda log=print: _works("calculix", "ngspice"))
    before = installer.disk_usage(env)
    installer.prune(_works("calculix", "ngspice"), lambda m: None)
    for kept in ("bin/ccx_2.23", "bin/ngspice", "lib/libopenblas.dylib", "lib/ngspice/analog.cm",
                 "share/terminfo/x/xterm", "conda-meta"):
        assert (env / kept).exists(), kept
    for gone in ("bin/xgettext", "bin/tic", "include", "share/doc", "share/locale", "lib/libc++.a",
                 "lib/pkgconfig", "lib/libgettextlib-0.25.1.dylib"):
        assert not (env / gone).exists(), gone
    assert installer.disk_usage(env) < before / 2
    assert not (home / "pruned").exists()


def test_prune_puts_everything_back_when_a_solver_breaks(home, monkeypatch):
    env = _fake_env(home)
    files = sorted(p.relative_to(env) for p in env.rglob("*"))
    monkeypatch.setattr(installer, "verify", lambda log=print: {"solvers": {"calculix": {"works": True},
                                                                             "ngspice": {"works": False}},
                                                                 "unavailable": {}})
    logs = []
    out = installer.prune(_works("calculix", "ngspice"), logs.append)
    assert out["solvers"]["ngspice"]["works"]
    assert sorted(p.relative_to(env) for p in env.rglob("*")) == files
    assert "ngspice" in logs[0]


def test_clean_cache_frees_downloads_but_keeps_the_env(home, monkeypatch):
    env = _fake_env(home)
    pkgs = home / "mamba" / "pkgs"
    (pkgs / "cache").mkdir(parents=True)
    (pkgs / "cache" / "repodata.json").write_bytes(b"r" * 5000)
    (pkgs / "calculix-2.23").mkdir()
    (pkgs / "calculix-2.23" / "ccx").hardlink_to(env / "bin" / "ccx_2.23")   # how micromamba links packages in
    monkeypatch.setenv("WOBBL_KEEP_CACHE", "1")
    assert installer.clean_cache(lambda m: None) == 0 and pkgs.exists()
    monkeypatch.delenv("WOBBL_KEEP_CACHE")
    freed = installer.clean_cache(lambda m: None)
    assert freed == 5000                       # the hard linked solver file stays in the env, so it is not freed
    assert not pkgs.exists() and (env / "bin" / "ccx_2.23").read_bytes() == b"x" * 1000


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin rights on Windows")
def test_clean_cache_keeps_a_cache_the_env_links_into(home):
    env = _fake_env(home)
    pkgs = home / "mamba" / "pkgs" / "calculix-2.23"
    pkgs.mkdir(parents=True)
    (pkgs / "ccx").write_text("x")
    (env / "bin" / "ccx_link").symlink_to(pkgs / "ccx")
    assert installer.clean_cache(lambda m: None) == 0
    assert pkgs.exists()


def test_disk_usage_counts_hard_links_once(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 4000)
    (tmp_path / "b").hardlink_to(tmp_path / "a")
    assert installer.disk_usage(tmp_path) == 4000
    assert installer.disk_usage(tmp_path, shared=False) == 0
