"""Remote machines: the transports, the ssh and copy commands, and the registry that never holds a credential."""
from __future__ import annotations

import json
import sys

import pytest

from wobbl.jobs import remote

posix = pytest.mark.skipif(sys.platform == "win32", reason="the remote side is a POSIX shell")


@posix
def test_local_transport_behaves_like_a_remote_root(tmp_path):
    t = remote.LocalTransport(tmp_path / "root")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "deck.inp").write_text("*NODE\n")
    t.put(tmp_path / "src", "job1/work")
    assert t.check("ls job1/work").split() == ["deck.inp"]
    t.check("echo 3 > job1/exit_code")
    assert t.read("job1/exit_code").strip() == "3"
    assert t.read("job1/missing") == ""
    t.check("printf 'abcdefgh' > job1/log")
    assert t.read("job1/log", tail=3) == "fgh"
    t.get("job1/log", tmp_path / "back" / "log")
    assert (tmp_path / "back" / "log").read_text() == "abcdefgh"
    with pytest.raises(remote.RemoteError):
        t.check("exit 4")


def test_ssh_commands_never_prompt_and_stay_in_the_root():
    t = remote.SSHTransport("me@box", "~/wobbl-jobs", port=2222)
    argv = t.ssh_argv("ls")
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and argv[argv.index("-p") + 1] == "2222"
    assert argv[-2] == "me@box" and argv[-1] == "mkdir -p wobbl-jobs && cd wobbl-jobs && ls"
    cp = t.copy_argv("local/dir", "me@box:wobbl-jobs/j1", True)
    assert cp[0] in ("rsync", "scp") and any("BatchMode=yes" in a for a in cp)
    assert not any("password" in a.lower() for a in argv + cp)


@posix
def test_registry_probes_and_keeps_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("WOBBL_HOME", str(tmp_path / "home"))
    entry = remote.add("box", f"local:{tmp_path / 'far'}")
    assert entry["cores"] >= 1 and entry["ram"] > 1e8 and entry["free"] > 0
    m = remote.machines()["box"]
    assert m.name == "box" and m.cores == entry["cores"] and m.ram == entry["ram"]
    saved = json.loads((tmp_path / "home" / "remotes.json").read_text())
    assert set(saved["box"]) <= {"host", "root", "port", "os", "cores", "ram", "free", "ccx", "python", "wobbl"}
    with pytest.raises(remote.RemoteError, match="wobbl remote add"):
        remote.get("other")
    assert remote.remove("box") and not remote.remove("box")
    with pytest.raises(remote.RemoteError):
        remote.add("bad name!", "local:/tmp")


@posix
def test_a_result_folder_comes_back_without_its_work_files(tmp_path):
    class TarOverShell(remote.LocalTransport):
        """Runs the same tar stream an SSH remote uses, in a local folder."""
        get_dir = remote.Transport.get_dir

        def argv(self, command):
            return ["sh", "-c", f"cd {self.root} && {command}"]

    for t in (remote.LocalTransport(tmp_path / "far"), TarOverShell(tmp_path / "far")):
        t.check("mkdir -p j/result/work j/result/images && echo 1 > j/result/outcome.json && "
                "echo big > j/result/work/static.frd && echo png > j/result/images/a.png")
        dest = tmp_path / f"back_{type(t).__name__}"
        t.get_dir("j/result", dest, exclude=("work",))
        assert (dest / "outcome.json").read_text() == "1\n" and (dest / "images" / "a.png").exists()
        assert not (dest / "work").exists()
        with pytest.raises(remote.RemoteError):
            t.get_dir("j/none", tmp_path / "x")
