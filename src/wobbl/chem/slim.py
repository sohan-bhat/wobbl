"""Make installed chemistry libraries smaller without changing what they compute.

Three steps, each checked by running the library on a small problem afterwards and undone if that fails:

  native pruning   run the calls Wobbl makes in a separate Python, note which of the package's shared libraries the
                   operating system loaded, and delete the ones it never touched (RDKit's drawing, standardizing and
                   search libraries, a third of its size)
  Accelerate       on macOS, point tblite at Apple's Accelerate framework for BLAS and LAPACK instead of the
                   62 MB OpenBLAS its wheel bundles, by rewriting one load command in its Mach-O file
  basis sets       keep only the Gaussian basis sets Wobbl's DFT uses (def2-SVP, def2-TZVP and their fitting set)
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"

# what each package's probe runs, the calls Wobbl itself makes
PROBES = {
    "rdkit": """
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
AllChem.EmbedMultipleConfs(m, 2, randomSeed=1)
AllChem.MMFFOptimizeMoleculeConfs(m)
AllChem.UFFOptimizeMoleculeConfs(m)
Chem.MolToSmiles(Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles("O"))))
Chem.MolFromMolBlock(Chem.MolToMolBlock(m))
""",
}
KEEP_FILES = {"pyscf": ("pyscf/gto/basis", ("def2-svp.dat", "def2-tzvp.dat", "def2-universal-jkfit.dat",
                                             "def2-svp-jkfit.dat", "sto-3g.dat", "sto3g.dat", "ano.dat", "*.py", "*.json"))}
DROP_DIRS = {"pyscf": ("pyscf/gto/basis/dyall-basis", "pyscf/gto/basis/ccecp-basis", "pyscf/gto/basis/f12-basis",
                       "pyscf/gto/basis/soecp")}

_LOADED = """
import sys, os, json
sys.path.insert(0, {root!r})
exec({probe!r})
paths = set()
if sys.platform == "darwin":
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib._dyld_image_count.restype = ctypes.c_uint32
    lib._dyld_get_image_name.restype = ctypes.c_char_p
    paths = {{os.path.realpath(lib._dyld_get_image_name(i).decode()) for i in range(lib._dyld_image_count())}}
elif sys.platform.startswith("linux"):
    for line in open("/proc/self/maps"):
        parts = line.split()
        if len(parts) >= 6 and parts[5].startswith("/"):
            paths.add(os.path.realpath(parts[5]))
print(json.dumps(sorted(paths)))
"""


def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def prune_native(name: str, root: Path) -> int:
    """Delete the package's shared libraries Wobbl's calls never load. Returns the bytes removed, 0 when skipped."""
    probe = PROBES.get(name)
    if not probe or sys.platform not in ("darwin", "linux"):
        return 0
    try:
        r = subprocess.run([sys.executable, "-c", _LOADED.format(root=str(root), probe=probe)], capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return 0
    if r.returncode != 0:
        return 0
    loaded = set(json.loads(r.stdout.strip().splitlines()[-1]))
    pkg = root / name
    victims = []
    for p in pkg.rglob("*"):
        if p.is_file() and (p.suffix in (".so", ".dylib") or ".so." in p.name):
            if os.path.realpath(p) not in loaded:
                victims.append(p)
    if not victims:
        return 0
    hold = root.with_name(root.name + ".native")
    shutil.rmtree(hold, ignore_errors=True)
    removed = 0
    for p in victims:
        dest = hold / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        removed += _size(p)
        shutil.move(str(p), str(dest))
    try:
        ok = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(root)!r})\n{probe}"],
                            capture_output=True, timeout=600).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:             # something needed was moved: put it all back
        for p in victims:
            shutil.move(str(hold / p.relative_to(root)), str(p))
        removed = 0
    shutil.rmtree(hold, ignore_errors=True)
    return removed


def prune_files(name: str, root: Path) -> int:
    removed = 0
    for rel in DROP_DIRS.get(name, ()):
        d = root / rel
        if d.is_dir():
            removed += sum(_size(p) for p in d.rglob("*") if p.is_file())
            shutil.rmtree(d, ignore_errors=True)
    if name in KEEP_FILES:
        folder, keep = KEEP_FILES[name]
        d = root / folder
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and not any(fnmatch.fnmatch(p.name, k) for k in keep):
                    removed += _size(p)
                    p.unlink()
    return removed


# ------------------------------------------------------------------------------------------- Mach-O
LC_LOAD_DYLIB, LC_SEGMENT_64 = 0xC, 0x19
MH_MAGIC_64 = 0xFEEDFACF


def relink_dylib(path: Path, old_prefix: str, new: str) -> bool:
    """Replace the load command for a library whose path starts with old_prefix by one for new, growing the load
    commands into the header padding. False when the file is not a thin 64 bit Mach-O or has no room."""
    data = bytearray(path.read_bytes())
    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != MH_MAGIC_64:
        return False
    ncmds, sizeofcmds = struct.unpack_from("<II", data, 16)
    off = 32
    cmds = []
    first_section = len(data)
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        cmds.append((cmd, bytes(data[off:off + size])))
        if cmd == LC_SEGMENT_64:
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            fileoff, filesize = struct.unpack_from("<QQ", data, off + 40)
            for s in range(nsects):
                so = off + 72 + s * 80
                sec_off = struct.unpack_from("<I", data, so + 48)[0]
                if sec_off:
                    first_section = min(first_section, sec_off)
        off += size
    new_cmds, changed = [], False
    for cmd, raw in cmds:
        if cmd == LC_LOAD_DYLIB:
            name_off = struct.unpack_from("<I", raw, 8)[0]
            name = raw[name_off:].split(b"\0", 1)[0].decode()
            if name.startswith(old_prefix):
                ts, cur, compat = struct.unpack_from("<III", raw, 12)
                s = new.encode() + b"\0"
                size = (24 + len(s) + 7) // 8 * 8
                pad = b"\0" * (size - 24 - len(s))
                raw = struct.pack("<IIIIII", LC_LOAD_DYLIB, size, 24, ts, cur, compat) + s + pad
                changed = True
        new_cmds.append(raw)
    if not changed:
        return False
    blob = b"".join(new_cmds)
    if 32 + len(blob) > first_section:
        return False
    data[20:24] = struct.pack("<I", len(blob))
    end = 32 + max(len(blob), sizeofcmds)
    data[32:end] = blob + b"\0" * (end - 32 - len(blob))
    path.write_bytes(bytes(data))
    return True


def accelerate_tblite(root: Path) -> int:
    """Use Accelerate instead of tblite's bundled OpenBLAS on macOS. Returns the bytes removed, 0 when skipped."""
    if sys.platform != "darwin" or not Path(ACCELERATE).parent.exists():
        return 0
    pkg = root / "tblite"
    dylibs = pkg / ".dylibs"
    blas = sorted(dylibs.glob("libopenblas*.dylib")) if dylibs.is_dir() else []
    exts = sorted(pkg.glob("_libtblite*.so"))
    if not blas or not exts:
        return 0
    ext = exts[0]
    backup = ext.read_bytes()
    if not relink_dylib(ext, "@loader_path/.dylibs/libopenblas", ACCELERATE):
        return 0
    if os.uname().machine == "arm64" and shutil.which("codesign"):
        subprocess.run(["codesign", "-s", "-", "-f", str(ext)], capture_output=True)
    (pkg / "wobbl-accelerate.json").write_text(json.dumps({"gfortran": ".dylibs/libgfortran.5.dylib"}))
    check = f"""
import sys, ctypes
sys.path.insert(0, {str(root)!r})
ctypes.CDLL({str(dylibs / 'libgfortran.5.dylib')!r}, mode=ctypes.RTLD_GLOBAL)
import numpy as np
from tblite.interface import Calculator
c = Calculator("GFN2-xTB", np.array([8, 1, 1]), np.array([[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]))
c.set("verbosity", 0)
print(repr(c.singlepoint().get("energy")))
"""
    try:
        r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, timeout=600)
        e = float(r.stdout.strip().splitlines()[-1])
        ok = -5.2 < e < -5.0
    except (ValueError, IndexError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        ext.write_bytes(backup)
        (pkg / "wobbl-accelerate.json").unlink(missing_ok=True)
        return 0
    freed = 0
    for b in blas:
        freed += _size(b)
        b.unlink()
    return freed


def preload(name: str, root: Path) -> None:
    """Load what a slimmed package needs before it is imported: tblite on Accelerate needs the Fortran runtime its
    OpenBLAS used to bring."""
    if name != "tblite":
        return
    marker = root / "tblite" / "wobbl-accelerate.json"
    if marker.exists():
        import ctypes
        lib = root / "tblite" / json.loads(marker.read_text())["gfortran"]
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
