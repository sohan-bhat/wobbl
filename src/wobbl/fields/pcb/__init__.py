"""Read a real printed circuit board from KiCad or Gerber files into one board model.

read_board takes a .kicad_pcb file, a folder or .zip of Gerber files, or one
Gerber file (its sibling files are read too) and returns a Board. net_graph
turns one net into nodes and edges for an RLC network, and reference_planes
finds the pours under a point.

Conventions every reader follows
    lengths are meters, x points right and y up as seen from the top (KiCad's y is flipped),
    z is measured up from the bottom face, and copper layers carry KiCad's names F.Cu, In1.Cu ... B.Cu
    arcs are split into straight tracks whose chords stay within 1% of the radius
    zones hold the filled copper (KiCad filled polygons, Gerber regions), not the zone outline
    pad angles are counter clockwise as seen from the top and pad sizes are before that turn
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from .gerber import looks_like_excellon, looks_like_gerber, read_gerber_texts
from .graph import Edge, NetGraph, infer_nets, net_graph, reference_planes
from .kicad import parse_sexpr, read_kicad_pcb
from .model import Board, Dielectric, Items, Layer, Pad, Stackup, Track, Via, Zone, build_stackup, default_stackup

__all__ = ["Board", "Dielectric", "Edge", "Items", "Layer", "NetGraph", "Pad", "Stackup", "Track", "Via", "Zone",
           "build_stackup", "default_stackup", "infer_nets", "net_graph", "parse_sexpr", "read_board",
           "read_gerbers", "read_kicad_pcb", "reference_planes"]

_MAX_BYTES = 400 * 1024 * 1024
_SKIP_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".step", ".stp", ".wrl", ".kicad_sch", ".kicad_pro", ".kicad_prl",
                  ".zip", ".csv", ".pos", ".rpt", ".html", ".svg", ".dxf", ".md", ".kicad_pcb"}


def _wanted(name: str, head: str) -> bool:
    return name.lower().endswith(".gbrjob") or looks_like_gerber(head) or looks_like_excellon(head)


def _folder_texts(folder: Path) -> dict[str, str]:
    texts, total = {}, 0
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() in _SKIP_SUFFIXES or f.name.startswith("."):
            continue
        total += f.stat().st_size
        if total > _MAX_BYTES:
            raise ValueError(f"The files in {folder} add up to more than 400 MB, which is too much for a board. "
                             "Point at the folder that holds only the Gerber and drill files.")
        with f.open("rb") as fh:
            head = fh.read(4096).decode("utf-8", errors="replace")
        if _wanted(f.name, head):
            texts[f.name] = f.read_bytes().decode("utf-8", errors="replace")
    return texts


def _zip_texts(path: Path) -> dict[str, str]:
    texts, total = {}, 0
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{path.name} is not a readable zip file. Unzip it yourself and pass the folder.") from None
    with zf:
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() in _SKIP_SUFFIXES:
                continue
            total += info.file_size
            if total > _MAX_BYTES:
                raise ValueError(f"{path.name} unpacks to more than 400 MB, which is too much for a board. Zip "
                                 "only the Gerber and drill files.")
            data = zf.read(info)
            name = info.filename.replace("\\", "/").split("/")[-1]
            if _wanted(name, data[:4096].decode("utf-8", errors="replace")):
                texts[name if name not in texts else info.filename] = data.decode("utf-8", errors="replace")
    return texts


def read_gerbers(files, name: str | None = None) -> Board:
    """Read Gerber (plus Excellon and .gbrjob) files given as a folder, a zip, or a list of paths."""
    if isinstance(files, (str, Path)):
        p = Path(files)
        if p.is_dir():
            return read_gerber_texts(_folder_texts(p), name or p.name)
        if p.suffix.lower() == ".zip":
            return read_gerber_texts(_zip_texts(p), name or p.stem)
        files = [p]
    texts = {}
    paths = [Path(f) for f in files]
    for f in paths:
        try:
            texts[f.name] = f.read_bytes().decode("utf-8", errors="replace")
        except OSError as e:
            raise ValueError(f"Could not open {f} ({e.strerror}). Check the path.") from None
    stem = paths[0].stem.split("-")[0] if paths else "board"
    return read_gerber_texts(texts, name or stem)


def read_board(path) -> Board:
    """Read a board from a .kicad_pcb file, a folder or zip of Gerbers, or one Gerber file and its siblings."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"{p} does not exist. Pass a .kicad_pcb file, a folder or zip of Gerber files, or one "
                         "Gerber file.")
    if p.is_dir():
        texts = _folder_texts(p)
        if any(looks_like_gerber(t[:4096]) for t in texts.values()):
            return read_gerber_texts(texts, p.name)
        boards = sorted(p.glob("*.kicad_pcb"))
        if len(boards) == 1:
            return read_kicad_pcb(boards[0])
        what = "several .kicad_pcb files" if boards else "no Gerber or .kicad_pcb files"
        raise ValueError(f"{p} holds {what}. Pass the .kicad_pcb file itself or a folder of Gerber files.")
    suffix = p.suffix.lower()
    if suffix == ".kicad_pcb":
        return read_kicad_pcb(p)
    if suffix == ".kicad_pro":
        return read_kicad_pcb(p.with_suffix(".kicad_pcb"))
    if suffix == ".zip":
        return read_gerbers(p)
    with p.open("rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
    if head.lstrip().startswith("(kicad_pcb"):
        return read_kicad_pcb(p)
    if _wanted(p.name, head):
        return read_gerber_texts(_folder_texts(p.parent), p.stem.split("-")[0])
    raise ValueError(f"{p.name} is neither a KiCad board nor a Gerber file. Pass a .kicad_pcb file, a folder or zip "
                     "of Gerber files, or one Gerber file.")
