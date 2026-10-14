"""Turn whatever the user brings into a Circuit: a SPICE file or text, a KiCad schematic or netlist export,
or a plain description of the circuit."""
from __future__ import annotations

import re
from pathlib import Path

from .netlist import Circuit, NetlistError, parse_spice

SPICE_SUFFIX = (".cir", ".sp", ".spice", ".net", ".ckt", ".txt", ".lib")


def kind_of(source: str) -> str:
    """netlist, kicad, kicad-netlist or description, from a path or from the text itself."""
    s = str(source).strip()
    p = _path(s)
    if p is not None:
        head = p.read_text(encoding="utf-8", errors="replace")[:400].lstrip()
        if p.suffix.lower() == ".kicad_sch" or head.startswith("(kicad_sch"):
            return "kicad"
        if head.startswith("(export"):
            return "kicad-netlist"
        return "netlist"
    if s.startswith("(kicad_sch"):
        return "kicad"
    if s.startswith("(export"):
        return "kicad-netlist"
    lines = [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("*")]
    spice = [ln for ln in lines if re.match(r"^([rclvidqmxjbegfhswk]\w*\s+\S+\s+\S+|\.\w+)", ln, re.I)]
    if len(lines) >= 2 and len(spice) >= 0.6 * len(lines):
        return "netlist"
    return "description"


def _path(s: str) -> Path | None:
    if not s or "\n" in s or len(s) > 1024:
        return None
    try:
        p = Path(s).expanduser()
        return p if p.is_file() else None
    except OSError:
        return None


def load(source: str, params: dict, assume=None) -> Circuit:
    kind = kind_of(source)
    p = _path(str(source).strip())
    if kind == "netlist":
        text = p.read_text(encoding="utf-8", errors="replace") if p else str(source)
        c = parse_spice(text, p.parent if p else None)
        c.source = "netlist"
        if not c.elements:
            raise NetlistError("the netlist has no parts", "A netlist line looks like 'R1 in out 10k'")
        return c
    if kind in ("kicad", "kicad-netlist"):
        from . import kicad
        text = p.read_text(encoding="utf-8", errors="replace") if p else str(source)
        c = kicad.parse_schematic(text, p, params) if kind == "kicad" else kicad.parse_netlist_export(text, params)
        c.source = "kicad"
        return c
    from . import describe
    c = describe.build(str(source), params, assume)
    c.source = "description"
    return c
