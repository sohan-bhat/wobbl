"""Read big solver decks without loading them: model size, steps and times in one streaming pass.

A 1.5 million element deck is a few hundred megabytes of text. To predict what solving it costs, Wobbl only
needs how big the model is and what the solver is asked to do, so each reader walks the file line by line,
follows includes, counts nodes and elements by type, keeps a thin sample of node coordinates for the
separator estimate in ``wobbl.estimate``, and reads the step definitions. Memory stays flat whatever the
size of the deck.

    CalculiX     .inp      nodes, elements, *STEP procedures with their periods and increments
    LS-DYNA      .k .key   nodes, elements, end time, mass scaled time step, plot interval
    OpenRadioss  .rad      starter (_0000.rad) counts, engine (_0001.rad) end time
    OpenFOAM     a case    cells from the polyMesh header or blockMeshDict, times from controlDict
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE = 20000          # node coordinates kept for the separator estimate
INP_EXT = {".inp"}
DYNA_EXT = {".k", ".key", ".dyn"}
RAD_EXT = {".rad"}


class DeckError(ValueError):
    pass


@dataclass
class Step:
    procedure: str              # static, heat, heat-steady, dynamic, frequency, buckle, explicit, cfd
    period: float = 1.0         # step time
    initial: float = 0.0        # first increment
    max_inc: float = 0.0        # largest increment
    nlgeom: bool = False
    modes: int = 0              # frequency and buckling steps


@dataclass
class Deck:
    path: Path
    solver: str                 # calculix, openradioss, openfoam
    format: str                 # inp, keyword, radioss, openfoam
    nodes: int = 0
    elements: dict[str, int] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    end_time: float = 0.0       # explicit and CFD runs
    time_step: float = 0.0      # explicit: the mass scaled step if the deck sets one; CFD: deltaT
    output_interval: float = 0.0
    includes: list[Path] = field(default_factory=list)
    bytes: int = 0              # the deck and everything it includes
    bbox: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    sample: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    notes: list[str] = field(default_factory=list)

    @property
    def n_elements(self) -> int:
        return int(sum(self.elements.values()))

    def to_dict(self) -> dict:
        return {"path": str(self.path), "solver": self.solver, "format": self.format, "nodes": self.nodes,
                "elements": self.elements, "n_elements": self.n_elements,
                "steps": [s.__dict__ for s in self.steps], "end_time": self.end_time, "time_step": self.time_step,
                "output_interval": self.output_interval, "includes": [str(p) for p in self.includes],
                "bytes": self.bytes, "bbox": [list(self.bbox[0]), list(self.bbox[1])], "notes": self.notes}


class _Sampler:
    """A fixed size, evenly spread sample of node coordinates however many nodes follow. Each node is kept when
    a hash of its position in the deck falls under a threshold that halves whenever the sample fills. Taking
    every k-th node instead would alias with the numbering of a structured mesh and keep whole layers."""

    def __init__(self, cap: int = SAMPLE):
        self.cap, self.i, self.limit, self.pts, self.keys = cap, 0, 1 << 32, [], []
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)

    def add(self, x: float, y: float, z: float):
        h = (self.i * 2654435761 + 12345) & 0xFFFFFFFF          # Knuth's multiplicative hash
        if h < self.limit:
            self.pts.append((x, y, z))
            self.keys.append(h)
            if len(self.pts) >= self.cap:
                self.limit //= 2
                keep = [j for j, k in enumerate(self.keys) if k < self.limit]
                self.pts = [self.pts[j] for j in keep]
                self.keys = [self.keys[j] for j in keep]
        self.i += 1
        p = (x, y, z)
        for d in range(3):
            if p[d] < self.lo[d]:
                self.lo[d] = p[d]
            if p[d] > self.hi[d]:
                self.hi[d] = p[d]

    def finish(self, deck: Deck):
        deck.sample = np.asarray(self.pts, float).reshape(-1, 3)
        if self.i:
            deck.bbox = (tuple(float(v) for v in self.lo), tuple(float(v) for v in self.hi))


def _floats(parts: list[str]) -> list[float]:
    out = []
    for p in parts:
        try:
            out.append(float(p.replace("D", "E").replace("d", "e")))
        except ValueError:
            out.append(0.0)
    return out


def _param(line: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*([^,\s]+)", line, re.I)
    return m.group(1) if m else None


def _include(base: Path, name: str) -> Path:
    p = Path(name.strip().strip("'\""))
    return p if p.is_absolute() else (base.parent / p)


# ---------------------------------------------------------------------------- CalculiX
_INP_PROC = {"*STATIC": "static", "*HEAT TRANSFER": "heat", "*DYNAMIC": "dynamic", "*FREQUENCY": "frequency",
             "*BUCKLE": "buckle", "*COUPLED TEMPERATURE-DISPLACEMENT": "dynamic", "*VISCO": "static",
             "*MODAL DYNAMIC": "dynamic", "*UNCOUPLED TEMPERATURE-DISPLACEMENT": "dynamic"}


def read_inp(path: str | Path) -> Deck:
    """Scan a CalculiX (Abaqus style) deck and the files it includes."""
    path = Path(path)
    deck = Deck(path, "calculix", "inp")
    sampler = _Sampler()
    state = {"block": None, "etype": "", "step": None, "want_data": False, "cont": False}
    _scan_inp(path, deck, sampler, state, set())
    sampler.finish(deck)
    if not deck.steps:
        deck.notes.append("The deck has no *STEP, so there is nothing for CalculiX to solve.")
    return deck


def _scan_inp(path: Path, deck: Deck, sampler: _Sampler, st: dict, seen: set):
    if not path.exists():
        raise DeckError(f"{path} is missing (included from the deck)" if seen else f"file not found: {path}")
    if path.resolve() in seen:
        return
    seen.add(path.resolve())
    deck.bytes += path.stat().st_size
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("**") or not line.strip():
                continue
            if line.startswith("*"):
                st["cont"] = False
                _inp_keyword(line, path, deck, sampler, st, seen)
                continue
            blk = st["block"]
            if blk == "node":
                parts = line.split(",")
                deck.nodes += 1
                if len(parts) >= 4:
                    sampler.add(*_floats(parts[1:4]))
                elif len(parts) >= 2:
                    sampler.add(*(_floats(parts[1:]) + [0.0, 0.0])[:3])
            elif blk == "element":
                # a record that does not fit on one line ends its line with a comma
                if not st["cont"]:
                    deck.elements[st["etype"]] = deck.elements.get(st["etype"], 0) + 1
                st["cont"] = line.rstrip().endswith(",")
            elif blk == "proc" and st["want_data"]:
                _inp_step_data(line, st["step"])
                st["want_data"] = False


def _inp_keyword(line: str, path: Path, deck: Deck, sampler: _Sampler, st: dict, seen: set):
    up = line.upper()
    key = up.split(",")[0].strip()
    st["block"] = None
    if key == "*INCLUDE":
        inc = _include(path, _param(line, "INPUT") or "")
        deck.includes.append(inc)
        _scan_inp(inc, deck, sampler, st, seen)
    elif key == "*NODE":
        st["block"] = "node"
    elif key == "*ELEMENT":
        st["block"] = "element"
        st["etype"] = (_param(line, "TYPE") or "unknown").upper()
    elif key == "*STEP":
        step = Step("static", nlgeom="NLGEOM" in up and "NLGEOM=NO" not in up.replace(" ", ""))
        deck.steps.append(step)
        st["step"] = step
    elif key in _INP_PROC and st["step"] is not None:
        step = st["step"]
        step.procedure = _INP_PROC[key]
        if key == "*HEAT TRANSFER" and "STEADY STATE" in up:
            step.procedure = "heat-steady"
        elif key == "*DYNAMIC" and "EXPLICIT" in up:
            step.procedure = "explicit"
        st["block"] = "proc"
        st["want_data"] = True
    elif key == "*END STEP":
        st["step"] = None


def _inp_step_data(line: str, step: Step):
    v = _floats([p for p in line.split(",") if p.strip()])
    if step.procedure in ("frequency", "buckle"):
        step.modes = int(v[0]) if v else 10
        return
    if len(v) >= 2:
        step.initial, step.period = v[0], v[1] or 1.0
    if len(v) >= 4:
        step.max_inc = v[3]


# ----------------------------------------------------------------------------- LS-DYNA
def _dyna_fields(line: str, widths=(10,) * 8) -> list[str]:
    if "," in line:
        return [p.strip() for p in line.split(",")]
    out, i = [], 0
    for w in widths:
        out.append(line[i:i + w].strip())
        i += w
    return out


def read_keyword(path: str | Path) -> Deck:
    """Scan an LS-DYNA keyword deck, which OpenRadioss reads directly."""
    path = Path(path)
    deck = Deck(path, "openradioss", "keyword")
    sampler = _Sampler()
    st = {"block": None, "lines": 1, "left": 0, "card": None}
    _scan_keyword(path, deck, sampler, st, set())
    sampler.finish(deck)
    if not deck.end_time:
        deck.notes.append("No *CONTROL_TERMINATION end time was found.")
    deck.steps = [Step("explicit", period=deck.end_time)]
    return deck


def _scan_keyword(path: Path, deck: Deck, sampler: _Sampler, st: dict, seen: set):
    if not path.exists():
        raise DeckError(f"{path} is missing (included from the deck)" if seen else f"file not found: {path}")
    if path.resolve() in seen:
        return
    seen.add(path.resolve())
    deck.bytes += path.stat().st_size
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("$") or not line.strip():
                continue
            if line.startswith("*"):
                key = line.strip().upper()
                st["block"], st["card"] = None, None
                if key.startswith("*INCLUDE"):
                    st["block"] = "include"
                elif key.startswith("*NODE") and not key.startswith("*NODE_"):
                    st["block"] = "node"
                elif key.startswith("*ELEMENT_"):
                    kind = key.split("_")[1]
                    st["block"], st["etype"] = "element", kind
                    # options that add a card to every element record
                    st["lines"] = 1 + sum(o in key for o in ("THICKNESS", "BETA", "MCID", "OFFSET", "SECTION",
                                                             "ORTHO", "DOF"))
                    st["left"] = 0
                elif key.startswith("*CONTROL_TERMINATION"):
                    st["card"] = "termination"
                elif key.startswith("*CONTROL_TIMESTEP"):
                    st["card"] = "timestep"
                elif key.startswith("*DATABASE_BINARY_D3PLOT"):
                    st["card"] = "d3plot"
                continue
            blk = st["block"]
            if blk == "node":
                f = _dyna_fields(line, (8, 16, 16, 16))
                deck.nodes += 1
                sampler.add(*_floats(f[1:4]))
            elif blk == "element":
                if st["left"] == 0:
                    deck.elements[st["etype"]] = deck.elements.get(st["etype"], 0) + 1
                    st["left"] = st["lines"]
                st["left"] -= 1
            elif blk == "include":
                inc = _include(path, line)
                deck.includes.append(inc)
                _scan_keyword(inc, deck, sampler, {"block": None, "lines": 1, "left": 0, "card": None}, seen)
                st["block"] = None
            elif st["card"]:
                v = _floats([x for x in _dyna_fields(line)])
                if st["card"] == "termination" and v:
                    deck.end_time = v[0]
                elif st["card"] == "timestep" and len(v) >= 5 and v[4]:
                    deck.time_step = abs(v[4])     # DT2MS: the step mass scaling holds
                elif st["card"] == "d3plot" and v:
                    deck.output_interval = v[0]
                st["card"] = None


# --------------------------------------------------------------------------- OpenRadioss
_RAD_ELEM = {"/SHELL": "SHELL", "/SH3N": "SHELL", "/BRICK": "SOLID", "/TETRA4": "SOLID", "/TETRA10": "SOLID",
             "/BEAM": "BEAM", "/SPRING": "DISCRETE", "/TRUSS": "BEAM", "/QUAD": "SHELL"}


def read_radioss(path: str | Path) -> Deck:
    """Scan an OpenRadioss starter deck (name_0000.rad) and, next to it, its first engine file."""
    path = Path(path)
    deck = Deck(path, "openradioss", "radioss")
    sampler = _Sampler()
    if path.stem.endswith("_0000") or not path.stem.endswith("_0001"):
        deck.bytes += path.stat().st_size
        blk = None
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or line.startswith("$") or not line.strip():
                    continue
                if line.startswith("/"):
                    head = "/" + line.strip().split("/")[1].upper()
                    blk = "node" if head == "/NODE" else (_RAD_ELEM.get(head) and ("el", _RAD_ELEM[head]))
                    continue
                if blk == "node":
                    f = _dyna_fields(line, (10, 20, 20, 20))
                    deck.nodes += 1
                    sampler.add(*_floats(f[1:4]))
                elif blk:
                    deck.elements[blk[1]] = deck.elements.get(blk[1], 0) + 1
        engine = path.with_name(path.stem.replace("_0000", "") + "_0001.rad")
    else:
        engine = path
    if engine.exists():
        _radioss_engine(engine, deck)
    sampler.finish(deck)
    deck.steps = [Step("explicit", period=deck.end_time)]
    return deck


def _radioss_engine(path: Path, deck: Deck):
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip() and not ln.startswith("#")]
    for i, ln in enumerate(lines):
        up = ln.strip().upper()
        nxt = _floats(lines[i + 1].split()) if i + 1 < len(lines) else []
        if up.startswith("/RUN/") and nxt:
            deck.end_time = nxt[0]
        elif up.startswith("/DT/NODA") and len(nxt) >= 2 and nxt[1]:
            deck.time_step = nxt[1]
        elif up.startswith("/ANIM/DT") and len(nxt) >= 2:
            deck.output_interval = nxt[1]


# ------------------------------------------------------------------------------ OpenFOAM
def _foam_value(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{key}\s+([^;]+);", text, re.M)
    return m.group(1).strip() if m else None


def read_foam(case: str | Path) -> Deck:
    """Cells and times of an OpenFOAM case, without reading the mesh itself."""
    case = Path(case)
    deck = Deck(case, "openfoam", "openfoam")
    ctrl = case / "system" / "controlDict"
    if not ctrl.exists():
        raise DeckError(f"{case} is not an OpenFOAM case (no system/controlDict)")
    text = ctrl.read_text(errors="replace")
    app = _foam_value(text, "application") or "simpleFoam"
    deck.end_time = float(_floats([_foam_value(text, "endTime") or "0"])[0])
    deck.time_step = float(_floats([_foam_value(text, "deltaT") or "1"])[0])
    deck.output_interval = float(_floats([_foam_value(text, "writeInterval") or "0"])[0])
    deck.steps = [Step("cfd", period=deck.end_time, initial=deck.time_step)]
    deck.notes.append(f"Solver application {app}.")
    owner = case / "constant" / "polyMesh" / "owner"
    block = case / "system" / "blockMeshDict"
    if owner.exists():
        with open(owner, "r", errors="replace") as fh:
            head = fh.read(4096)
        m = re.search(r"nPoints:\s*(\d+)\s+nCells:\s*(\d+)", head)
        if m:
            deck.nodes, deck.elements = int(m.group(1)), {"CELL": int(m.group(2))}
    if not deck.elements and block.exists():
        cells = 0
        for m in re.finditer(r"hex\s*\([^)]*\)\s*(?:\w+\s*)?\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)",
                             block.read_text(errors="replace")):
            cells += int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        if cells:
            deck.elements = {"CELL": cells}
            deck.notes.append("Cells counted from blockMeshDict; a snappyHexMesh step would add more.")
    if not deck.elements:
        deck.notes.append("The case has no mesh yet, so its cell count is unknown.")
    deck.bytes = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
    return deck


def read_deck(path: str | Path) -> Deck:
    """Pick the reader from the file type: .inp, .k or .key, .rad, or an OpenFOAM case folder."""
    p = Path(path)
    if p.is_dir():
        return read_foam(p)
    ext = p.suffix.lower()
    if ext in INP_EXT:
        return read_inp(p)
    if ext in DYNA_EXT:
        return read_keyword(p)
    if ext in RAD_EXT:
        return read_radioss(p)
    raise DeckError(f"Wobbl reads CalculiX .inp, LS-DYNA .k, OpenRadioss .rad decks and OpenFOAM case folders, "
                    f"not '{ext}'")
