"""A circuit as Wobbl holds it: parts with named pins on nets, plus the models and subcircuits they use.

Every input becomes a Circuit. A SPICE netlist is read line by line, a KiCad schematic is traced from its
wires (kicad.py), a plain description is composed from parts (describe.py) and the templates write their
own netlists. Lines stay as written, so a user's models and parameters reach ngspice untouched.

Instrumenting a circuit puts a 0 V source in series with the pins of each part, so ngspice reports the
current into every pin. With the node voltages that gives each part's voltage, current and power over
time, whatever the part is inside (a resistor, a transistor model, a regulator subcircuit), and it gives
a KCL and power account of the whole board for free.

Board parasitics join here too. ``join_parasitics`` takes trace resistance and inductance between two pins
of a net and copper capacitance to ground, as a board study measures them, and splices them into the
netlist before it is solved.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..units import parse_spice_value

# pin names in node order for each SPICE letter; parts with control inputs list only the terminals that
# carry current (the control nodes of E, G and S elements stay in the rest of the line)
PINS = {"r": ["1", "2"], "c": ["1", "2"], "l": ["1", "2"], "d": ["a", "k"], "v": ["+", "-"], "i": ["+", "-"],
        "b": ["+", "-"], "e": ["+", "-"], "g": ["+", "-"], "f": ["+", "-"], "h": ["+", "-"], "s": ["+", "-"],
        "w": ["+", "-"], "q": ["c", "b", "e", "s"], "j": ["d", "g", "s"], "m": ["d", "g", "s", "b"],
        "t": ["p1", "r1", "p2", "r2"], "z": ["d", "g", "s"]}
NODES = {"r": 2, "c": 2, "l": 2, "d": 2, "v": 2, "i": 2, "b": 2, "e": 2, "g": 2, "f": 2, "h": 2, "s": 2, "w": 2,
         "j": 3, "z": 3, "t": 4}
ANALYSES = (".tran", ".ac", ".dc", ".op", ".noise", ".tf", ".sens", ".four", ".pz", ".disto", ".sp")
DROP = (".end", ".print", ".plot", ".probe", ".save", ".backanno", ".width", ".meas", ".measure", ".control",
        ".endc", ".title")
GROUND = {"0", "gnd", "gnd!", "agnd", "dgnd", "gnda", "gndd", "vss", "gnd_", "ground", "earth", "pgnd", "sgnd"}


class NetlistError(ValueError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class Element:
    name: str                       # as written: R1, D2, XU1
    letter: str                     # the SPICE letter, lower case
    nodes: list[str]                # current carrying terminals, lower case
    rest: list[str]                 # everything after those nodes, as written
    pins: list[str]                 # a name for each node
    kind: str = ""                  # resistor, led, nmos, regulator ... set when parts are bound
    part: dict = field(default_factory=dict)   # the bound library part: ratings, model, provenance
    meta: dict = field(default_factory=dict)   # footprint, part number, fields, the annotation after ';'

    @property
    def ref(self) -> str:
        """The name people use: XU1 is U1."""
        return self.name[1:] if self.letter == "x" and len(self.name) > 1 and self.name[1].isalpha() else self.name

    @property
    def model(self) -> str:
        """The model or subcircuit a semiconductor or X line names."""
        toks = [t for t in self.rest if "=" not in t and t.lower() != "params:"]
        if self.letter == "x":
            return toks[-1] if toks else ""
        if self.letter in "dqmjzsw":
            return toks[0] if toks else ""
        return ""

    @property
    def value(self) -> float | None:
        """The numeric value of a resistor, capacitor or inductor, when it is a plain number."""
        if self.letter not in "rcl":
            return None
        for t in self.rest:
            k, _, v = t.partition("=")
            if v and k.lower() in ("r", "c", "l", "resistance", "capacitance", "inductance"):
                t = v
            elif v:
                continue
            try:
                return parse_spice_value(t)
            except Exception:  # noqa: BLE001
                return None
        return None

    def set_value(self, v: float):
        """Replace the value of a resistor, capacitor or inductor, keeping its options."""
        for k, t in enumerate(self.rest):
            if "=" not in t or t.split("=")[0].lower() in ("r", "c", "l"):
                self.rest[k] = f"{v:.6g}"
                return
        self.rest.insert(0, f"{v:.6g}")

    def line(self, nodes: list[str] | None = None) -> str:
        return " ".join([self.name, *(nodes or self.nodes), *self.rest])


@dataclass
class Subckt:
    name: str
    ports: list[str]
    text: str                       # the whole definition, .subckt to .ends


@dataclass
class Circuit:
    title: str = ""
    elements: list[Element] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)      # lower case name -> .model line
    subckts: dict[str, Subckt] = field(default_factory=dict)  # lower case name -> definition
    directives: list[str] = field(default_factory=list)       # .param, .options, .temp, .func, .global
    analyses: list[str] = field(default_factory=list)         # analysis cards found in the source
    assumptions: list[str] = field(default_factory=list)
    source: str = "netlist"         # netlist, kicad, description, template
    names: dict[str, str] = field(default_factory=dict)       # spice node -> the name the user gave it

    # --- lookups --------------------------------------------------------------------------------------
    def element(self, name: str) -> Element | None:
        low = name.lower()
        return next((e for e in self.elements if e.name.lower() == low or e.ref.lower() == low), None)

    def nets(self) -> dict[str, list[tuple[Element, str]]]:
        """Each net with the (part, pin) pairs on it."""
        out: dict[str, list[tuple[Element, str]]] = {}
        for e in self.elements:
            for n, p in zip(e.nodes, e.pins):
                out.setdefault(n, []).append((e, p))
        return out

    def nodes(self) -> list[str]:
        seen = []
        for e in self.elements:
            for n in e.nodes:
                if n not in seen:
                    seen.append(n)
        return seen

    def label(self, node: str) -> str:
        return self.names.get(node, node)

    def copy(self) -> "Circuit":
        return copy.deepcopy(self)

    # --- writing --------------------------------------------------------------------------------------
    def probes(self) -> list[tuple[Element, int, str]]:
        """The (part, pin index, ammeter name) of every instrumented pin. Two terminal parts need one ammeter,
        since what enters one pin leaves the other; parts with more pins get one on each, which also checks
        that their pin currents sum to zero."""
        out = []
        for e in self.elements:
            if e.letter in ("k", "a") or not e.nodes:
                continue
            idx = [0] if len(e.nodes) == 2 else range(len(e.nodes))
            for i in idx:
                out.append((e, i, f"vwb_{_safe(e.name)}_{i}"))
        return out

    def spice(self, instrument: bool = False, analyses: list[str] | tuple = (), extra: list[str] | tuple = (),
              save: bool = True) -> str:
        """The netlist as ngspice reads it. Instrumented, each probed pin moves to its own node behind a 0 V
        source, and only node voltages and those currents are saved."""
        lines = [f"* {self.title or 'circuit'}"]
        lines += self.directives
        moved: dict[tuple[str, int], str] = {}
        amm = []
        if instrument:
            for e, i, vname in self.probes():
                inner = f"wb_{_safe(e.name)}_{i}"
                moved[(e.name, i)] = inner
                amm.append(f"{vname} {e.nodes[i]} {inner} 0")
        for e in self.elements:
            nodes = [moved.get((e.name, i), n) for i, n in enumerate(e.nodes)]
            lines.append(e.line(nodes))
        lines += amm
        lines += list(self.models.values())
        lines += [s.text for s in self.subckts.values()]
        lines += list(extra)
        if instrument and save:
            vecs = [f"v({n})" for n in self.nodes() if n != "0"] + [f"i({v})" for _, _, v in self.probes()]
            for k in range(0, len(vecs), 12):
                lines.append(".save " + " ".join(vecs[k:k + 12]))
        lines += list(analyses)
        lines.append(".end")
        return "\n".join(lines) + "\n"


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def safe_node(name: str) -> str:
    """A net name ngspice reads as a node: ground names become 0, anything else keeps letters, digits and _."""
    n = name.strip().lower()
    if n in GROUND or n.lstrip("/") in GROUND:
        return "0"
    s = n.replace("+", "p").replace("-", "m").replace(".", "p")
    s = re.sub(r"[^a-z0-9_]", "_", s).strip("_") or "n"
    return s if not s[0].isdigit() or s == "0" else "n" + s


# --- reading SPICE ------------------------------------------------------------------------------------
def _logical_lines(text: str) -> list[tuple[str, str]]:
    """Lines with continuations joined, comments dropped, and the text after an inline ';' kept apart
    (Wobbl reads it as the part's annotation, such as '; 0805 50V' or '; part=IRLZ44N')."""
    out: list[list[str]] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s[0] == "*":
            continue
        note = ""
        m = re.search(r"\s;|^;|\s\$\s|//", s)
        if m:
            note = s[m.end():].strip()
            s = s[:m.start()].strip()
            if not s:
                continue
        if s[0] == "+" and out:
            out[-1][0] += " " + s[1:].strip()
            out[-1][1] = (out[-1][1] + " " + note).strip()
        else:
            out.append([s, note])
    return [(a, b) for a, b in out]


def _expand_includes(text: str, base: Path | None, depth: int = 0) -> str:
    """Inline .include and .lib files so their models and subcircuits are read with the rest."""
    if depth > 5:
        return text

    def sub(m):
        kind, arg = m.group(1).lower(), m.group(2).strip()
        toks = arg.split()
        fname = toks[0].strip("'\"")
        p = Path(fname).expanduser()
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.exists():
            return f"* (missing {kind} {fname})"
        body = p.read_text(errors="replace")
        if kind == ".lib" and len(toks) > 1:
            sec = re.search(rf"(?ims)^\.lib\s+{re.escape(toks[1])}\s*$(.*?)^\.endl", body)
            body = sec.group(1) if sec else body
        return _expand_includes(body, p.parent, depth + 1)

    return re.sub(r"(?im)^\s*(\.include|\.inc|\.lib)\s+(.+)$", sub, text)


def parse_spice(text: str, base: Path | None = None) -> Circuit:
    """A Circuit from SPICE text. Control blocks, prints and the .end card are dropped, analysis cards are
    kept apart, models and subcircuits are indexed by name."""
    title = ""
    first = text.lstrip().splitlines()[0].strip() if text.strip() else ""
    drop_first = False
    if first.startswith("*"):
        title = first.lstrip("*").strip()
    elif first and not first.startswith(".") and not _looks_like_element(first):
        title, drop_first = first, True     # SPICE reads the first line as the title, whatever it says
    text = re.sub(r"(?ims)^\s*\.control\b.*?^\s*\.endc\b[^\n]*$", "", text)
    text = _expand_includes(text, base)
    lines = _logical_lines(text)
    if drop_first and lines:
        lines = lines[1:]
    c = Circuit(title=title)
    pending: list[tuple[str, str]] = []
    depth, sub_lines, sub_head = 0, [], ""
    for s, note in lines:
        low = s.lower()
        if low.startswith(".subckt"):
            depth += 1
            if depth == 1:
                sub_head, sub_lines = s, [s]
                continue
        if depth:
            sub_lines.append(s)
            if low.startswith(".ends"):
                depth -= 1
                if not depth:
                    t = sub_head.split()
                    ports = [p for p in t[2:] if "=" not in p and p.lower() != "params:"]
                    if "params:" in [p.lower() for p in t]:
                        ports = t[2:[p.lower() for p in t].index("params:")]
                    c.subckts[t[1].lower()] = Subckt(t[1], [p.lower() for p in ports], "\n".join(sub_lines))
            continue
        if low.startswith(".model"):
            t = s.split(None, 2)
            if len(t) >= 2:
                c.models[t[1].lower()] = s
        elif low.startswith(ANALYSES):
            c.analyses.append(s)
        elif low.startswith(DROP) or low.startswith((".include", ".inc", ".lib", ".endl")):
            continue
        elif s.startswith("."):
            c.directives.append(s)
        else:
            pending.append((s, note))
    for s, note in pending:
        e = _element(s, c)
        if e is not None:
            if note:
                e.meta["note"] = note
            c.elements.append(e)
    return c


def _looks_like_element(line: str) -> bool:
    """Whether a first line is a part rather than a title: 'V1 in 0 5' is a part, 'LED test circuit' is not."""
    t = line.split()
    if len(t) < 4 or not re.match(r"^[a-zA-Z]+[\w.]*\d[\w.]*$|^[a-zA-Z]\d", t[0]):
        return False
    letter = t[0][0].lower()
    if letter in "rclvi":
        v = t[3].lower()
        return _num(v) or v.startswith(("{", "dc", "ac", "pulse", "sin", "pwl", "exp", "r=", "c=", "l="))
    return letter in "dqmjxbegfhswk"


def _is_model(tok: str, c: Circuit) -> bool:
    from .parts import lookup

    t = tok.lower()
    return t in c.models or t in c.subckts or lookup(tok) is not None


def _element(s: str, c: Circuit) -> Element | None:
    t = s.split()
    name, letter = t[0], t[0][0].lower()
    args = t[1:]
    if letter == "k":
        return Element(name, letter, [], args, [])
    if letter == "x":
        toks = [a for a in args if a.lower() != "params:"]
        bare = [k for k, a in enumerate(toks) if "=" not in a]
        if not bare:
            raise NetlistError(f"'{name}' names no subcircuit")
        sub = bare[-1]
        nodes = [n.lower() for n in toks[:sub]]
        rest = args[len(nodes):]
        d = c.subckts.get(toks[sub].lower())
        pins = d.ports if d and len(d.ports) == len(nodes) else [str(k + 1) for k in range(len(nodes))]
        return Element(name, letter, nodes, rest, list(pins))
    if letter == "q":
        n = 3
        if len(args) >= 5 and not _is_model(args[3], c) and "=" not in args[4] and not _num(args[4]):
            n = 4
        return Element(name, letter, [a.lower() for a in args[:n]], args[n:], PINS["q"][:n])
    if letter == "m":
        n = 4
        if len(args) >= 4:
            mdl = c.models.get(args[3].lower(), "")
            if "vdmos" in mdl.lower() or len(args) == 4 or (not mdl and _is_model(args[3], c) and
                                                            ("=" in args[4] or _num(args[4]))):
                n = 3     # a VDMOS model or a library power MOSFET: drain, gate, source
        return Element(name, letter, [a.lower() for a in args[:n]], args[n:], PINS["m"][:n])
    if letter in NODES:
        n = NODES[letter]
        if len(args) < n:
            raise NetlistError(f"'{s}' has too few nodes for a {letter.upper()} element")
        return Element(name, letter, [a.lower() for a in args[:n]], args[n:], PINS[letter][:n])
    if letter == "a":     # XSPICE code models keep their own node syntax; solved, not instrumented
        return Element(name, letter, [], args, [])
    raise NetlistError(f"Wobbl does not know the SPICE element '{name}'")


def _num(tok: str) -> bool:
    try:
        parse_spice_value(tok)
        return True
    except Exception:  # noqa: BLE001
        return False


# --- board parasitics ---------------------------------------------------------------------------------
def join_parasitics(c: Circuit, spec: dict) -> list[str]:
    """Splice board parasitics into the circuit and say what was added.

    spec = {"traces": [{"net": "sw", "to": "L1.1", "r": 0.012, "l": 6e-9}, ...],
            "nets": {"sw": {"c": 3e-10}, ...}, "source": "board study in wobbl-out/board"}

    A trace moves the pin named in "to" off its net onto a new node joined back through the trace's
    resistance and inductance, so the part sees the copper between it and the rest of the net. A net's
    "c" is the copper's capacitance to ground. The added parts are kind "parasitic": they count in the
    power account but are not judged against ratings."""
    said = []
    src = spec.get("source", "the board")
    for k, tr in enumerate(spec.get("traces", [])):
        net = safe_node(str(tr["net"]))
        ref, _, pin = str(tr["to"]).partition(".")
        e = c.element(ref)
        if e is None:
            raise NetlistError(f"parasitic trace to {tr['to']}: no part {ref}")
        idx = next((i for i, (n, p) in enumerate(zip(e.nodes, e.pins)) if n == net and (not pin or p == pin.lower()
                                                                                          or str(i + 1) == pin)), None)
        if idx is None:
            raise NetlistError(f"parasitic trace: {tr['to']} is not on net {tr['net']}")
        mid, end = f"{net}_pt{k}", f"{net}_pe{k}"
        e.nodes[idx] = end
        r, ind = float(tr.get("r", 0) or 0), float(tr.get("l", 0) or 0)
        c.elements.append(Element(f"Rpar{k}", "r", [net, mid], [f"{max(r, 1e-6):.6g}"], ["1", "2"], "parasitic"))
        if ind > 0:
            c.elements.append(Element(f"Lpar{k}", "l", [mid, end], [f"{ind:.6g}"], ["1", "2"], "parasitic"))
        else:
            c.elements[-1].nodes[1] = end
        said.append(f"trace on {c.label(net)} to {tr['to']}: {r * 1000:.3g} mohm and {ind * 1e9:.3g} nH from {src}")
    for k, (net, v) in enumerate((spec.get("nets") or {}).items()):
        cap = float(v.get("c", 0) or 0)
        if cap > 0:
            n = safe_node(str(net))
            c.elements.append(Element(f"Cpar{k}", "c", [n, "0"], [f"{cap:.6g}"], ["1", "2"], "parasitic"))
            said.append(f"{cap * 1e12:.3g} pF from {c.label(n)} to ground from {src}")
    c.assumptions += [f"Board parasitic: {s}." for s in said]
    return said
