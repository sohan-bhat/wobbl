"""Give every part of a circuit its kind, its library entry and a model ngspice can solve.

A part's identity comes from wherever the input carries it: the model or subcircuit name on a SPICE line,
the note after its ';' (such as '0805', '25V X7R' or 'part=IRLZ44N'), or a KiCad symbol's value, footprint
and part number fields. A model the netlist defines is used as written, with the library's ratings when the
library knows the part by that name. A part the netlist names but does not define gets the library's model.
A part the library does not know gets a generic model of its kind, marked generic so the report says so and
treats its ratings as assumed. Overrides such as R1.power=1W or C3.voltage=25V win over everything.
"""
from __future__ import annotations

import re

from ..units import parse_spice_value
from . import ics
from .netlist import Circuit, Element, NetlistError
from .parts import Part, bjt, capacitor, diode, inductor, led, lookup, mosfet, resistor

TIME_FN = ("pulse", "sin", "pwl", "exp", "sffm", "am", "trnoise", "trrandom")
PKG = r"(?<![0-9])(0201|0402|0603|0805|1206|1210|1812|2010|2512|0204|0207|0309|0411|0414|0516|0614|0617|0918|0922)" \
      r"(?![0-9])"


def notes(e: Element) -> dict:
    """What a part's note, value and footprint say about it: package, ratings, dielectric, part number."""
    text = " ".join(str(e.meta.get(k, "")) for k in ("note", "value", "footprint", "description"))
    out: dict = {}
    m = re.search(PKG, text)
    if m:
        out["pkg"] = m.group(1)
    for k in ("mpn", "part"):
        if e.meta.get(k):
            out["part"] = str(e.meta[k])
    m = re.search(r"\b(?:part|mpn)\s*=\s*(\S+)", text, re.I)
    if m:
        out["part"] = m.group(1)
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[vV](\d)?(?![a-zA-Z])", text)
    if m:
        out["volts"] = float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))
    m = re.search(r"(?<![\w.])(\d+/\d+|\d+(?:\.\d+)?)\s*(m?)[wW]\b", text)
    if m:
        a, b = (m.group(1).split("/") + ["1"])[:2]
        out["power"] = float(a) / float(b) * (1e-3 if m.group(2) else 1)
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(m?)[aA]\b", text)
    if m:
        out["amps"] = float(m.group(1)) * (1e-3 if m.group(2) else 1)
    m = re.search(r"\b(x7r|x5r|c0g|np0|y5v|x7s|x6s|x8r)\b", text, re.I)
    if m:
        out["dielectric"] = m.group(1).upper()
    if re.search(r"electrolytic|\belec\b|cp_|polari[sz]ed|\bcp\b|tantalum|polymer|radial", text, re.I):
        out["polarized"] = True
        out["dielectric"] = "tantalum" if re.search(r"tantalum", text, re.I) else out.get("dielectric", "aluminum")
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%", text)
    if m:
        out["tol"] = float(m.group(1)) / 100
    m = re.search(r"\b(red|green|blue|yellow|orange|white|warm-white|amber|uv|ir|pink|purple)\b", text, re.I)
    if m:
        out["color"] = m.group(1).lower()
    return out


def _model_kind(text: str, name: str) -> str:
    t = text.lower()
    m = re.search(r"^\.model\s+\S+\s+(\w+)", t)
    typ = m.group(1) if m else ""
    if typ == "d":
        bv = re.search(r"\bbv\s*=\s*([\d.e+-]+\w*)", t)
        try:
            bvv = parse_spice_value(bv.group(1)) if bv else 1e9
        except Exception:  # noqa: BLE001
            bvv = 1e9
        if "led" in name.lower():
            return "led"
        return "zener" if bvv < 60 and re.match(r"^(z|dz|bz|1n47|1n52|mmsz)", name.lower()) else "diode"
    if typ in ("npn", "pnp"):
        return typ
    if typ in ("nmos", "pmos"):
        return typ
    if typ == "vdmos":
        return "pmos" if "pchan" in t else "nmos"
    if typ in ("sw", "csw"):
        return "switch"
    return "part"


def _source_kind(e: Element) -> str:
    low = " ".join(e.rest).lower()
    if any(re.search(rf"\b{f}\s*\(", low) or low.startswith(f) for f in TIME_FN):
        return "signal"
    return "supply"


def dc_value(e: Element) -> float | None:
    """The DC value of a V or I source as written (the first plain number, or after DC)."""
    toks = " ".join(e.rest).replace("(", " ( ").split()
    for k, t in enumerate(toks):
        if t.lower() == "dc" and k + 1 < len(toks):
            try:
                return parse_spice_value(toks[k + 1])
            except Exception:  # noqa: BLE001
                return None
    for t in toks:
        if t.lower() in TIME_FN or t == "(" or t.lower() == "ac":
            break
        try:
            return parse_spice_value(t)
        except Exception:  # noqa: BLE001
            continue
    return None


def bind(c: Circuit, overrides: dict | None = None) -> None:
    """Fill in each element's kind and part, adding library models the netlist names but does not define."""
    over = {k.lower(): v for k, v in (overrides or {}).items() if "." in k}
    for e in list(c.elements):
        if e.kind == "parasitic":
            continue
        n = notes(e)
        part_name = n.get("part") or ""
        _bind_one(c, e, n, part_name)
        _apply_overrides(e, {k.split(".", 1)[1]: v for k, v in over.items() if k.split(".", 1)[0] == e.ref.lower()
                             or k.split(".", 1)[0] == e.name.lower()})


def _use_library(c: Circuit, e: Element, p: Part) -> None:
    """Point the element at a library model and add the model's text to the circuit."""
    e.part = _as_dict(p)
    e.kind = p.kind
    if not p.text:
        return
    if p.letter in ("d", "q", "m") and e.letter == p.letter:
        if e.letter == "m" and len(e.nodes) == 4:      # a VDMOS has its body tied to its source
            e.nodes, e.pins = e.nodes[:3], e.pins[:3]
        toks = [t for t in e.rest if "=" not in t]
        if toks:
            e.rest[e.rest.index(toks[0])] = p.mname
        else:
            e.rest.insert(0, p.mname)
        c.models[p.mname.lower()] = p.text
    elif p.letter == "x":
        if e.letter != "x":
            e.name, e.letter = "X" + e.name, "x"
            e.rest = [p.mname]
        else:
            bare = [k for k, t in enumerate(e.rest) if "=" not in t and t.lower() != "params:"]
            e.rest[bare[-1] if bare else 0] = p.mname
        if p.ports and len(p.ports) == len(e.nodes) and all(x.isdigit() for x in e.pins):
            e.pins = list(p.ports)
        c.subckts[p.mname.lower()] = _subckt(p)


def _subckt(p: Part):
    from .netlist import Subckt
    return Subckt(p.mname, list(p.ports), p.text)


def _as_dict(p: Part) -> dict:
    return {"key": p.key, "kind": p.kind, "ratings": dict(p.ratings), "source": p.source, "generic": p.generic,
            "params": dict(p.params), "ports": list(p.ports), "mname": p.mname}


def _bind_one(c: Circuit, e: Element, n: dict, part_name: str) -> None:
    L = e.letter
    if e.meta.get("as") and L != "r":     # a battery, motor or input a schematic already identified
        e.part, e.kind = dict(e.meta["as"]), e.meta["as"]["kind"]
        return
    if L == "r":
        e.part, e.kind = _as_dict(resistor(n.get("pkg", ""), n.get("power"), None, n.get("tol", 0.05))), "resistor"
        if e.meta.get("as"):            # a fuse, a pot half or a coil that a schematic turned into a resistor
            e.part, e.kind = e.meta["as"], e.meta["as"]["kind"]
        elif re.match(r"^r_?(l|load|out|lamp|heater)\d*$", e.name, re.I) and not n.get("pkg") and not n.get("power"):
            e.kind = "load"              # RL and Rload stand for whatever the circuit drives, not a resistor part
            e.part = {"key": "load", "kind": "load", "ratings": {}, "generic": False, "params": {},
                      "source": "a load the circuit drives, not judged as a resistor"}
        return
    if L == "c":
        cp = capacitor(n.get("volts"), n.get("dielectric", ""), n.get("pkg", ""), bool(n.get("polarized")), None,
                       n.get("tol"))
        e.part, e.kind = _as_dict(cp), cp.kind
        return
    if L == "l":
        e.part, e.kind = _as_dict(inductor(n.get("amps"))), "inductor"
        return
    if L in "vi":
        e.kind = _source_kind(e) if L == "v" else "isource"
        bat = ics.battery(e.meta.get("note", "") or e.meta.get("value", "")) if L == "v" else None
        if bat:
            e.part = _as_dict(bat)
        return
    if L in "begfh":
        e.kind = "source"
        return
    if L in "sw":
        e.kind = "switch"
        return
    if L in "kta":
        e.kind = {"k": "coupling", "t": "line", "a": "xspice"}[L]
        return
    name = e.model
    defined = name.lower() in c.models or name.lower() in c.subckts
    lib = lookup(part_name) or lookup(name) or (lookup(str(e.meta.get("value", ""))) if e.meta.get("value") else None)
    if defined:
        kind = _model_kind(c.models.get(name.lower(), ""), name) if name.lower() in c.models else "subcircuit"
        e.kind = lib.kind if lib and (kind in ("part", "subcircuit") or lib.kind.endswith(kind[-3:])) else kind
        if lib:
            d = _as_dict(lib)
            d["source"] = f"model {name} as the netlist defines it, ratings from the library ({lib.source})"
            e.part = d
        else:
            e.part = {"key": name, "kind": e.kind, "ratings": {}, "source": f"model {name} as the netlist defines it, "
                      "no ratings known", "generic": True, "params": {}}
        return
    if lib is not None and (lib.letter == L or lib.letter == "x" or (L == "x" and lib.kind != "mcu")):
        return _use_library(c, e, lib)
    if lib is not None and lib.kind == "mcu":
        e.part, e.kind = _as_dict(lib), "mcu"
        if all(x.isdigit() for x in e.pins):       # a netlist line: supply, ground, then the pins it uses
            e.pins = ["vdd", "gnd"] + [f"io{k}" for k in range(1, len(e.nodes) - 1)]
        return
    if L == "x" and not all(x.isdigit() for x in e.pins):
        return _generic_ic(c, e, name)
    _generic(c, e, n, name)


def _generic_ic(c: Circuit, e: Element, name: str) -> None:
    """An IC the library does not know, from a schematic that names its pins: its supply pins draw 5 mA and its
    other pins are left high impedance, so the rest of the board can still be checked."""
    sup = [k for k, p in enumerate(e.pins) if ics.SUPPLY_PIN.match(p)]
    gnd = [k for k, p in enumerate(e.pins) if ics.GROUND_PIN.match(p)]
    m = "UG_" + re.sub(r"[^A-Za-z0-9]", "_", name or e.ref) + f"_{len(e.pins)}"
    ports = [f"p{k}" for k in range(len(e.pins))]
    lines = [f".subckt {m} " + " ".join(ports)]
    if sup and gnd:
        for k in sup:
            lines.append(f"B{k} p{k} p{gnd[0]} I = {0.005 / len(sup):.4g}*min(1, max(0, V(p{k},p{gnd[0]})/1.0))")
        for k in gnd[1:]:
            lines.append(f"Rg{k} p{k} p{gnd[0]} 0.01")
    lines.append(".ends")
    from .netlist import Subckt
    c.subckts[m.lower()] = Subckt(m, ports, "\n".join(lines))
    bare = [k for k, t in enumerate(e.rest) if "=" not in t and t.lower() != "params:"]
    if bare:
        e.rest[bare[-1]] = m
    else:
        e.rest = [m]
    e.kind = "ic"
    e.part = {"key": name or e.ref, "kind": "ic", "ratings": {}, "generic": True, "params": {},
              "source": "not in the parts library: modeled as a 5 mA load on its supply pins with its other pins "
                        "high impedance"}
    c.assumptions.append(f"{e.ref} ({name or 'no part number'}) is not in the parts library, so it is modeled as a "
                         "5 mA load on its supply with its other pins left high impedance.")


def _generic(c: Circuit, e: Element, n: dict, name: str) -> None:
    """A stand in for a part the library does not know, with its ratings marked as assumed."""
    L = e.letter
    hint = (name + " " + str(e.meta.get("lib_id", "")) + " " + str(e.meta.get("value", ""))).lower()
    if L == "d":
        if "led" in hint:
            p = led(n.get("color", "red"), n.get("pkg", ""))
        else:
            p = diode("1N4001")
            p.source = "generic silicon rectifier with 1N4001 ratings (1 A, 50 V)"
        p.generic, p.key = True, name or p.key
        what = "LED" if p.kind == "led" else "diode"
    elif L == "q":
        pnp = "pnp" in hint
        p = bjt("2N3906" if pnp else "2N3904")
        p.source = f"generic small signal {'PNP' if pnp else 'NPN'} with 2N390{'6' if pnp else '4'} ratings"
        p.generic, p.key, what = True, name or p.key, "transistor"
    elif L == "m":
        pmos = "pmos" in hint or "p-ch" in hint or "pchan" in hint
        big = n.get("pkg") is None and re.search(r"to-?220|to-?247|d2pak|dpak|to-?252|to-?263", hint)
        p = mosfet(("IRF9540N" if pmos else "IRFZ44N") if big else ("AO3401" if pmos else "AO3400"))
        p.source = f"generic {'P' if pmos else 'N'} channel MOSFET with {p.key} ratings"
        p.generic, p.key, what = True, name or p.key, "MOSFET"
    else:
        raise NetlistError(f"{e.ref} uses '{name}', which neither the netlist nor Wobbl's parts library defines",
                           "Include its model with .include, or name a part the library has")
    c.assumptions.append(f"{e.ref} ({name or 'no part number'}) is not in the parts library, so it runs as a generic "
                         f"{what}: {p.source}.")
    _use_library(c, e, p)
    e.part["generic"] = True


def _apply_overrides(e: Element, o: dict) -> None:
    if not o:
        return
    r = e.part.setdefault("ratings", {})
    units = {"power": ("power_W", "W"), "voltage": ("voltage_V", "V"), "volts": ("voltage_V", "V"),
             "current": ("if_A" if e.kind in ("diode", "led", "schottky") else "id_A" if e.kind.endswith("mos")
                         else "isat_A" if e.kind == "inductor" else "iout_A", "A"),
             "isat": ("isat_A", "A"), "irms": ("irms_A", "A"), "theta_ja": ("theta_ja", ""), "tj_max": ("tj_max_C", "C"),
             "ripple": ("ripple_A", "A")}
    for k, v in o.items():
        if k in units:
            key, unit = units[k]
            r[key] = parse_spice_value(str(v).lower().rstrip(unit.lower()).rstrip("c/w") or "0")
            e.part["generic"] = False
            e.part["source"] = (e.part.get("source", "") + f"; {k} {v} as given").lstrip("; ")
        elif k == "heatsink":
            e.part.setdefault("params", {})["heatsink"] = parse_spice_value(str(v).lower().replace("c/w", ""))
        else:
            e.meta.setdefault("modes", {})[k] = str(v).lower()
