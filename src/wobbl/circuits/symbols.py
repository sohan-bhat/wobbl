"""Schematic symbols as parts: what a KiCad symbol is, from its library name, reference, value and part number,
and which of its pins is which, by pin name.

KiCad's own simulation fields (Sim.Device, Sim.Name, Sim.Pins, or Spice_Primitive and Spice_Model) win when a
symbol has them. Otherwise resistors, capacitors, inductors, diodes, LEDs and transistors are recognized from
the Device and Transistor libraries, ICs from the parts library by value or part number with their pins mapped
by name (VI, VO and GND on a regulator; +, - and the output on each unit of an op amp; TR, THR and DIS on a
555), batteries and USB or barrel jack inputs become supplies, switches and jumpers become closed or open
contacts, and anything the library does not know becomes a generic IC that draws a little from its supply
pins, with a note saying so.
"""
from __future__ import annotations

import re

from ..units import parse_spice_value
from . import ics
from .netlist import Circuit, Element, safe_node
from .parts import lookup

RAIL = re.compile(r"^[+]?(\d+)(?:[v.](\d+))?v?$|^[+]?(\d+(?:\.\d+)?)v$", re.I)
NAMED_RAILS = {"vcc": 5.0, "vdd": 3.3, "vbus": 5.0, "+vbus": 5.0, "usb_vbus": 5.0, "vusb": 5.0, "vbat": 3.7,
               "+vbat": 3.7, "vin": None, "+vin": None, "v+": None, "vsys": None, "vmot": None, "vm": None}


def rail_volts(name: str) -> float | None:
    n = name.strip().lstrip("/").lower()
    neg = n.startswith("-")
    m = RAIL.match(n.lstrip("-"))
    if m:
        v = float(m.group(3)) if m.group(3) else float(f"{m.group(1)}.{m.group(2) or 0}")
        return -v if neg else v
    return None


def _volts(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*v(\d)?\b", text, re.I)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def _name(letter: str, ref: str) -> str:
    """A SPICE element name for a reference: LED1 is not an inductor, so a diode named LED1 becomes DLED1."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", ref)
    return clean if clean[:1].lower() == letter else letter.upper() + clean


def attach(c: Circuit, s, pins: list[tuple], params: dict) -> None:
    from .kicad import mpn

    lib = s.lib_id.lower()
    libname = lib.split(":")[-1]
    ref = s.ref
    pre = (re.match(r"^[A-Za-z]+", ref) or re.match(r".*", ref)).group(0).upper()
    part_no = mpn(s)
    val = s.value
    meta = {"value": val, "footprint": s.footprint, "lib_id": s.lib_id, "ref": ref}
    if part_no:
        meta["mpn"] = part_no
    by_num = {str(n): safe_node(net) for n, _, net, _ in pins}
    by_name: dict[str, str] = {}
    for n, nm, net, _ in pins:
        by_name.setdefault(nm.lower(), safe_node(net))
    order = [safe_node(net) for _, _, net, _ in sorted(pins, key=lambda p: _num(p[0]))]

    def add(letter, nodes, rest, pin_names, **extra):
        e = Element(_name(letter, extra.get("ref", ref)), letter, nodes, rest, pin_names, meta={**meta, **extra})
        c.elements.append(e)
        return e

    def pick(*names):
        for k in names:
            if k in by_name:
                return by_name[k]
            if k in by_num:
                return by_num[k]
        return None

    if _sim_fields(c, s, pins, add, by_num):
        return
    found = lookup(part_no) or lookup(val) or lookup(libname)
    if _is(lib, "device:r", "device:thermistor") or pre in ("R", "RV", "TH", "RT", "LDR") and not lib.startswith(
            ("device:l", "device:c", "connector")):
        if len(pins) >= 3 or "potentiometer" in libname or pre == "RV":
            total = _value(val) or 10e3
            w = pick("2", "w", "wiper") or order[1]
            a, b = pick("1") or order[0], pick("3") or order[-1]
            add("r", [a, w], [f"{total / 2:.6g}"], ["1", "2"], ref=ref + "a")
            add("r", [w, b], [f"{total / 2:.6g}"], ["1", "2"], ref=ref + "b")
            c.assumptions.append(f"{ref} is a potentiometer, set to its middle ({total / 2:.6g} ohm each side).")
            return
        add("r", order[:2], [_num_text(val, "1k")], ["1", "2"])
        return
    if lib.startswith("connector") and not ("usb" in libname or "barrel" in libname or "jack_dc" in libname):
        _connector(c, s, pins, params, add, order)
        return
    if pre in ("D", "LED") or lib.startswith(("device:d", "device:led", "diode:", "led:")):
        _diode(c, s, pins, add, pick, order, lib, libname, val, part_no)
        return
    if _is(lib, "device:c") or pre in ("C", "CP") and not lib.startswith("device:l"):
        pol = "polar" in libname or libname.startswith("cp") or pre == "CP"
        e = add("c", [pick("1", "+") or order[0], pick("2", "-") or order[1]], [_num_text(val, "100n")], ["1", "2"])
        if pol:
            e.meta["note"] = "polarized electrolytic " + e.meta.get("note", "")
        return
    if _is(lib, "device:l") or pre == "L":
        add("l", order[:2], [_num_text(val, "10u")], ["1", "2"])
        return
    if pre == "FB" or "ferrite" in libname:
        add("r", order[:2], ["0.05"], ["1", "2"])
        c.assumptions.append(f"{ref} is a ferrite bead, simulated as 50 mohm.")
        return
    if pre == "Q" or lib.startswith(("device:q_", "transistor_")) or (found and found.kind in ("npn", "pnp", "nmos",
                                                                                                  "pmos")):
        kind = found.kind if found else ("pmos" if "pmos" in libname or "p-ch" in libname else "nmos" if "mos" in
                                         libname or "fet" in lib else "pnp" if "pnp" in libname else "npn")
        model = (part_no if lookup(part_no) else val) or kind.upper()
        if kind in ("nmos", "pmos"):
            add("m", [pick("d", "drain") or order[0], pick("g", "gate") or order[1], pick("s", "source") or order[2]],
                [model], ["d", "g", "s"], hint=kind)
        else:
            add("q", [pick("c", "collector") or order[0], pick("b", "base") or order[1], pick("e", "emitter") or
                      order[2]], [model], ["c", "b", "e"], hint=kind)
        return
    if pre == "BT" or lib.startswith("device:battery"):
        bat = ics.battery(val) or ics.battery(part_no)
        p_, n_ = pick("+", "1") or order[0], pick("-", "2") or order[-1]
        if bat:
            c.subckts[bat.mname.lower()] = _sub(bat)
            add("x", [p_, n_], [bat.mname], ["p", "n"], battery=val, **{"as": _part(bat)})
            return
        v = _volts(val) or 3.7
        add("v", [p_, n_], [f"{v:g}"], ["+", "-"], note=f"{v:g}V battery")
        c.assumptions.append(f"{ref} is a {v:g} V battery ({val or 'no value'}), simulated as an ideal source.")
        return
    if lib.startswith("connector:usb") or ("usb" in libname and pre in ("J", "P", "USB", "CN")):
        vb, g = pick("vbus", "v_bus", "vcc", "5v"), pick("gnd", "shield")
        if vb and g:
            usb = ics.battery("USB-C" if "c_" in libname or "type-c" in libname or "usb_c" in libname else "USB")
            c.subckts[usb.mname.lower()] = _sub(usb)
            add("x", [vb, g], [usb.mname], ["p", "n"], battery="USB", **{"as": _part(usb)})
            c.assumptions.append(f"{ref} is a USB input: {usb.source}.")
        return
    if "barrel" in libname or "dc_jack" in libname or "jack_dc" in libname:
        v = _volts(val) or _supply(params)
        if v is None:
            v = 12.0
            c.assumptions.append(f"{ref} is a DC input with no voltage on the schematic; assumed 12 V (set supply=).")
        add("v", [pick("1", "+", "tip") or order[0], pick("2", "-", "sleeve") or order[-1]], [f"{v:g}"], ["+", "-"],
            note=f"{v:g} V DC input")
        return
    if pre in ("SW", "S", "BTN", "K") and ("sw" in libname or "switch" in lib or "button" in libname):
        momentary = "push" in libname or "tact" in libname or "button" in libname
        state = str(params.get(f"{ref}.state", params.get(ref, "open" if momentary else "closed"))).lower()
        if state.startswith("clos") or state in ("on", "pressed"):
            add("r", order[:2], ["0.05"], ["1", "2"], **{"as": {"key": "switch", "kind": "switch", "ratings": {},
                                                                 "generic": False, "params": {},
                                                                 "source": "a closed switch, 50 mohm"}})
        c.assumptions.append(f"{ref} is a {'push button' if momentary else 'switch'}, taken as "
                             f"{'closed' if state.startswith('clos') or state in ('on', 'pressed') else 'open'} "
                             f"(set {ref}.state=open or closed).")
        return
    if pre == "JP" or "jumper" in libname:
        if "open" not in libname:
            add("r", order[:2], ["0.01"], ["1", "2"], **{"as": {"key": "jumper", "kind": "switch", "ratings": {},
                                                                 "generic": False, "params": {},
                                                                 "source": "a closed jumper"}})
        return
    if pre == "F" or "fuse" in libname:
        amps = _amps(val) or _amps(part_no) or 1.0
        f = ics.fuse(amps, "poly" in libname or "ptc" in val.lower(), val or "fuse")
        add("r", order[:2], [f"{f.params['r']:.4g}"], ["1", "2"],
            **{"as": {"key": val or "fuse", "kind": "fuse", "ratings": dict(f.ratings), "generic": False,
                      "params": dict(f.params), "source": f.source}})
        return
    if pre in ("LS", "SP") or "speaker" in libname:
        add("r", order[:2], ["8"], ["1", "2"], **{"as": _load("speaker", "an 8 ohm speaker")})
        return
    if pre in ("BZ",) or "buzzer" in libname:
        add("r", order[:2], ["170"], ["1", "2"], **{"as": _load("buzzer", "a buzzer drawing about 30 mA at 5 V")})
        return
    if pre == "M" and "motor" in lib:
        v = _volts(val) or _supply(params) or 6.0
        m = ics.motor(v, name=val or "motor")
        c.subckts[m.mname.lower()] = _sub(m)
        add("x", order[:2], [m.mname], ["p", "n"], **{"as": _part(m)})
        c.assumptions.append(f"{ref} is a DC motor: {m.source}.")
        return
    if pre in ("Y", "X") and ("crystal" in libname or "resonator" in libname or "osc" in libname):
        return
    if found is not None and found.kind not in ("led", "diode", "schottky", "zener", "npn", "pnp", "nmos", "pmos"):
        _ic(c, s, pins, found, add, by_name, by_num)
        return
    names = [nm for _, nm, _, _ in pins]
    if all(re.match(r"^\d*$|^~$", n) for n in names):
        c.assumptions.append(f"{ref} ({val or s.lib_id}) is not something Wobbl can simulate from its pins, so it is "
                             "left out.")
        return
    nodes, pnames = _unique_pins(pins)
    add("x", nodes, [val or libname or ref], pnames)


def _is(lib: str, *names: str) -> bool:
    """The symbol is one of these Device symbols or a variant of one (Device:R or Device:R_Small, not Device:RV)."""
    return any(lib == n or lib.startswith(n + "_") for n in names)


def _diode(c, s, pins, add, pick, order, lib, libname, val, part_no):
    ref = s.ref
    if len(pins) > 2 and "led" in lib:
        c.assumptions.append(f"{ref} has {len(pins)} pins; only its first die is simulated.")
    a = pick("a", "anode", "2") or order[-1]
    k = pick("k", "cathode", "1") or order[0]
    if "led" in lib or "led" in val.lower() or s.ref.upper().startswith("LED"):
        color = re.search(r"red|green|blue|yellow|orange|white|amber|uv|ir", f"{val} {libname} {part_no}", re.I)
        pkg = re.search(r"0402|0603|0805|1206|3mm|5mm", f"{s.footprint} {val}", re.I)
        model = f"LED_{(color.group(0) if color else 'red').upper()}" + (f"_{pkg.group(0).upper()}" if pkg else "")
        add("d", [a, k], [part_no if lookup(part_no) else model], ["a", "k"])
        return
    model = part_no or val
    if "zener" in libname and not lookup(model):
        vz = _volts(val) or _volts(part_no) or 5.1
        model = f"BZX55C{int(vz)}V{round((vz - int(vz)) * 10)}"
    if not lookup(model):
        model = val if lookup(val) else ("1N4148" if "small" in libname else model or "D_GENERIC")
    add("d", [a, k], [model], ["a", "k"])


def _connector(c, s, pins, params, add, order):
    """A connector is where the board meets the world: a load set on it (J2.load=2A or 12ohm) is drawn between
    its first two pins, and anything else is taken as unplugged."""
    load = params.get(f"{s.ref}.load") or params.get(f"{s.ref.lower()}.load")
    if load and len(order) >= 2:
        txt = str(load).lower().replace("ohm", "").strip()
        if txt.endswith("a"):
            from .netlist import safe_node
            v = rail_volts(next((n for n in (c.names.get(order[0], ""), c.names.get(order[1], "")) if rail_volts(n)),
                                "")) or _supply(params) or 12.0
            ohms = v / parse_spice_value(txt[:-1])
            _ = safe_node
        else:
            ohms = parse_spice_value(txt)
        e = add("r", order[:2], [f"{ohms:.4g}"], ["1", "2"], **{"as": _load("load", f"what {s.ref} drives, {load}")})
        e.meta["ref"] = f"{s.ref} load"
        e.name = f"RLOAD_{s.ref}"
        c.assumptions.append(f"{s.ref} ({s.value or 'connector'}) drives {load}, as set.")
        return
    c.assumptions.append(f"{s.ref} ({s.value or 'connector'}) has nothing plugged in; set {s.ref}.load=2A or "
                         f"{s.ref}.load=12ohm to put a load on it.")


def _part(p) -> dict:
    from .bind import _as_dict
    return _as_dict(p)


def _load(key: str, text: str) -> dict:
    return {"key": key, "kind": "load", "ratings": {}, "generic": False, "params": {}, "source": text}


def _sub(p):
    from .netlist import Subckt
    return Subckt(p.mname, list(p.ports), p.text)


def _unique_pins(pins):
    """Pin names made unique for a subcircuit, with duplicates on one net merged (several GND pins)."""
    seen: dict[str, str] = {}
    nodes, names = [], []
    for n, nm, net, _ in sorted(pins, key=lambda p: _num(p[0])):
        base = (nm if nm and nm != "~" else f"p{n}").replace(" ", "_")
        node = safe_node(net)
        if base.lower() in seen:
            if seen[base.lower()] == node:
                continue
            k = 2
            while f"{base}_{k}".lower() in seen:
                k += 1
            base = f"{base}_{k}"
        seen[base.lower()] = node
        nodes.append(node)
        names.append(base)
    return nodes, names


def _ic(c, s, pins, part, add, by_name, by_num):
    """A library IC with its pins mapped by name; op amps get one part per unit, sharing the power pins."""
    if part.kind == "mcu":
        nodes, names = _unique_pins(pins)
        add("x", nodes, [s.value or part.key], names, mpn=part.key)
        return
    ports = part.ports
    if part.kind in ("opamp", "comparator"):
        units = sorted({u for _, nm, _, u in pins if nm.lower() in ("+", "-", "~", "out", "in+", "in-")} or {1})
        power = {}
        for _, nm, net, u in pins:
            pt = ics.port_of(nm, ("vcc", "vee"))
            if pt:
                power[pt] = safe_node(net)
        for k, u in enumerate(units):
            mine = {ics.port_of(nm, ("inp", "inn", "out")): safe_node(net) for _, nm, net, uu in pins
                    if uu in (u, 0) and ics.port_of(nm, ("inp", "inn", "out"))}
            if len(units) == 1:
                mine = {ics.port_of(nm, ("inp", "inn", "out")): safe_node(net) for _, nm, net, _ in pins
                        if ics.port_of(nm, ("inp", "inn", "out"))}
            got = {**power, **{k_: v for k_, v in mine.items() if k_}}
            if "vee" not in got:
                got["vee"] = "0"
                c.assumptions.append(f"{s.ref}'s negative supply pin is not on the sheet; taken as ground.")
            missing = [p for p in ports if p not in got]
            if missing:
                c.assumptions.append(f"{s.ref} unit {u} has no {', '.join(missing)} pin on the sheet, so it is left out.")
                continue
            suffix = "ABCDEFGH"[k] if len(units) > 1 else ""
            add("x", [got[p] for p in ports], [s.value or part.key], list(ports), ref=s.ref + suffix,
                package=s.ref, mpn=part.key)
        return
    if part.kind == "opto" and not any(ics.port_of(nm, ports) for _, nm, _, _ in pins if nm not in ("~", "")):
        m = {"1": "a", "2": "k", "3": "e", "4": "c"}
        got = {m[str(n)]: safe_node(net) for n, _, net, _ in pins if str(n) in m}
    else:
        got = {}
        for n, nm, net, _ in pins:
            pt = ics.port_of(nm, ports)
            if pt and pt not in got:
                got[pt] = safe_node(net)
        if part.kind == "regulator" and "gnd" in ports and "gnd" not in got and "adj" in {
                ics.port_of(nm, ("adj",)) for _, nm, _, _ in pins}:
            got["gnd"] = next(safe_node(net) for _, nm, net, _ in pins if ics.port_of(nm, ("adj",)))
    for opt, tie in (("stat", None), ("ctrl", None), ("reset", "vcc")):     # pins a symbol may leave off
        if opt in ports and opt not in got:
            got[opt] = got.get(tie) if tie and tie in got else f"nc_{s.ref.lower()}_{opt}"
    missing = [p for p in ports if p not in got]
    if missing:
        nodes, names = _unique_pins(pins)
        add("x", nodes, [f"UNMAPPED_{part.key}"], names)
        c.assumptions.append(f"{s.ref} ({part.key}) has no pin named for {', '.join(missing)}, so it runs as a "
                             "generic IC.")
        return
    add("x", [got[p] for p in ports], [s.value or part.key], list(ports), mpn=part.key)


def _sim_fields(c, s, pins, add, by_num) -> bool:
    """KiCad's own simulation fields: Sim.Device/Sim.Name/Sim.Pins (KiCad 7+) or Spice_Primitive/Spice_Model."""
    f = s.fields
    dev = (f.get("sim.device") or f.get("spice_primitive") or "").strip().upper()
    if not dev:
        return False
    model = f.get("sim.name") or f.get("spice_model") or s.value
    seq = f.get("sim.pins") or f.get("spice_node_sequence") or ""
    nums = [p.split("=")[0] for p in seq.split()] if seq else [str(n) for n, _, _, _ in sorted(pins, key=lambda p:
                                                                                                 _num(p[0]))]
    nodes = [by_num.get(n, "0") for n in nums]
    letter = {"R": "r", "C": "c", "L": "l", "D": "d", "NPN": "q", "PNP": "q", "NMOS": "m", "PMOS": "m", "Q": "q",
              "M": "m", "V": "v", "I": "i", "SUBCKT": "x", "X": "x"}.get(dev)
    if letter is None:
        return False
    params = f.get("sim.params", "")
    if letter in "rcl":
        m = re.search(r"\b[rcl]\s*=\s*(\S+)", params, re.I)
        add(letter, nodes[:2], [m.group(1) if m else _num_text(s.value, "1k")], ["1", "2"])
    elif letter in "vi":
        m = re.search(r"\bdc\s*=\s*(\S+)", params, re.I)
        add(letter, nodes[:2], [m.group(1) if m else (s.value or "0")], ["+", "-"])
    else:
        pin_names = {"d": ["a", "k"], "q": ["c", "b", "e"], "m": ["d", "g", "s"]}.get(letter) or nums
        add(letter, nodes, [model], pin_names[:len(nodes)])
    lib = f.get("sim.library") or f.get("spice_lib_file")
    if lib:
        c.directives.append(f".include {lib}")
    return True


def power_rails(c: Circuit, params: dict) -> None:
    """A supply for every rail that nothing on the sheet drives, at the voltage its name gives (or the one set)."""
    driven = set()
    for e in c.elements:
        if e.letter in "vi" or e.meta.get("battery"):
            driven.add(e.nodes[0])
        if e.letter == "x" and "out" in e.pins and lookup(str(e.meta.get("mpn") or e.rest[0]) or "") is not None:
            p = lookup(str(e.meta.get("mpn") or e.rest[0]))
            if p and p.kind == "regulator":
                driven.add(e.nodes[e.pins.index("out")])
        if e.letter == "x" and "bat" in e.pins:
            driven.add(e.nodes[e.pins.index("bat")])
    users = {n for e in c.elements for n in e.nodes}
    for node, name in sorted(c.names.items()):
        if node == "0" or node in driven or node not in users:
            continue
        low = name.lstrip("/").lower()
        set_v = params.get(name) or params.get(low) or params.get(node)
        v = rail_volts(low)
        if v is None and low in NAMED_RAILS:
            v = NAMED_RAILS[low] if NAMED_RAILS[low] is not None else _supply(params)
            if v is None:
                v = 12.0
            if not set_v:
                c.assumptions.append(f"The {name} rail has no voltage in its name; assumed {v:g} V (set {name}=...).")
        if set_v:
            v = parse_spice_value(str(set_v).lower().rstrip("v"))
        if v is None:
            continue
        c.elements.append(Element(_name("v", "RAIL_" + node), "v", [node, "0"], [f"{v:g}"], ["+", "-"],
                                  meta={"ref": f"{name} supply", "note": f"{v:g} V rail"}))
        c.assumptions.append(f"Nothing on the sheet drives {name}, so it is held at {v:g} V as if supplied from "
                             "off the sheet.")


def _supply(params: dict) -> float | None:
    v = params.get("supply")
    if not v:
        return None
    try:
        return parse_spice_value(str(v).lower().rstrip("v"))
    except Exception:  # noqa: BLE001
        return None


def _num(x) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(x)) or 0)
    except ValueError:
        return 0.0


def _value(text: str) -> float | None:
    try:
        return parse_spice_value(_num_text(text, ""))
    except Exception:  # noqa: BLE001
        return None


def _num_text(text: str, default: str) -> str:
    """The value part of a KiCad value field: '10k 1%' is 10k, '100nF/50V' is 100n, '4k7' stays 4k7."""
    t = str(text or "").strip().replace("µ", "u").replace("μ", "u").replace("Ω", "").replace("ohm", "")
    m = re.match(r"^(\d+(?:\.\d+)?(?:meg|[pnumkgtrRMKG])?\d*)", t.replace(",", "."))
    if not m:
        return default
    v = m.group(1)
    v = re.sub(r"^(\d+)[rR](\d*)$", lambda x: f"{x.group(1)}.{x.group(2) or 0}", v)
    v = v.replace("M", "meg") if re.search(r"\dM", v) and "meg" not in v else v
    return v.replace("K", "k").replace("G", "g")


def _amps(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m?)a\b", str(text or ""), re.I)
    if m:
        return float(m.group(1)) * (1e-3 if m.group(2) else 1)
    m = re.search(r"(?:MF|MICROSMD|SMD|MF-[A-Z]+)(\d{3})", str(text or ""), re.I)
    return int(m.group(1)) / 100 if m else None
