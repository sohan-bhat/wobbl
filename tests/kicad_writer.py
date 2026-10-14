"""Write small KiCad 7 schematics for tests and examples: real .kicad_sch files with library symbols, placed and
rotated parts, wires, labels and power symbols, the way eeschema saves them.

    s = Sheet("blinker")
    s.part("Device:R", "R1", "330", (100, 60), footprint="Resistor_SMD:R_0805_2012Metric")
    s.net("VOUT", ("R1", "2"), ("D1", "2"))       # a label on a short wire from each pin
    s.power("GND", ("D1", "1"))                   # a GND power symbol on each pin
    s.wire_pins(("R1", "1"), ("U1", "3"))          # a straight wire between two pins
    Path("blinker.kicad_sch").write_text(s.text())
"""
from __future__ import annotations

import math
import uuid

from wobbl.circuits.kicad import transform

F = '(effects (font (size 1.27 1.27)))'
FH = '(effects (font (size 1.27 1.27)) hide)'

# library symbol: (reference letter, units: {unit: [(number, name, type, x, y, angle, hidden)]}, power symbol)
TWO_V = {1: [("1", "~", "passive", 0, 3.81, 270, False), ("2", "~", "passive", 0, -3.81, 90, False)]}
DIODE = {1: [("1", "K", "passive", -3.81, 0, 0, False), ("2", "A", "passive", 3.81, 0, 180, False)]}
BJT = {1: [("1", "B", "input", -5.08, 0, 0, False), ("2", "C", "passive", 2.54, 5.08, 270, False),
           ("3", "E", "passive", 2.54, -5.08, 90, False)]}
FET = {1: [("1", "G", "input", -5.08, 0, 0, False), ("2", "D", "passive", 2.54, 5.08, 270, False),
           ("3", "S", "passive", 2.54, -5.08, 90, False)]}
LIBS = {
    "Device:R": ("R", TWO_V, False), "Device:C": ("C", TWO_V, False), "Device:C_Polarized": ("C", TWO_V, False),
    "Device:L": ("L", TWO_V, False), "Device:LED": ("D", DIODE, False), "Device:D": ("D", DIODE, False),
    "Device:D_Schottky": ("D", DIODE, False), "Device:D_Zener": ("D", DIODE, False),
    "Device:Q_NPN_BCE": ("Q", BJT, False), "Device:Q_PNP_BCE": ("Q", BJT, False),
    "Device:Q_NMOS_GDS": ("Q", FET, False), "Device:Q_PMOS_GDS": ("Q", FET, False),
    "Device:Battery": ("BT", {1: [("1", "+", "passive", 0, 5.08, 270, False),
                                  ("2", "-", "passive", 0, -5.08, 90, False)]}, False),
    "Device:Fuse": ("F", TWO_V, False),
    "Connector:Barrel_Jack": ("J", {1: [("1", "~", "passive", 5.08, 2.54, 180, False),
                                        ("2", "~", "passive", 5.08, -2.54, 180, False)]}, False),
    "Connector_Generic:Conn_01x02": ("J", {1: [("1", "Pin_1", "passive", -5.08, 0, 0, False),
                                               ("2", "Pin_2", "passive", -5.08, -2.54, 0, False)]}, False),
    "Regulator_Linear:AMS1117-3.3": ("U", {1: [("1", "GND", "power_in", 0, -7.62, 90, False),
                                               ("2", "VO", "power_out", 7.62, 0, 180, False),
                                               ("3", "VI", "power_in", -7.62, 0, 0, False)]}, False),
    "Regulator_Linear:L7805": ("U", {1: [("1", "IN", "power_in", -7.62, 0, 0, False),
                                         ("2", "GND", "power_in", 0, -7.62, 90, False),
                                         ("3", "OUT", "power_out", 7.62, 0, 180, False)]}, False),
    "Timer:NE555P": ("U", {1: [("1", "GND", "power_in", 0, -12.7, 90, False),
                               ("2", "TR", "input", -12.7, 5.08, 0, False),
                               ("3", "Q", "output", 12.7, 5.08, 180, False),
                               ("4", "R", "input", -12.7, -5.08, 0, False),
                               ("5", "CV", "input", -12.7, 0, 0, False),
                               ("6", "THR", "input", 12.7, 0, 180, False),
                               ("7", "DIS", "input", 12.7, -5.08, 180, False),
                               ("8", "VCC", "power_in", 0, 12.7, 270, False)]}, False),
    "Amplifier_Operational:LM358": ("U", {
        1: [("1", "~", "output", 7.62, 0, 180, False), ("2", "-", "input", -7.62, -2.54, 0, False),
            ("3", "+", "input", -7.62, 2.54, 0, False)],
        2: [("5", "+", "input", -7.62, 2.54, 0, False), ("6", "-", "input", -7.62, -2.54, 0, False),
            ("7", "~", "output", 7.62, 0, 180, False)],
        3: [("4", "V-", "power_in", -2.54, -7.62, 90, False), ("8", "V+", "power_in", -2.54, 7.62, 270, False)]},
        False),
    "MCU_Microchip_ATtiny:ATtiny85-20PU": ("U", {1: [
        ("1", "PB5", "bidirectional", -12.7, -5.08, 0, False), ("2", "PB3", "bidirectional", 12.7, -2.54, 180, False),
        ("3", "PB4", "bidirectional", 12.7, -5.08, 180, False), ("4", "GND", "power_in", 0, -12.7, 90, False),
        ("5", "PB0", "bidirectional", 12.7, 5.08, 180, False), ("6", "PB1", "bidirectional", 12.7, 2.54, 180, False),
        ("7", "PB2", "bidirectional", 12.7, 0, 180, False), ("8", "VCC", "power_in", 0, 12.7, 270, False)]}, False),
    "Battery_Management:MCP73831-2-OT": ("U", {1: [
        ("1", "STAT", "output", 10.16, 2.54, 180, False), ("2", "VSS", "power_in", 0, -10.16, 90, False),
        ("3", "VBAT", "power_out", 10.16, 0, 180, False), ("4", "VDD", "power_in", 0, 10.16, 270, False),
        ("5", "PROG", "passive", -10.16, 0, 0, False)]}, False),
}


def _power_lib(value: str) -> tuple:
    angle = 270 if value.upper().startswith("GND") else 90
    return ("#PWR", {1: [("1", value, "power_in", 0, 0, angle, True)]}, True)


def _u() -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"wobbl-test-{next(_COUNTER)}"))


def _count():
    k = 0
    while True:
        k += 1
        yield k


_COUNTER = _count()


class Sheet:
    def __init__(self, name: str):
        self.name = name
        self.root = _u()
        self.libs: dict[str, tuple] = {}
        self.items: list[str] = []
        self.pins: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]] = {}
        self.npwr = 0

    def part(self, lib: str, ref: str, value: str, at: tuple[float, float], angle: float = 0, mirror: str | None = None,
             footprint: str = "", fields: dict | None = None, units: tuple[int, ...] | None = None,
             spacing: float = 25.4):
        letter, unit_pins, power = LIBS[lib] if lib in LIBS else _power_lib(value)
        self.libs[lib] = (letter, unit_pins, power)
        for k, unit in enumerate(units or tuple(unit_pins)):
            x, y = at[0] + k * spacing, at[1]
            t = transform(angle, mirror)
            props = [("Reference", ref), ("Value", value), ("Footprint", footprint), ("Datasheet", "~"),
                     *((fields or {}).items())]
            body = [f'(symbol (lib_id "{lib}") (at {x:g} {y:g} {angle:g})' + (f" (mirror {mirror})" if mirror else "")
                    + f' (unit {unit}) (in_bom yes) (on_board yes) (dnp no) (uuid "{_u()}")']
            for i, (k_, v) in enumerate(props):
                body.append(f'  (property "{k_}" "{v}" (at {x:g} {y - 5 - i:g} 0) {F if i < 2 else FH})')
            for num, name, typ, px, py, pa, hidden in unit_pins[unit]:
                sx, sy = x + t[0] * px + t[1] * py, y + t[2] * px + t[3] * py
                ox, oy = math.cos(math.radians(pa + 180)), math.sin(math.radians(pa + 180))
                dx, dy = t[0] * ox + t[1] * oy, t[2] * ox + t[3] * oy
                self.pins[(ref, num)] = ((round(sx, 3), round(sy, 3)), (round(dx), round(dy)))
                body.append(f'  (pin "{num}" (uuid "{_u()}"))')
            body.append(f'  (instances (project "{self.name}" (path "/{self.root}" (reference "{ref}") '
                        f'(unit {unit}))))\n)')
            self.items.append("\n".join(body))

    def _stub(self, pin: tuple[str, str]) -> tuple[float, float]:
        (x, y), (dx, dy) = self.pins[pin]
        end = (round(x + 2.54 * dx, 3), round(y + 2.54 * dy, 3))
        self.wire((x, y), end)
        return end

    def wire(self, a, b):
        self.items.append(f'(wire (pts (xy {a[0]:g} {a[1]:g}) (xy {b[0]:g} {b[1]:g})) (stroke (width 0) '
                          f'(type default)) (uuid "{_u()}"))')

    def wire_pins(self, a: tuple[str, str], b: tuple[str, str]):
        self.wire(self.pins[a][0], self.pins[b][0])

    def net(self, name: str, *pins: tuple[str, str], kind: str = "label"):
        for p in pins:
            x, y = self._stub(p)
            self.items.append(f'({kind} "{name}" (at {x:g} {y:g} 0) (fields_autoplaced) {F} (uuid "{_u()}"))')

    def power(self, value: str, *pins: tuple[str, str]):
        for p in pins:
            x, y = self._stub(p)
            self.npwr += 1
            lib = f"power:{value}"
            self.libs[lib] = _power_lib(value)
            self.items.append(
                f'(symbol (lib_id "{lib}") (at {x:g} {y:g} 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) '
                f'(uuid "{_u()}")\n  (property "Reference" "#PWR0{self.npwr}" (at {x:g} {y:g} 0) {FH})\n'
                f'  (property "Value" "{value}" (at {x:g} {y - 3:g} 0) {F})\n  (pin "1" (uuid "{_u()}"))\n'
                f'  (instances (project "{self.name}" (path "/{self.root}" (reference "#PWR0{self.npwr}") '
                f'(unit 1))))\n)')

    def junction(self, at):
        self.items.append(f'(junction (at {at[0]:g} {at[1]:g}) (diameter 0) (color 0 0 0 0) (uuid "{_u()}"))')

    def text(self) -> str:
        libs = []
        for lib, (letter, units, power) in self.libs.items():
            base = lib.split(":")[1]
            out = [f'(symbol "{lib}"' + (" (power)" if power else "") + ' (pin_names (offset 0)) (in_bom yes) '
                   '(on_board yes)', f'  (property "Reference" "{letter}" (at 0 0 0) {F})',
                   f'  (property "Value" "{base}" (at 0 0 0) {F})',
                   f'  (symbol "{base}_0_1" (rectangle (start -2.54 -2.54) (end 2.54 2.54) (stroke (width 0.254) '
                   '(type default)) (fill (type none))))']
            for unit, pins in units.items():
                out.append(f'  (symbol "{base}_{unit}_1"')
                for num, name, typ, px, py, pa, hidden in pins:
                    out.append(f'    (pin {typ} line (at {px:g} {py:g} {pa:g}) (length 2.54)' + (" hide" if hidden
                                                                                                  else "") +
                               f' (name "{name}" {F}) (number "{num}" {F}))')
                out.append("  )")
            libs.append("\n".join(out) + "\n)")
        return ("(kicad_sch (version 20231120) (generator \"eeschema\") (generator_version \"8.0\")\n"
                f'(uuid "{self.root}")\n(paper "A4")\n(lib_symbols\n' + "\n".join(libs) + "\n)\n" +
                "\n".join(self.items) + "\n(sheet_instances (path \"/\" (page \"1\")))\n)\n")
