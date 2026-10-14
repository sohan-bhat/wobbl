"""Write the KiCad schematics the circuit tests and examples read: python tests/make_kicad_fixtures.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kicad_writer import Sheet  # noqa: E402

OUT = Path(__file__).parent / "data" / "circuits"
R0805, C0805 = "Resistor_SMD:R_0805_2012Metric", "Capacitor_SMD:C_0805_2012Metric"


def led_battery() -> Sheet:
    s = Sheet("led_battery")
    s.part("Device:Battery", "BT1", "9V", (50, 60))
    s.part("Device:R", "R1", "220", (80, 50), footprint=R0805)
    s.part("Device:LED", "D1", "Red", (100, 70), angle=90, footprint="LED_SMD:LED_0805_2012Metric")
    s.wire_pins(("BT1", "1"), ("R1", "1"))
    s.net("LED_A", ("R1", "2"), ("D1", "2"))
    s.power("GND", ("BT1", "2"), ("D1", "1"))
    return s


def ldo_mcu_board() -> Sheet:
    """A microcontroller board's power tree: a 12 V jack into an L7805, an ATtiny85 on 5 V switching an LED and a
    MOSFET that drives a 12 V load on J2."""
    s = Sheet("ldo_mcu_board")
    s.part("Connector:Barrel_Jack", "J1", "12V", (30, 60))
    s.part("Regulator_Linear:L7805", "U1", "L7805", (70, 60), footprint="Package_TO_SOT_THT:TO-220-3_Vertical")
    s.part("Device:C_Polarized", "C1", "100u 25V", (55, 75), footprint="Capacitor_THT:CP_Radial_D6.3mm_P2.50mm")
    s.part("Device:C", "C2", "100n 50V", (90, 75), footprint=C0805)
    s.part("MCU_Microchip_ATtiny:ATtiny85-20PU", "U2", "ATtiny85-20PU", (130, 60))
    s.part("Device:C", "C3", "100n 16V", (115, 80), footprint=C0805)
    s.part("Device:R", "R1", "330", (170, 50), footprint=R0805)
    s.part("Device:LED", "D1", "Green", (185, 70), angle=90, footprint="LED_SMD:LED_0805_2012Metric")
    s.part("Device:R", "R2", "100", (165, 90), angle=90, footprint=R0805)
    s.part("Device:R", "R3", "10k", (180, 100), footprint=R0805)
    s.part("Device:Q_NMOS_GDS", "Q1", "IRLZ44N", (200, 95), footprint="Package_TO_SOT_THT:TO-220-3_Vertical")
    s.part("Connector_Generic:Conn_01x02", "J2", "LED strip", (230, 80))
    s.net("VIN", ("J1", "1"), ("U1", "1"), ("C1", "1"))
    s.power("+12V", ("J1", "1"))
    s.power("+5V", ("U1", "3"), ("C2", "1"), ("U2", "8"), ("C3", "1"))
    s.power("GND", ("J1", "2"), ("U1", "2"), ("C1", "2"), ("C2", "2"), ("U2", "4"), ("C3", "2"), ("D1", "1"),
            ("R3", "2"), ("Q1", "3"))
    s.net("LED", ("U2", "5"), ("R1", "1"))
    s.net("LED_K", ("R1", "2"), ("D1", "2"))
    s.net("GATE_DRV", ("U2", "6"), ("R2", "1"))
    s.net("GATE", ("R2", "2"), ("R3", "1"), ("Q1", "1"))
    s.net("STRIP-", ("Q1", "2"), ("J2", "2"))
    s.power("+12V", ("J2", "1"))
    return s


def blinker_555() -> Sheet:
    s = Sheet("blinker_555")
    s.part("Timer:NE555P", "U1", "NE555P", (100, 80), footprint="Package_DIP:DIP-8_W7.62mm")
    s.part("Device:R", "R1", "1k", (140, 50), footprint=R0805)
    s.part("Device:R", "R2", "68k", (140, 75), footprint=R0805)
    s.part("Device:C_Polarized", "C1", "10u 16V", (70, 110), footprint="Capacitor_SMD:CP_Elec_4x5.4")
    s.part("Device:C", "C2", "10n", (60, 80), footprint=C0805)
    s.part("Device:R", "R3", "470", (160, 100), angle=90, footprint=R0805)
    s.part("Device:LED", "D1", "Yellow", (180, 115), angle=90, footprint="LED_SMD:LED_0805_2012Metric")
    s.power("+9V", ("U1", "8"), ("U1", "4"), ("R1", "1"))
    s.power("GND", ("U1", "1"), ("C1", "2"), ("C2", "2"), ("D1", "1"))
    s.net("DIS", ("U1", "7"), ("R1", "2"), ("R2", "1"))
    s.net("TIMING", ("R2", "2"), ("U1", "6"), ("U1", "2"), ("C1", "1"))
    s.net("CV", ("U1", "5"), ("C2", "1"))
    s.net("OUT", ("U1", "3"), ("R3", "1"))
    s.net("LED_A", ("R3", "2"), ("D1", "2"))
    return s


def opamp_dual() -> Sheet:
    """An LM358 on 12 V: unit A amplifies a sensor divider by 11, unit B buffers a 2.5 V reference."""
    s = Sheet("opamp_dual")
    s.part("Amplifier_Operational:LM358", "U1", "LM358", (100, 70), footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    for ref, val, at in (("R1", "10k", (40, 50)), ("R2", "1k", (40, 80)), ("R3", "100k", (100, 40)),
                         ("R4", "10k", (80, 100)), ("R5", "10k", (150, 40)), ("R6", "10k", (150, 90))):
        s.part("Device:R", ref, val, at, footprint=R0805)
    s.power("+12V", ("U1", "8"), ("R1", "1"), ("R5", "1"))
    s.power("GND", ("U1", "4"), ("R2", "2"), ("R4", "2"), ("R6", "2"))
    s.net("SENSE", ("R1", "2"), ("R2", "1"), ("U1", "3"))
    s.net("FB", ("U1", "2"), ("R3", "2"), ("R4", "1"))
    s.net("AMP_OUT", ("U1", "1"), ("R3", "1"))
    s.net("REF", ("R5", "2"), ("R6", "1"), ("U1", "5"))
    s.net("REF_BUF", ("U1", "6"), ("U1", "7"))
    return s


def charger_usb() -> Sheet:
    """An MCP73831 charging a Li-ion cell, fed from a +5V rail that comes from off the sheet."""
    s = Sheet("charger_usb")
    s.part("Battery_Management:MCP73831-2-OT", "U1", "MCP73831-2-OT", (100, 70),
           footprint="Package_TO_SOT_SMD:SOT-23-5")
    s.part("Device:R", "R1", "2k", (80, 90), footprint=R0805)
    s.part("Device:R", "R2", "1k", (130, 50), footprint=R0805)
    s.part("Device:LED", "D1", "Red", (150, 60), angle=90, footprint="LED_SMD:LED_0805_2012Metric")
    s.part("Device:C", "C1", "4.7u 10V", (70, 60), footprint=C0805)
    s.part("Device:C", "C2", "4.7u 10V", (130, 90), footprint=C0805)
    s.part("Device:Battery", "BT1", "LiPo 3.7V", (160, 95))
    s.power("+5V", ("U1", "4"), ("C1", "1"), ("R2", "1"))
    s.power("GND", ("U1", "2"), ("R1", "2"), ("C1", "2"), ("C2", "2"), ("BT1", "2"))
    s.net("PROG", ("U1", "5"), ("R1", "1"))
    s.net("VBAT", ("U1", "3"), ("C2", "1"), ("BT1", "1"))
    s.net("STAT_LED", ("R2", "2"), ("D1", "2"))
    s.net("STAT", ("D1", "1"), ("U1", "1"))
    return s


def relay_driver() -> Sheet:
    """A 2N2222 switching a 12 V coil (drawn as its inductance and resistance) with a flyback diode, its base
    driven by an ATtiny85 on a 5 V rail."""
    s = Sheet("relay_driver")
    s.part("MCU_Microchip_ATtiny:ATtiny85-20PU", "U1", "ATtiny85-20PU", (60, 70))
    s.part("Device:R", "R1", "1k", (100, 60), angle=90, footprint=R0805)
    s.part("Device:Q_NPN_BCE", "Q1", "2N2222A", (130, 75), footprint="Package_TO_SOT_THT:TO-92_Inline")
    s.part("Device:L", "L1", "1.2", (140, 40))
    s.part("Device:R", "R2", "400", (140, 20))
    s.part("Device:D", "D1", "1N4148", (165, 30), angle=90, footprint="Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal")
    s.part("Device:C", "C1", "100n 16V", (40, 90), footprint=C0805)
    s.power("+5V", ("U1", "8"), ("C1", "1"))
    s.power("+12V", ("R2", "1"), ("D1", "1"))
    s.power("GND", ("U1", "4"), ("Q1", "3"), ("C1", "2"))
    s.net("DRIVE", ("U1", "5"), ("R1", "1"))
    s.net("BASE", ("R1", "2"), ("Q1", "1"))
    s.net("COIL_R", ("R2", "2"), ("L1", "1"))
    s.net("COIL", ("L1", "2"), ("Q1", "2"), ("D1", "2"))
    return s


def led_matrix() -> Sheet:
    """Six blue 0805 LEDs on a 5 V rail, each with its own 68 ohm 0603 resistor."""
    s = Sheet("led_matrix")
    for k in range(6):
        s.part("Device:R", f"R{k + 1}", "68", (40 + 20 * k, 50), footprint="Resistor_SMD:R_0603_1608Metric")
        s.part("Device:LED", f"D{k + 1}", "Blue", (40 + 20 * k, 80), angle=90,
               footprint="LED_SMD:LED_0805_2012Metric")
        s.power("+5V", (f"R{k + 1}", "1"))
        s.net(f"LED{k + 1}", (f"R{k + 1}", "2"), (f"D{k + 1}", "2"))
        s.power("GND", (f"D{k + 1}", "1"))
    return s


BOARDS = {"led_battery": led_battery, "ldo_mcu_board": ldo_mcu_board, "blinker_555": blinker_555,
          "opamp_dual": opamp_dual, "charger_usb": charger_usb, "relay_driver": relay_driver, "led_matrix": led_matrix}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in BOARDS.items():
        (OUT / f"{name}.kicad_sch").write_text(fn().text())
        print(OUT / f"{name}.kicad_sch")
