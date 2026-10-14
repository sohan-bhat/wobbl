"""Validation cases for the field studies (GetDP): magnetics, electrostatics, board traces, antennas and sound.

Each one runs the same study a user gets and compares one number with an exact
or published answer. All run in well under a minute on a laptop.
"""
from __future__ import annotations

import math
from pathlib import Path

from .cases import Case

MU0 = 4e-7 * math.pi


def solenoid_field(out: Path):
    """An air core solenoid 20 diameters long. The center field is mu0 n I times the finite length factor
    L / sqrt(L^2 + D^2), averaged over the winding's thickness."""
    from ..fields.magnetic import run_magnetic

    o = run_magnetic("solenoid", {"core": "none", "coil_id": "20 mm", "coil_od": "21 mm", "length": "410 mm",
                                  "plunger_d": "10 mm", "turns": 1000, "current": "1 A", "stroke": "1 mm"}, out)
    return o.values["b_center_T"] * 1e3, o.values["b_exact_T"] * 1e3, {"confidence": o.confidence,
                                                                     "mu0_n_I_mT": MU0 * 1000 / 0.41 * 1e3}


def plate_capacitor(out: Path):
    """Two 100 mm discs 2 mm apart, 0.5 mm thick with square edges, against Kirchhoff's formula with its fringing
    and thickness terms."""
    from ..fields.electric import kirchhoff, run_electric

    o = run_electric("plates", {"diameter": "100 mm", "gap": "2 mm", "thickness": "0.5 mm", "edge": "square",
                                "voltage": "1000 V"}, out)
    return o.values["capacitance_F"] * 1e12, kirchhoff(0.05, 0.002, 0.0005) * 1e12, {"confidence": o.confidence}


def microstrip_z0(out: Path):
    """A 1 mm trace on 0.5 mm of er 4.4, copper 5 um thick, against Hammerstad and Jensen."""
    from ..fields.board import hammerstad_jensen, run_board

    o = run_board("microstrip", {"width": "1 mm", "height": "0.5 mm", "thickness": "0.005 mm", "er": "4.4"}, out)
    return o.values["z0_ohm"], hammerstad_jensen(1e-3, 0.5e-3, 5e-6, 4.4)[0], {"confidence": o.confidence}


def coax_z0(out: Path):
    """RG-58 proportions: 0.91 mm inner, 2.95 mm dielectric, polyethylene."""
    from ..fields.board import coax_z, run_board

    o = run_board("coax", {"inner_d": "0.91 mm", "outer_d": "2.95 mm", "er": "2.25"}, out)
    return o.values["z0_ohm"], coax_z(0.455e-3, 1.475e-3, 2.25), {"confidence": o.confidence}


def dipole_resistance(out: Path):
    """A thin dipole (diameter a thousandth of a wavelength) swept through its first resonance. Its input
    resistance at resonance should sit near the 73.1 ohm of the induced EMF theory for a thin half wave dipole."""
    from ..fields.antenna import run_antenna

    o = run_antenna("dipole", {"frequency": "1 GHz", "length": "150 mm", "diameter": "0.3 mm", "points": 7,
                               "span": 0.12}, out)
    z = o.values["z_resonance_ohm"]
    return (z[0] if z else float("nan")), 73.1, {"confidence": o.confidence,
                                                 "resonance_MHz": round(o.values["resonance_Hz"] / 1e6, 2)}


def room_modes(out: Path):
    """A 5 x 4 x 3 m box with rigid walls. The 15th mode against c / 2 sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)."""
    from ..fields.sound import box_modes, run_room

    o = run_room("room", {"size": "5x4x3", "fmax": "110 Hz"}, out)
    exact = [f for f, _ in box_modes((5.0, 4.0, 3.0), 110.0)]
    i = min(14, len(o.values["modes_Hz"]) - 1)
    return o.values["modes_Hz"][i], exact[i], {"confidence": o.confidence, "modes": len(o.values["modes_Hz"])}


def muffler_peak(out: Path):
    """An expansion chamber of area ratio 9 (50 mm pipes, 150 mm chamber, 300 mm long), swept around its first
    peak, against plane wave theory's 20 log10((m + 1/m) / 2) = 13.2 dB."""
    from ..fields.sound import C_AIR, run_muffler

    f0 = C_AIR / (4 * 0.3)
    o = run_muffler({"pipe_d": "50 mm", "chamber_d": "150 mm", "chamber_length": "300 mm",
                     "fmin": f"{0.85 * f0} Hz", "fmax": f"{1.15 * f0} Hz", "points": 13}, out)
    m = 9.0
    return max(o.values["tl_dB"]), 20 * math.log10((m + 1 / m) / 2), {"confidence": o.confidence}


FIELD_CASES = [
    Case("solenoid-field", "Long air core solenoid", "magnetic", "mu0 n I with the finite length factor",
         "flux density at the center", "mT", 0.01, True, solenoid_field),
    Case("plate-capacitor", "Parallel discs with fringing", "electric", "Kirchhoff's formula with thickness",
         "capacitance", "pF", 0.01, False, plate_capacitor),
    Case("microstrip-z0", "50 ohm microstrip on FR4", "board", "Hammerstad and Jensen (1980)",
         "characteristic impedance", "ohm", 0.01, False, microstrip_z0),
    Case("coax-z0", "RG-58 coax cross section", "board", "60 ln(b/a) / sqrt(er)", "characteristic impedance", "ohm",
         0.005, True, coax_z0),
    Case("dipole", "Thin half wave dipole at resonance", "antenna", "induced EMF theory, 73.1 ohm",
         "input resistance at resonance", "ohm", 0.05, False, dipole_resistance),
    Case("room-modes", "Rectangular room modes", "sound", "exact rigid box modes", "15th mode frequency", "Hz",
         0.005, True, room_modes),
    Case("muffler", "Expansion chamber muffler", "sound", "plane wave theory, 20 log((m + 1/m)/2)",
         "peak transmission loss", "dB", 0.03, False, muffler_peak),
]
