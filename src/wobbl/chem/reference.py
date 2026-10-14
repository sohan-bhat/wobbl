"""Measured reference data that runs are checked against, and what each method is known to get right.

Geometries, vibrational fundamentals and dipoles are NIST's Computational Chemistry Comparison and Benchmark
Database (CCCBDB, cccbdb.nist.gov) experimental values. Heats of formation are Active Thermochemical Tables
(ATcT, atct.anl.gov, version 1.130) at 298.15 K for gases. Each method's accuracy is the typical error reported
for it against such data, stated in plain units so a result can say how far to trust each number.
"""
from __future__ import annotations

# bonds as (element, element, angstrom), angles as (element, center, element, degrees), fundamentals in cm-1 with
# degenerate modes repeated, dipole in debye
MOLECULES = {
    "H2O": {"bonds": [("O", "H", 0.958)], "angles": [("H", "O", "H", 104.5)], "freqs": [1595, 3657, 3756],
            "dipole": 1.855},
    "CO2": {"bonds": [("C", "O", 1.160)], "angles": [("O", "C", "O", 180.0)], "freqs": [667, 667, 1333, 2349],
            "dipole": 0.0},
    "CH4": {"bonds": [("C", "H", 1.087)], "angles": [("H", "C", "H", 109.47)],
            "freqs": [1306, 1306, 1306, 1534, 1534, 2917, 3019, 3019, 3019], "dipole": 0.0},
    "H3N": {"bonds": [("N", "H", 1.012)], "angles": [("H", "N", "H", 106.7)],
            "freqs": [950, 1627, 1627, 3337, 3444, 3444], "dipole": 1.47},
    "CO": {"bonds": [("C", "O", 1.128)], "freqs": [2143], "dipole": 0.11},
    "N2": {"bonds": [("N", "N", 1.098)], "freqs": [2330], "dipole": 0.0},
    "H2": {"bonds": [("H", "H", 0.741)], "freqs": [4161], "dipole": 0.0},
    "O2": {"bonds": [("O", "O", 1.208)], "freqs": [1556], "dipole": 0.0},
    "FH": {"bonds": [("H", "F", 0.917)], "freqs": [3962], "dipole": 1.82},
    "ClH": {"bonds": [("H", "Cl", 1.275)], "freqs": [2886], "dipole": 1.11},
    "CHN": {"bonds": [("C", "H", 1.064), ("C", "N", 1.156)], "freqs": [712, 712, 2097, 3311], "dipole": 2.98},
    "C2H2": {"bonds": [("C", "C", 1.203), ("C", "H", 1.063)], "freqs": [612, 612, 730, 730, 1974, 3289, 3374],
             "dipole": 0.0},
    "C2H4": {"bonds": [("C", "C", 1.339), ("C", "H", 1.086)], "angles": [("H", "C", "H", 117.4)], "dipole": 0.0},
    "C2H6": {"bonds": [("C", "C", 1.535), ("C", "H", 1.094)], "dipole": 0.0},
    "C6H6": {"bonds": [("C", "C", 1.397), ("C", "H", 1.084)], "dipole": 0.0},
    "CH2O": {"bonds": [("C", "O", 1.205), ("C", "H", 1.111)], "angles": [("H", "C", "H", 116.1)], "dipole": 2.33},
    "O2S": {"bonds": [("S", "O", 1.431)], "angles": [("O", "S", "O", 119.3)], "freqs": [518, 1151, 1362],
            "dipole": 1.63},
    "H2S": {"bonds": [("S", "H", 1.336)], "angles": [("H", "S", "H", 92.1)], "freqs": [1183, 2615, 2626],
            "dipole": 0.97},
    "O3": {"bonds": [("O", "O", 1.272)], "angles": [("O", "O", "O", 116.8)], "dipole": 0.53},
    "CH4O": {"bonds": [("C", "O", 1.427)], "dipole": 1.70},
}

# ATcT 1.130 heats of formation at 298.15 K, gas, kJ/mol, by Hill formula
HEATS_OF_FORMATION = {
    "H2O": -241.83, "CO2": -393.47, "CO": -110.52, "CH4": -74.52, "H3N": -45.56, "C2H6": -83.98, "C2H4": 52.39,
    "C2H2": 228.28, "CH4O": -200.71, "CH2O": -109.23, "CHN": 129.37, "NO": 91.12, "N2O": 82.03, "ClH": -92.17,
    "FH": -272.68, "C6H6": 82.9, "H2O2": -135.46, "O2S": -296.84, "O3": 141.73, "C3H8": -104.94,
    "C2H6O": -234.57, "C2H4O2": -432.3, "CH2O2": -378.4, "H2S": -20.13, "C3H6O": -216.4, "C2H4O": -165.5,
}

# typical errors, in plain units
ACCURACY = {
    "GFN2-xTB": {"bond": 0.02, "angle": 2.5, "freq": 0.08, "dipole": 0.4, "reaction": 25.0, "barrier": 30.0,
                 "solvation": 8.0, "gap": "a trend only, often off by over 1 eV"},
    "GFN1-xTB": {"bond": 0.03, "angle": 3.0, "freq": 0.10, "dipole": 0.5, "reaction": 30.0, "lattice": 0.03,
                 "bulk": 0.30},
    "B3LYP/def2-SVP": {"bond": 0.012, "angle": 1.2, "freq": 0.04, "dipole": 0.2, "reaction": 15.0,
                       "gap": "about 1 eV smaller than the fundamental gap"},
    "B3LYP/def2-TZVP": {"bond": 0.008, "angle": 0.8, "freq": 0.03, "dipole": 0.1, "reaction": 12.0,
                        "gap": "about 1 eV smaller than the fundamental gap"},
    "EMT": {"lattice": 0.015, "bulk": 0.20, "cohesive": 0.15},
    "PBE/GTH-SZV": {"lattice": 0.03, "bulk": 0.25, "cohesive": 0.30},
    "PBE/GTH-DZVP": {"lattice": 0.015, "bulk": 0.15, "cohesive": 0.15},
    "Lennard-Jones": {"lattice": 0.02, "cohesive": 0.10},
    "Born-Mayer": {"lattice": 0.02, "bulk": 0.30},
    "tables": {"formation": 1.0},
}
# harmonic frequencies sit above measured fundamentals, by these factors on average (CCCBDB)
FREQ_SCALE = {"B3LYP/def2-SVP": 0.965, "B3LYP/def2-TZVP": 0.967, "GFN2-xTB": 1.0, "GFN1-xTB": 1.0}


def molecule(formula: str) -> dict | None:
    return MOLECULES.get(formula)
