---
title: "Units"
description: "The units Wobbl reads for geometry, loads, heat and circuits, and how it prints results in imperial units."
lead: "Geometry is always in millimeters, and everything else takes whatever common unit you write."
order: 6
---

## Geometry

Files are read in millimeters, including STL, which carries no units at all. A part drawn in
inches needs `--scale 25.4` and one in meters needs `--scale 1000`. If the bounding box in
`wobbl inspect` looks 25.4 or 1000 times off, that is the cause, and a part under 1 mm or over
20 m across gets a note saying so. Points in loads and selectors, like `at 0,-30,66` or
`within 15 of 50,-25,35`, are in millimeters too.

## Loads and stresses

| Quantity | Units |
|---|---|
| force | `N`, `kN`, `mN`, `lbf` (also `lb`, `lbs`, `pound`), `kip`, `kgf` (also `kg`), `gf` |
| pressure and stress | `Pa`, `kPa`, `MPa`, `GPa`, `N/mm2`, `psi`, `ksi`, `bar`, `mbar`, `atm`, `psf` |
| moment | `N*mm`, `N*m` (also `Nm`, `N.m`, `N-m`), `kN*m`, `in-lb` (also `lbf*in`), `ft-lb` (also `lbf*ft`), `oz-in` |
| acceleration | `g`, `m/s2`, `mm/s2`, `ft/s2` |

When any load uses an imperial unit, the results print stresses in psi or ksi and lengths in inches, with the metric values in
parentheses next to them.

## Heat

Power takes `W`, `mW` or `kW`, films take `W/m2K` (also written `W/(m2K)` or `W/m2/K`), and
durations take `s`, `min` or `h`. Temperatures take `C`, `K` or `F`, and a bare number is
read as Celsius.

## Circuits

Circuit values stay in plain SI and accept SPICE style suffixes, so `10k` is 10 kilohm,
`4k7` is 4.7 kilohm, `22u` is 22 microfarads and `2.2meg` is 2.2 megohm. Units written after
the number, like `5V`, `10uH` or `500kHz`, are fine too.

## What Wobbl uses inside

Every quantity is converted once, on the way in, to one consistent system of millimeters,
newtons, MPa, tonnes, seconds, milliwatts and degrees Celsius, so the solvers never see mixed
units. The results files keep that system and name the unit in each key, as in
`governing_MPa` and `max_disp_mm`.
