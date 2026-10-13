---
title: "Materials"
description: "The 23 sourced materials built into Wobbl, custom material properties and how printed parts are judged."
lead: "Pick a material by its key, or give your own properties, and Wobbl knows how to judge it."
order: 4
wide: true
---

```sh
wobbl run part.step -m 6061-t6 ...
wobbl materials
```

Names are forgiving, so `aluminum`, `6061`, `stainless`, `polycarbonate` and `pla+` all find
the right entry, and a typo gets a suggestion. Each material carries its source in the
assumptions of every result.

## The built in materials

| Key | Material | Made | Limit (MPa) | Judged by | Default target |
|---|---|---|---|---|---|
| `pla` | PLA, 3D printed | printed | 50 | tensile strength | 2.5 |
| `petg` | PETG, 3D printed | printed | 47 | yield | 2.5 |
| `abs` | ABS, 3D printed | printed | 33 | yield | 2.5 |
| `asa` | ASA, 3D printed | printed | 35 | yield | 2.5 |
| `nylon` | Nylon PA6, conditioned | printed | 50 | yield | 2.5 |
| `pa12` | Nylon PA12, SLS or MJF | printed | 48 | yield | 2 |
| `pc` | Polycarbonate | molded | 62 | yield | 2 |
| `hdpe` | HDPE | molded | 26 | yield | 2.5 |
| `6061-t6` | Aluminum 6061-T6 | wrought | 276 | yield | 1.5 |
| `7075-t6` | Aluminum 7075-T6 | wrought | 503 | yield | 1.5 |
| `6063-t5` | Aluminum 6063-T5 | wrought | 145 | yield | 1.5 |
| `5052-h32` | Aluminum 5052-H32 | wrought | 193 | yield | 1.5 |
| `a356-t6` | Aluminum A356-T6, sand cast | cast | 165 | yield | 2 |
| `mild-steel` | Mild steel, ASTM A36 | wrought | 250 | yield | 1.5 |
| `4140` | Steel 4140, quenched and tempered | wrought | 655 | yield | 1.5 |
| `ss304` | Stainless steel 304, annealed | wrought | 215 | yield | 1.5 |
| `ti-6al-4v` | Titanium Ti-6Al-4V | wrought | 880 | yield | 1.5 |
| `copper` | Copper C110 | wrought | 69 | yield | 1.5 |
| `brass` | Brass C360 | wrought | 310 | yield | 1.5 |
| `wood` | Douglas fir, along the grain | natural | 50 | tensile strength | 3 |
| `plywood` | Birch plywood | natural | 35 | tensile strength | 3 |
| `fr4` | FR4 PCB laminate, in plane | laminated | 300 | tensile strength | 2 |
| `li-ion-cell` | Li-ion cell, lumped | | 10 | tensile strength | 1.5 |

Materials judged by yield use von Mises stress, and the ones judged by tensile strength use the
largest principal stress, which is how brittle materials crack. `wobbl materials` also lists
stiffness, density, thermal conductivity and the service temperature that heat studies use as
their default limit.

## Your own values

Give any properties after `custom:`, in MPa and kg/m3.

```text Flag
-m "custom:E=72400,nu=0.33,yield=165,process=cast"
```

The keys are `E`, `nu`, `yield` (or `sy`), `uts`, `density` (or `rho`), `k` for conductivity,
`cp` for specific heat, `cte` and `tmax` for the service temperature, plus `process` and
`ductile`. The process says how the part is made, one of printed, molded, cast, wrought,
natural or laminated, and it sets the caveats the result lists along with the layer strength
for prints. Anything you leave out comes from the library material made the same way that is
closest in stiffness, so a custom casting gets a casting's density. Custom materials are
judged by yield unless you add `ductile=0`.

To change one property of a library material, name it first.

```text Flag
-m "pla:yield=40"
```

## Printed parts

Printed plastics are treated as solid, isotropic material at 100% infill, loaded in the plane
of the layers. Pulling the layers apart is weaker, from 60% of the in plane strength for PLA,
ABS and ASA to 90% for PA12, and `--print-direction z` tells Wobbl which way the layers stack
so that it judges the pull across them against that weaker strength. Without it, the result
says how much weaker the part is across its layers and suggests passing the direction.
