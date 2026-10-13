# Heat sink: a 15 W chip without a fan

An extruded 6063 aluminum heat sink, 60 x 60 mm base with nine 30 mm fins,
sitting in a 40 C cabinet with no fan. A 15 W chip touches the middle 25 x 25 mm
of the base. Still air takes heat away at about 6 W/(m2 K) and the anodized
surface radiates (emissivity 0.85).

```sh
./run.sh
```

| Result | Value |
|---|---|
| Hottest point (under the chip) | 70.3 C, below the 85 C limit |
| Coolest point (fin tips) | 67.1 C |
| Mesh convergence | rise of 30.3, 30.2, 30.3 C over three meshes |
| Energy balance | 15 W in, 15 W out through convection and radiation |
| Hand calculation | a uniform skin shedding 15 W by convection and radiation sits at 68.1 C; the FEA skin averages 68.1 C |

The aluminum spreads heat so well (Biot number 3e-5) that the whole sink is
within 3 C; the margin is set by how much surface the fins give the air, not by
conduction. Doubling the cooling (a small fan) would roughly halve the 30 C rise.
