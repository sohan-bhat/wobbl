# Copper crystal

What is copper's crystal structure and how stiff is it?

Wobbl takes fcc copper from its name, picks EMT (the right calculator for a metal) through ASE, and
fits an equation of state over nine volumes to get the lattice constant, density, bulk modulus and cohesive
energy. Other solids use GFN-FF or xTB instead.

```sh
./run.sh
```

| Result | Value | Measured |
|---|---|---|
| Lattice constant | 3.590 A | 3.61 A |
| Density | 9.12 g/cm3 | |
| Bulk modulus | 133 GPa | 137 GPa |
| Cohesive energy | 3.52 eV per atom | 3.49 eV |

`output/playback.json` holds the atoms, bonds and frames the viewer plays.
