# Caffeine

What does caffeine's structure and IR spectrum look like?

Wobbl reads the name, builds the 3D structure with RDKit and relaxes it with GFN2-xTB. It then finds the
vibrations, the IR spectrum and the thermochemistry. Nothing in the run is specific to caffeine, any name,
formula or SMILES goes the same way.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Molecule | caffeine, C8H10N4O2, 194.19 g/mol |
| Strongest IR band | 1712 cm-1, the C=O stretch |
| Dipole | 3.98 D |
| HOMO-LUMO gap | 3.40 eV |
| Entropy at 298 K | 470.8 J/(mol K) |
| Vibrations | all 66 real, so the structure is a true minimum |

`output/structure/playback.json` and `output/spectrum/playback.json` hold the atoms, bonds and frames the viewer plays.
