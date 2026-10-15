# Cast pulley

Will the hub of this sand cast aluminum pulley shrink and leave porosity?

A belt pulley with a 120 mm rim, a 6 mm web and a 40 mm hub with a 14 mm bore, poured in A356 at 720 C into bonded
silica sand and lying flat. The thick hub is the last part to freeze, so it needs a riser. None is given, so Wobbl
places one over the hot spot by the modulus rule and checks it.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Verdict | MARGINAL, sound with no shrinkage pockets, medium confidence |
| Freezing | 2.2 min for the casting, 2.24 min for the riser |
| Porosity | at most 0.466% by the Niyama criterion, 5.8% of the casting over 0.1% |
| Mesh convergence | the time to freeze settled within 2.1% over two meshes |
| Hand calculation | Chvorinov's rule gives 1.31 min from one modulus and the solve gives 2.2 min (1.68 times), a spread a shape this far from compact is expected to have |

The riser's modulus is 1.5 times the casting's, yet it freezes only just after the casting because it sits on the hub
and feeds it heat. A wider riser or a chill under the hub would widen the margin.

Open `output/simulation.html` to watch the freezing front move through the metal and the sand, played from `output/scene.wobbl`.
