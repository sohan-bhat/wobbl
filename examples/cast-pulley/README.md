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

Open `output/simulation.html` for the casting cut open through its last spot to freeze, with a slider over the
riser's diameter from 12.5 to 40 mm that opens at the 25 mm Wobbl placed. The riser's freezing time scales with the
square of its size by Chvorinov's rule, against the casting's own solved times. The view reads MARGINAL at 25 mm,
FAILS below about 24.8 mm where the riser freezes before the hub, and the worst porosity climbs from the solve's
0.466% to 4.3% at 12.5 mm as the hub shrinks with nothing left to feed it.
