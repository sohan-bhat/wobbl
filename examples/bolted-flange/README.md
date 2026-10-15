# Bolted flange

Do four M8 bolts tightened to 20 N m keep this flange closed under a 30 kN pull?

Two 14 mm steel flanges, 120 mm across with a 50 mm bore, are held face to face by four M8 class 8.8
bolts on a 90 mm circle, the way two lengths of pipe or a shaft and a hub are joined. The flanges are
separate bodies, so they press together where the bolts squeeze them and can open apart elsewhere. Wobbl
finds the bolts from the four holes that run through both flanges (the bore is too wide to be a bolt
hole), tightens each to the preload its torque gives, then pulls the top flange off the bottom one.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Verdict | holds, the flanges stay pressed together under the 30 kN pull |
| Preload | 12.5 kN in each bolt, from 20 N m with a nut factor of 0.2 |
| Worst bolt | 12.6 kN under the pull, 59% of the 21.2 kN proof load |
| Clamp left | 4.41 kN at the least clamped bolt (4.4 to 5.7 kN over the four), down from 12.5 kN |
| Margin to opening | VDI 2230 puts the opening load at 14.6 kN of pull per bolt against the 7.5 kN each bolt takes |
| Hand calculation | VDI 2230 leaves 6.09 kN of clamp and adds 1.09 kN to each bolt, the FEA adds 0.1 kN because the load enters the flange away from the bolts |
| Mesh convergence | worst bolt force 12,604 then 12,602 N, least clamp 4,669 then 4,413 N (5.8% apart, so confidence is medium) |
| Equilibrium | force and moment balance to 5e-6% and 1e-5% |
| Time | the two meshes of 3,936 and 5,464 nodes were predicted to take 649 s together and took 791 s on a busy machine |

The run writes `output/scene.wobbl`, the 3D view embedded in `output/report.html`. In it the bolts tighten and
the flanges press together, then the pull comes on and the clamp between them falls, increment by
increment. `output/playback.json` holds the same frames as plain arrays, with each bolt's force.
