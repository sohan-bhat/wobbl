# Motor mount fatigue

Does the cast motor mount last ten years of starts and running without a fatigue crack?

This reads the finished strength result of the [cast-motor-mount](../cast-motor-mount) example, a sand
cast aluminum mount whose peak of 124 MPa under the 2 kN belt pull sits on the 4 mm inside fillet at
the rib tip. A static safety factor of 1.33 says nothing about how often that load comes and goes,
so this run gives it a duty cycle for each year of service.

- 3650 starts, ten a day, each taking the belt pull from nothing to the full 2 kN and back.
- 100 million turns of running, each swinging the pull between 50% and 70% of 2 kN.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Verdict | MARGINAL, a factor of 1.03 on stress against the target of 2 for a casting |
| Damage over ten years | 62.7% of the fatigue life by Miner's rule, about 16 years before a crack starts |
| Starts | 62 MPa either side of 62 MPa, a life of 58,800 cycles, almost all of the damage |
| Running | 12 MPa either side of 74 MPa, a life of 1.7e11 cycles, 0.06% of the life each year |
| S-N line | 42.7 MPa at 500 million cycles for this casting, estimated from sand cast A356-T6 |
| Mean stress | Goodman 1.03, Gerber 1.29 and Soderberg 1.03, so the verdict does not hang on the choice |
| Static check | the highest stress in a cycle stays under yield, so the stress life method applies |

The running vibration barely counts, because its swing sits far down the S-N line. The starts do
the damage, each one a full cycle from zero to 124 MPa at the fillet. The mount should not crack
in ten years, but a factor of 1.03 on stress leaves no room for porosity near the fillet, and
fatigue lives scatter by a factor of three to ten. Halving the stress at the fillet, with about 1.4
times the wall there, would meet the casting's factor of 2. The strength result itself is medium
confidence, because a rigid motor would carry the pull differently, and this answer inherits that.

The run also writes `output/playback.json`, with the damage building block by block over the ten years and
the damage at every scale of the duty's loads. The report's 3D view has a duty load slider, with the mount
colored by each spot's share of its fatigue strength and the damage at the fillet read out.
