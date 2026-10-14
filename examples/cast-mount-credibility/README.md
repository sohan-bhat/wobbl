# Cast mount credibility

How far can simulation stand in for testing a cast motor mount under a 2 kN belt pull, and which physical tests
does it still need?

This is the sand cast A356-T6 mount from the [cast-motor-mount](../cast-motor-mount) example, with the same 2 kN
emergency stop belt pull and the same house rule of safety factor 2 on yield. On top of the strength study it
builds the evidence that virtual testing frameworks ask for, the odds of failure under load and material scatter,
a comparison with gauge readings, an assessment in the manner of ASME V&V 40 and a plan for the physical tests that
matter. Wobbl supplies that evidence and certifies nothing. Whether it is enough is the reviewer's call.

```sh
./run.sh
```

The study runs without heatmaps to keep the output small, and the cast-motor-mount example has them. Open
`output/credibility.html` for the page a reviewer would read, or `output/report.html` for the full run.

| Step | Result |
|---|---|
| Strength | peak 124 MPa on the 3.99 mm fillet at the rib tip, safety factor 1.33 against a target of 2, settled to 0.1% over three meshes |
| Scatter | belt pull lognormal 10%, yield normal 8% (165 MPa, sd 13.2), Young's modulus normal 3%, and the 0.1% discretization error |
| Probability of yielding | about 1 in 77 (1.3%, 95% interval 1.25% to 1.35%) from 200,000 samples, and 99.9% of samples miss the target of 2 |
| Safety factor | mean 1.35, standard deviation 0.173, 90% between 1.08 and 1.65 |
| What matters most | the belt pull (total Sobol index 0.61), then the yield strength (0.39), with the modulus and the mesh near zero |
| Rigid servo bound | with the servo treated as rigid, no failures in 200,000 samples (below 1.8e-5) |
| Readings | 16 synthetic readings, 16 within twice their validation uncertainty, comparison error -2.3% on average |
| Idealization | the readings favour the distributed attachment by a likelihood ratio of more than 1,000 |
| Calibration | Young's modulus 70,900 MPa, 95% interval 70,100 to 71,800, 4.8x narrower than before |
| Credibility | model risk 4 of 5, so a high goal on each factor, met on 4 of 9 |
| Test plan | a gauged load test, a gauge where the idealizations differ, a service load measurement and coupons |

The uncertainty needs no new solve. A linear elastic part with one material has a stress field that follows the
load and does not depend on Young's modulus, so the converged solution scales exactly to every sample, and the
200,000 samples and the Sobol indices take seconds.

## The readings are synthetic

`readings.csv` does not come from a real test. `make_readings.py` made it from Wobbl's own solution, read the way
three strain gauges and a dial gauge would read it at 500, 1000, 1500 and 2000 N, with Young's modulus set to
70.6 GPa and gauge noise added. Comparing the model with readings made from the model shows the workflow and how
calibration and the idealization comparison behave. Calibration does find the modulus again, and 70.6 GPa sits
inside its interval. It says nothing about how well the model matches a real mount, and Wobbl marks every result
made from these readings as synthetic. That is also why comparator, output comparison and applicability stay low
in the assessment.

## The assessment

The mount's design decision leans on the model (high influence) and a wrong call means a field failure but no
injury (medium consequence), which puts the model risk at 4 of 5. The evidence meets a high goal on code
verification (13 strength cases with known answers, the largest error 4.3%), solution verification, inputs stated
explicitly and model inputs. It falls short on the model form, because the answer still depends on how the servo
carries the belt pull, on the three factors that need a real test, and on the answer's own uncertainty, since a
1.3% chance of yielding is too much to stand on the model alone.

## The test plan

1. A static load test in steps of 500, 1000, 1500 and 2000 N, with a strain gauge at the rib tip (1486 ± 89
   microstrain at 2 kN), one on a calm spot of the upright (818 ± 49) and a dial gauge where the mount moves most
   (0.265 ± 0.016 mm). The stress falls 26% across a 3 mm grid at the rib tip, so that gauge should be a 1 mm grid.
2. A strain gauge on the servo face at (-34, 0, 29.5), where a rigid servo would put its peak. The two
   idealizations disagree there by a factor of three, so this one reading settles the choice that moves the
   answer most.
3. A load cell on the belt in service, since knowing the real pull removes about 61% of the variance of the safety
   factor.
4. Tensile coupons from the same casting lot, which remove most of the rest.

With real readings that agree, the plan says a static proof test at the design load could likely be left to the
simulation if the failure probability were also small. Here it is not, so the proof test stays. A test to failure,
the coupons themselves and any fatigue, impact or vibration test cannot be replaced by this static linear model.
