---
title: "Credibility evidence"
description: "Turn a finished Wobbl run into the evidence a reviewer asks for, with the odds of failure, a comparison with test readings, a V&V 40 style assessment and a plan for the physical tests that still matter."
lead: "A converged answer is one number. These four commands say how far to trust it and which physical tests it can stand in for."
order: 5
---

```sh
wobbl uncertainty wobbl-out/mount --vary "load lognormal 10%" --vary "yield 8%" --vary "E 3%"
wobbl validate-against wobbl-out/mount readings.csv
wobbl credibility wobbl-out/mount --influence high --consequence medium
wobbl test-plan wobbl-out/mount
```

Each command reads the result folder of a finished run, writes its own JSON file there and adds
a section to `report.html` and a line to `summary.txt`. Run them in this order for the strongest
evidence, though each one works on its own. Wobbl supplies the evidence and certifies nothing.
The reviewer, the customer or the regulator decides what it is enough for.

Agents get the same four steps as the MCP tools `quantify_uncertainty`, `validate_against_tests`,
`assess_credibility` and `plan_tests`.

## Uncertainty

`wobbl uncertainty` draws 200,000 Latin hypercube samples of the inputs you name, carries each
one through the run and writes `uncertainty.json` and `images/uncertainty.png`. It returns the
probability of failure with its 95% interval, the mean, spread and percentiles of every output,
and Sobol indices that say how much of the spread each input explains, alone (first order) and
with its interactions (total). The discretization error the mesh convergence check measured is
added as one more input unless you pass `--no-mesh-error`.

| Scatter | Example |
|---|---|
| a percentage, normal by default | `"load 10%"` |
| another distribution | `"yield lognormal 8%"`, `"E uniform 70000..75000"` |
| a standard deviation | `"vf sd 0.05"` |
| a tolerance band, read as plus or minus three standard deviations | `"thickness tol 0.2"` |
| a range | `"film uniform 3..8"` |

It works on a run of any domain, a part, a circuit, a flight, a pipe, a motor or a cell. Every
run keeps `request.json`, the values it was made from, and any number in it can scatter by its
name, such as `load1`, `thickness`, `rocket.cd`, `film` or `ambient`, in the unit it was written
in. A name it does not know gets the list of names back. The run is then solved again through its
own analysis at a design of values, and the answer is a polynomial chaos fit of those runs. What
fails comes from the criteria the run lists, each a value it must stay under or over, and the
numerical error from the change its own convergence checks measured.

Some domains have a faster exact path behind the same command.

| Input | How it is carried |
|---|---|
| load, Young's modulus and strength of a strength run | exact scaling of the converged solution, in seconds |
| one load at a time, `load1`, `load2` and so on, when a study has several | one solve per load on the final mesh, summed exactly for every sample |
| Poisson's ratio `nu` and the stiffness of an elastic support (`support`) | a few solves on the final mesh, fitted with polynomial chaos |
| heat inputs such as `film`, `power` and `ambient` | heat solves on the converged mesh, fitted with polynomial chaos |
| anything else, in any domain | the whole run again through its analysis, fitted with polynomial chaos |

Every fitted answer reports its leave one out error, the share of the variance the fit misses
on a solve it did not see. When a run also solved the other load attachment, the probability of
failure under that idealization comes back next to the main one, because that choice often
moves the answer more than any scatter.

## Comparing with tests

`wobbl validate-against` reads a CSV of physical readings, one per row, and predicts each one
from the run. A reading's quantity is any value the run reports, by the name in its
`outcome.json`, given in any unit of the same kind, so an altimeter's apogee in feet compares
with `apogee_m`. With a time it is read from a time series the run wrote, such as a flight's
trajectory. On a part a strain gauge is read the way a real one reads, averaged over its grid on
the surface along its axis, and a dial gauge as the motion of the nearest node along its direction.

```text readings.csv
name,quantity,x,y,z,direction,load,value,uncertainty,unit
SG1,strain,43.84,8.91,79.39,"-0.179,-0.627,0.758",2000,1566.0,30.5,microstrain
D1,displacement,24.23,0,112,"-0.190,-0.972,-0.137",2000,0.2686,0.005,mm
```

```text flight.csv
name,quantity,value,uncertainty,unit
ALT,apogee_m,10560,30,ft
GPS,landing_speed_mps,5.9,0.3,m/s
```

| Column | Meaning |
|---|---|
| `name` | the gauge or sensor |
| `quantity` | `strain`, `displacement`, `temperature`, or a circuit value such as `i_led_A` or a signal such as `v(out)` |
| `value`, `unit` | the reading, in microstrain, mm, in, C, F, K, V, mV, A or mA |
| `uncertainty` | its standard uncertainty, in the same unit |
| `x`, `y`, `z` | where the gauge sits, in mm |
| `direction` | the gauge axis, such as `x`, `-z` or `0.87,0,-0.5` |
| `load` | the load during the reading, in the study's load unit |
| `time` | seconds into a transient or a scope trace |
| `gauge_length` | the grid length in mm, 3 by default |

For each reading it computes the comparison error and the validation uncertainty as ASME V&V 20
does, from the reading's own uncertainty, the mesh error, the test load's uncertainty
(`--load-uncertainty`, 1% by default) and the input scatter. A reading agrees when the error is
within twice its validation uncertainty. The area metric then compares the readings with the
model's whole distribution, pooled across gauges. Unless you pass `--no-calibrate`, the inputs
are calibrated from the readings by Bayesian inference, with their new means and 95% intervals
and how much narrower the test made them.

For a strength run that solved both load attachments, the modulus is fitted to each one on its
own and the likelihood ratio says which the readings favour. A gauge placed where the two
disagree settles that choice. A line starting with `#` in the CSV is a note, and a note that
says the readings are synthetic marks every result made from them as synthetic.

## The assessment

`wobbl credibility` weighs the evidence in the manner of ASME V&V 40. You say how much the
decision leans on the model (`--influence`) and how bad a wrong decision would be
(`--consequence`), each `low`, `medium` or `high`. Together they set the model risk from 1 to 5
and a proposed goal for each factor.

| Factor | Evidence |
|---|---|
| code verification | the validation suite cases of the run's domain |
| solution verification | the run's own convergence and balance checks |
| inputs as intended | whether Wobbl guessed any load, support or material |
| model form | the idealizations the checks flagged, and which one the readings favour |
| model inputs | whether the scatter of the inputs that matter was propagated |
| comparator | how many readings, how real and how well stated |
| output comparison | agreement within the validation uncertainty and the area metric |
| applicability | whether the gauges sit near the governing point and the test reaches the design load |
| uncertainty in the answer | the probability of failure, across the idealizations Wobbl solved |

Each factor gets a level, the evidence behind it and, when it falls short, what would close the
gap. The command writes `credibility.json` and `credibility.html`, a self contained page with the
question, the answer and its checks, the uncertainty, the comparison with tests, the factors and
the test plan, ready to hand to a reviewer.

## The test plan

`wobbl test-plan` recommends the fewest physical tests that most reduce the risk, in order.

| Test | When it is chosen |
|---|---|
| a validation load test | whenever a validation factor is short, with gauge positions, axes, load steps and predicted readings with 95% bands |
| a gauge where the idealizations differ | whenever the model form is short, at the spot that tells the two load attachments apart |
| input measurements | load cells, coupons or a first article inspection, by first order Sobol index until they cover 80% of the variance measuring could remove |

The load steps stop where the peak reaches 80% of the strength limit, so the test does not damage
the part. The plan ends with the usual tests and whether the simulation evidence can likely stand
in for each one, cannot yet, or cannot at all, with the reason. A static strength model can never
stand in for a test to failure, the coupons it rests on, or fatigue, impact and vibration tests.

The [cast mount credibility](https://github.com/sohan-bhat/wobbl/tree/main/examples/cast-mount-credibility)
example runs all four steps on a cast motor mount with clearly synthetic readings.
