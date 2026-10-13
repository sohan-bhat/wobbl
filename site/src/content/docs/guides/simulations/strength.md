---
title: "Strength"
description: "Run a static strength study with CalculiX, refined until the peak stress settles, and read its verdict."
lead: "A strength study tells you the peak stress, where it sits, the safety factor and how far the part bends."
order: 1
---

```sh
wobbl run bracket.py -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6"
```

That is the [shelf bracket example](/examples/shelf-bracket/), a printed PLA L bracket
screwed to a wall and carrying 40 N. A study needs a part, a material, at least one support
and at least one load, and the [inputs guides](/guides/inputs/) cover how to write each one.

## What happens during a run

Wobbl reads the part and finds its holes, fillets, sharp inside corners and wall thickness.
It builds a coarse first mesh of second order tetrahedra with the curved features resolved,
solves it with CalculiX and looks for every stress peak. Then it refines the mesh around the
peaks that matter, whole fillets when they are tight and a ball around any other peak, and
solves again. It keeps going until the governing stress changes by less than the tolerance
between two meshes, with at least three meshes behind a pass.

After that come the [checks](/guides/results/checks/). Peaks at sharp corners and at
idealized supports are set aside and named, the reactions are compared with the loads, beam
theory is run through the critical section, and a part that bends a lot is solved again
with large deflection theory. Last, Wobbl draws the heatmaps and writes the summary, the
report and the data files.

## The verdict

The safety factor is the material's limit divided by the governing stress, and the verdict
compares it with your target.

| Verdict | When |
|---|---|
| holds | the safety factor meets the target |
| marginal | the part survives the load but the safety factor is below the target |
| fails | the safety factor is below 1, so the part yields or cracks |

Ductile materials are judged by von Mises stress against yield. Brittle ones, which here
means printed PLA, wood, plywood, FR4 and the lumped battery cell, are judged by the largest
principal stress against tensile strength. Each material has its own default target, 1.5 for
wrought metals and up to 3 for wood, and `--target-sf` sets your own.
[Materials](/guides/inputs/materials/) lists them all.

## Quality and time

| `--quality` | Settles when the change is under | Can stop after | Most meshes | Largest mesh |
|---|---|---|---|---|
| `fast` | 5% | 2 meshes | 3 | 80,000 nodes |
| `normal` | 3% | 3 meshes | 4 | 250,000 nodes |
| `fine` | 2% | 3 meshes | 4 | 400,000 nodes |

`fast` is for a quick look and not for an answer you report. It can stop as soon as two meshes
agree, and since two coarse meshes can agree and both still be off, its convergence check
stays a warning when that happens. Every study also has a time budget, `--budget`, 300
seconds by default. Wobbl spends it by a cost model of each mesh and solve rather than the
clock, so the same input reaches the same mesh level on every run. When the budget runs out,
the last finished level comes back with its checks and its heatmaps, and the convergence check
says how far it got.

## Loads that act off the part

A belt pull at a pulley or the thrust of a motor reaches the part through something attached
to it, and how stiff that something is can change the answer. By default Wobbl spreads such a
load over the faces it acts through without stiffening them, and `rigid` ties those faces
together as one rigid body instead. Either way, it then solves the other idealization too and
reports the spread.

On the [cast motor mount](/examples/cast-motor-mount/) a 2 kN belt pull through four bolts
puts 123.7 MPa on a rib tip, a safety factor of 1.33 against a target of 2. Treating the
servo as rigid moves the peak next to a bolt hole, where it converges to 55.0 MPa with a
safety factor of 3.00. The verdict depends on that choice, so Wobbl marks the result medium
confidence and says why. [Loads](/guides/inputs/loads/) has the syntax.
