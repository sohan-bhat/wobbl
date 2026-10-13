---
title: "Fixing parts"
description: "Turn Wobbl's suggested fixes into edited parts with wobbl fix --apply, solve each one again and compare the numbers."
lead: "wobbl fix turns the suggestions in a result into edited parts, solves each one again and lines up the numbers."
---

```sh
wobbl fix wobbl-out/bracket                          # list the fixes, each with the number it needs
wobbl fix wobbl-out/bracket --apply                  # build them and run the study again for each
wobbl fix wobbl-out/bracket --apply --only fillet,gusset
```

## The shelf bracket, four ways

The [shelf bracket](/examples/shelf-bracket/) cracks at its 0.5 mm inside corner. One
`wobbl fix --apply` builds three versions of it and runs the full study on each.

| Version | Peak stress | Safety factor (PLA, 50 MPa) | Verdict |
|---|---|---|---|
| As drawn, 0.5 mm corner | 52.3 MPa | 0.96 | fails |
| 3 mm fillet | 24.1 MPa | 2.08 | marginal, the target is 2.5 |
| Legs thickened to 10 mm | 21.0 MPa | 2.38 | marginal |
| Gusset across the corner | 9.82 MPa | 5.09 | holds |

Thicker legs barely help, because the sharp corner still multiplies the stress, and rounding
or bracing the corner is what works. The gusset's own toe is a sharp corner too, which Wobbl
excludes as a singularity and flags with medium confidence, so give it a small fillet in the
real part.

![The bracket with a gusset across its corner, where the peak drops to 9.82 MPa near the gusset toe](/img/bracket-gusset.webp "The gusset version, with the heatmap scaled to its own peak.")

## What it suggests

| Fix | When | The number it gives |
|---|---|---|
| fillet | the peak sits on a sharp inside corner or a fillet tighter than 40% of the wall | a radius of about half the wall, at least twice the current radius and at least 1 mm |
| thicken | the part misses its target | the wall thickness that reaches the target, since bending stress falls with the thickness squared and direct stress with the thickness |
| gusset | the peak sits where two walls meet and bend, and the part misses its target | a triangular gusset as thick as the wall, with legs about half as long as the shorter wall |
| material | a stronger material of the same kind reaches the target | the safety factor each candidate gives and how much the deflection changes |
| lever | the part misses its target and the load bends the critical section | the shorter distance that brings the bending stress to the target |
| lighten | the part holds with a safety factor over four times the target | how thin the wall could get and still meet the target |

A printed part is only offered other printable plastics, and other parts are offered materials
of the same family. Candidates are tried in a fixed order that starts with the ordinary
choices, so the first suggestion is the most ordinary material that works, and when you set
`--target-sf` yourself every candidate has to meet it.

## What it applies

Fillets and gussets go into any STEP file through OpenCASCADE, and the edited STEP is saved in
`fix-<kind>/geometry/` for your CAD tool. On a parametric part Wobbl changes the fillet and
thickness parameters and rebuilds it instead, which is how the shelf bracket got its 3 mm
fillet and its 10 mm legs, while its gusset went in through OpenCASCADE.
The loads and supports are carried over to the new faces by themselves, and each version is a
full result in `fix-<kind>/` with its own report and checks.

On a parametric part only those two parameters change, and every other parameter keeps the
value the first run used. A thicker wall on a STEP file, a shorter lever and a lighter part stay
as numbers for you to apply in CAD. `--only` takes `fillet`, `thicken`, `gusset` and `material`,
and `thicken` applies to parametric parts with a thickness parameter.

A material swap often needs no new solve. For one linear elastic material held by its supports
and loaded by forces, the stress field depends on Poisson's ratio but not on stiffness, so
Wobbl divides the new limit by the same stress and scales the deflection by the change in
stiffness. It solves the part again instead when Poisson's ratio differs by more than about
0.05, when a load depends on the material, as self weight does through density, and when the
new material is judged by a different stress, such as printed PLA by its largest principal
stress against PETG by von Mises, or a different strength across print layers. The terminal
prints which way each swap was found, and the JSON has it under `method`.

When the fixes are done, the original report gains a before and after pair of heatmaps for the
fix with the best safety factor, and the terminal prints the table.

## From an agent

`suggest_fix(result_dir)` returns the suggestions, and `suggest_fix(result_dir, apply=true)`
does the same as `--apply` and returns the before and after numbers, with `which` to pick the
kinds. It also takes a heat or circuit result and returns its fixes, such as more airflow or a
larger resistor, with a note to call `solve_thermal` or `solve_circuit` again with the new value,
since those fixes change an input rather than the part. [MCP tools](/guides/agents/mcp-tools/)
has the details.
