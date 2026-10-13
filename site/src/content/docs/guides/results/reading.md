---
title: "Reading a result"
description: "How to read what Wobbl prints after a study, from the verdict and the confidence to the fixes and the assumptions."
lead: "Every result opens with a verdict and one sentence that says what happens and where."
order: 1
---

## The headline

The first line gives the verdict and the governing number with its location, in the words the
material calls for.

```text
FAILS. Peak 52.3 MPa on the 0.5 mm inside fillet (face 3) at (6.19, 12.6, 6.11). The part cracks at about 0.96x this load.
MARGINAL. Peak 124 MPa on the 3.99 mm inside fillet (face 65) at (43.9, 8.96, 79.6). It survives the load, but the safety factor 1.33 is below the 2 target. Confidence is medium.
HOLDS. Peak 9.46 MPa on face 2 at (25.8, -8, 0), safety factor 5.07 against yield (target 3). Confidence is medium.
```

Brittle materials crack and ductile ones yield, so the wording follows the material. The
location names the feature when there is one, such as a fillet, a hole or a boss, along with
the face id and the coordinates in millimeters. Heat results say ok, warm or hot, and circuit
results say ok, warm, hot or fails, each with a sentence of their own.

## The numbers and the confidence

In the terminal the headline sits in a box with the peak stress, the safety factor, the target
and the largest deflection, and below them the confidence level with its reasons. Read those
reasons before the number. High confidence lists the checks that passed, while medium and low
say exactly what raised the doubt, such as a verdict that flips with an assumption or a peak
still moving between meshes. [Confidence](/guides/results/confidence/) goes through each
reason.

## Checks

Next comes one line per check with its status.

| Status | Meaning |
|---|---|
| pass | the check ran and the answer met it |
| warn | something to look at, which lowers the confidence to medium |
| fail | the answer did not meet the check, which makes the confidence low |
| info | worth knowing, with no effect on the confidence |
| skip | the check does not apply to this setup, with the reason |

[Checks](/guides/results/checks/) explains every one of them.

## Fixes

When the part fails or is marginal, a section called how to fix it lists concrete changes,
each with the number it needs, such as a 3 mm fillet or a wall thickened from 6 to 9.7 mm.
Fixes marked auto are ones `wobbl fix --apply` can build and solve again by itself. When the
part holds, the same section is called worth knowing, and when the safety factor is more than
four times the target it says how thin the wall could get. [Fixing parts](/guides/fixing/)
covers both.

## Assumptions

Every result lists what the answer rests on, such as linear elastic material, static loads,
how the supports are idealized, the material's source, and the caveats that come with how the
part is made, like layer strength for a print or porosity for a casting. When you asked in
plain words, each guess Wobbl made comes first, starting with Wobbl guessed, so you can see
which supports and load placement it chose and override them. The terminal shows the first
eight, and the files keep them all.

## Files

The last lines point to the heatmap, the report and the summary in the result folder, which
is `wobbl-out/<part>` unless you passed `-o`. Add `--json` to any run to print the whole
result as JSON instead of the readable version, which is what scripts and agents use.
[Files](/guides/results/files/) describes everything in the folder.
