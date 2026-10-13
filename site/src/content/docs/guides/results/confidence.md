---
title: "Confidence"
description: "How Wobbl turns its checks into a high, medium or low confidence level, what each reason means and what to do about it."
lead: "The confidence level sums up the checks in one word, and the reasons next to it say exactly why."
order: 3
---

| Level | Means |
|---|---|
| high | every check passed or did not apply |
| medium | at least one warning, such as a number still settling, a verdict that depends on an assumption or a sharp corner in a brittle part |
| low | at least one check failed |

A result that failed a check is never reported as high confidence. The reasons are always
listed next to the level, in the terminal, in `summary.txt`, in `outcome.json` and at the top
of the report, so an agent quoting the number has the reasons in the same place.

## Medium is information, not a miss

Medium usually means Wobbl found something the number leans on and is telling you. On the
cast motor mount the verdict flips if the servo is treated as rigid. On the enclosure shell,
sharp corners at the boss roots are excluded because the mesh sets their value. On the motor
mount the peak still moves 6.6% between the last two meshes, even though the highest value on
any mesh leaves the verdict unchanged. Each is a fact about the setup you can act on, and
none of them means the answer is wrong.

## What to do at each level

| Reason | What to do |
|---|---|
| still moving between meshes | run again with `--quality fine` or a longer `--budget`, or accept the stated error when the verdict holds either way |
| the verdict depends on how the load enters the part | decide whether the attached part is closer to rigid or flexible, and design for the worse case when you cannot tell |
| a sharp inside corner in a brittle material | add a fillet to the CAD, since the real peak depends on the real corner radius |
| an excluded peak far above the answer | model the support or the corner more realistically, for example a bolted hole instead of a fixed face |
| large deflection changes the peak | use the scaled value in the check, which says whether linear theory is conservative |
| force balance or sanity failed | check the loads, the units and the supports, since the setup itself is off |

## Same input, same answer

The same study on the same input gives the same numbers on every run, digit for digit,
including the rigid attachment variant of the load check. Meshing runs on one thread with a
fixed seed, the wall thickness is sampled from points laid out on the CAD faces, the exclusion
distance around singular spots comes from the sizing rules rather than the elements gmsh
happened to make, and the time budget is spent by a cost model instead of the clock. A test
runs a study twice and compares every number.

The one exception is a study stopped by the wall clock backstop, which only happens on a
machine far slower than the cost model expects or when a solver stalls, and the result says
so in its notes. A different machine can also give slightly different numbers when its gmsh,
OpenCASCADE or CalculiX build differs, since library versions and floating point change where
nodes land.
