---
title: "Validation"
description: "Eighteen cases with known answers, from textbook formulas and NAFEMS benchmarks to real parts, heat and circuits, run through Wobbl's own pipeline."
lead: "Eighteen cases with known answers run through the same pipeline your studies use, and every one lands inside its tolerance."
cards:
  - title: "How the cases are set up"
    href: "/validation/cases/"
    text: "The part, the loads and the reference behind each of the 18 cases."
  - title: "Bake-off"
    href: "/validation/bakeoff/"
    text: "A coding agent on its own against Wobbl, on six parts with a time limit."
  - title: "Limits"
    href: "/validation/limits/"
    text: "What Wobbl does not model yet, so you know when its answer is not the whole story."
---

Every number above comes from one full run of `wobbl validate`, which took 29 minutes on a 2018
six core laptop. Each case whose reference is exact for the modeled setup lands within 2%. The
real parts are compared with reference models that were refined until they stopped changing,
and their tolerances follow how settled those references were.

Medium confidence in this table is Wobbl naming what a number leans on, not a miss. On NAFEMS
LE10 the 2 mm band that stands in for the benchmark's line support makes a peak far above the
answer, which is excluded and named. On the cast motor mount the verdict flips if the servo is
treated as rigid instead of a bolt group. On the enclosure shell large deflection theory lowers
the peak by 11%, so the linear answer is the conservative one, and the reference is linear too.
On the filleted bracket the bonded screw bore reads nearly twice the real peak. The buck
converter is compared with the ideal waveform formula, which leaves out the switch and diode
drops that ngspice models, so it gets a wider tolerance.

You can run every case yourself with [`wobbl validate`](/guides/cli/validate/).
