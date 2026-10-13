---
title: "First run"
description: "Ask Wobbl a circuit question with no files, then run the shelf bracket example and read the answer."
lead: "Start with a question that needs no files, then run a real part from the examples."
order: 3
---

## A question with no files

```sh
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

Wobbl reads the supply, the resistor and the LED color out of the question, builds the
circuit in ngspice and solves the diode equation by hand to compare. It takes well under a
second.

```text
FAILS   The LED fries: 36.9 mA through it, above its 30 mA rating (1.2x).

LED current   rating   resistor power   LED power
36.9 mA       30 mA    63.9 mW          120 mW

confidence high
  passed hand calculation, power balance

HOW TO FIX IT
  Use a 91 ohm resistor
  Gives about 19.8 mA; it dissipates 35.6 mW, so a 0.25 W part is enough.
```

The hand calculation and ngspice agree to better than a ten thousandth of a percent, and the
power the supply gives matches what the resistor and the LED burn. Everything is also saved to `wobbl-out/led/`, with a
`summary.txt`, an `outcome.json` and a `report.html` you can open in a browser.

## A real part

The repository ships with eight examples. The shelf bracket is a printed PLA L bracket,
screwed to a wall through two holes, carrying 40 N of an 8 kg shelf.

```sh
git clone https://github.com/sohan-bhat/wobbl
cd wobbl/examples/shelf-bracket
wobbl inspect bracket.py
```

`inspect` reads the part and lists what you can point at, which is how you find the names
for the supports and the load.

```text
bracket.py  90 x 25 x 80 mm, volume 24.42 cm3, walls about 6 mm (thinnest 6 mm)

HOLES
  d 4.5 mm at (3, 12.5, 25)  select with 'hole d=4.5' or 'face 10'
  d 4.5 mm at (3, 12.5, 65)  select with 'hole d=4.5' or 'face 11'

INSIDE FILLETS
  radii 0.5 mm
```

Now hold it the way the real one is mounted and load it. The two screws become
`bolt holes d=4.5`, the back of the upright rests on the wall and can lift off, and the shelf
presses on a 12 mm patch of the arm 65 mm out.

```sh
wobbl run bracket.py -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6"
```

This one takes about two minutes, because Wobbl meshes the part four times and solves each
mesh until the stress settles. The answer leads with the verdict.

```text
FAILS. Peak 52.3 MPa on the 0.5 mm inside fillet (face 3) at (6.19, 12.6, 6.11). The part cracks at about 0.96x this load.

Confidence: high. passed mesh convergence, force balance, hand calculation, singularities, bolt bearing, units and sanity
Max deflection 3.47 mm at (90, 0, 6).
```

Printed PLA breaks at about 50 MPa, so 52.3 MPa at the inside corner means the bracket
cracks, and since every check passed the confidence is high.
[Reading a result](/guides/results/reading/) goes through the rest of the output.

## Look at it and fix it

```sh
wobbl report --open
wobbl fix wobbl-out/bracket --apply
```

The report has the heatmap with the failure point marked, the checks, every stress peak and
the mesh history. `wobbl fix --apply` then builds a 3 mm fillet, thicker legs and a gusset,
runs the whole study again for each one and lines them up, which is what
[Fixing parts](/guides/fixing/) shows.

## Your own part

```sh
wobbl inspect part.step
wobbl part.step "will this 6061 bracket yield under 50 lbf at the tip?"
```

A plain question works, and Wobbl prints every guess it made about the supports, where the
load goes and the material. For an answer you are going to rely on, give those yourself with
[`wobbl run`](/guides/cli/run/), using the [inputs guides](/guides/inputs/) for the syntax.
