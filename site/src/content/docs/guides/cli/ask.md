---
title: "wobbl ask"
description: "Ask Wobbl about a part or a circuit in plain words, and see every guess it makes about what the question leaves out."
lead: "Put a question after the part, or ask about a circuit with no file at all, and Wobbl works out the rest."
order: 1
---

```sh
wobbl bracket.step "will this PLA bracket crack under 200 N on the top face?"
wobbl pack.step "will this 12 W battery pack overheat in a sealed box at 35 C?"
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

You never need to type `ask` itself, because Wobbl treats a file name or a quoted question in
its place as a question. `wobbl ask PART "question"` works too.

## Which study it runs

A netlist file, or a question about an LED, a resistor, ripple, a MOSFET, a regulator, a buck
converter, a capacitor, a circuit, volts or amps, becomes a circuit check. A question about
heat, temperature, overheating, cooling or watts becomes a heat study. Anything else is a
strength study.

## What it guesses

| Domain | Guesses when the question does not say |
|---|---|
| strength | aluminum 6061-T6 as the material, bolted through the most common hole size, or the bottom face held when there are no holes, and the load pushing down on the flat face farthest from the supports, or on the top when the question says so |
| heat | aluminum 6061-T6, air at 25 C, and cooling from the wording, 5 W/(m2 K) for a sealed box, 25 for a fan, 500 for water and 10 for open air |
| circuits | a 5 V supply, a 150 ohm resistor and a red LED, for a buck converter 12 V in, 5 V out, 1 A at 500 kHz with 10 uH and 22 uF, and for a MOSFET 10 A, 10 milliohm and 62 C/W to air |

Words like printed, steel or aluminum pick a material, a phrase like a safety factor of 2 sets
the target, and words like bolted, screwed, fixed at the base or on the top steer the supports
and the load. Every guess is printed while the study runs and recorded first among the
assumptions, starting with Wobbl guessed, together with the flag that would override it.

## Options

A question takes the same options as [`wobbl run`](/guides/cli/run/), and anything you pass
explicitly is used instead of a guess.

```sh
wobbl bracket.step "will it hold 200 N at the tip?" -m petg --fix "bolt holes d=5.5"
```

For an answer you rely on, give the material, supports and loads yourself with `wobbl run`.
A question is the quickest way to a first look, and the guesses are only the most common
reading.
