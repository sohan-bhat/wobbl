---
title: "Circuits"
description: "Check LEDs, RC networks, buck converters, MOSFETs or any SPICE netlist with ngspice, compared against the textbook answer."
lead: "Four ready made circuit checks and any netlist of your own, each run in ngspice and checked against the textbook."
order: 3
---

## An LED that might fry

```sh
wobbl circuit led --set supply=5V --set resistor=47 --set led=blue
```

```text
FAILS. The LED fries: 36.9 mA through it, above its 30 mA rating (1.2x).

Checks:
  [pass] hand calculation: diode equation solved by hand gives 36.9 mA, ngspice 36.9 mA (5.4e-05% apart); the quick estimate (Vs - Vf)/R is 38.3 mA
  [pass] power balance: supply gives 184 mW = resistor 63.9 mW + LED 120 mW (5.4e-08% apart)
```

The fix it suggests is a 91 ohm resistor, which gives about 19.8 mA. LED colors are red,
orange, yellow, green, blue, white, uv, ir, power-white and power-red, and `vf=` and `imax=`
override the forward voltage and the current rating.

## Ripple on a buck converter

```sh
wobbl circuit buck --set vin=12 --set vout=5 --set current=2 --set frequency=500k \
  --set inductor=10u --set capacitor=22u --set max_ripple=0.05
```

This gives 8.56 mV peak to peak on 5.07 V, inside the 50 mV limit. The ideal formula says
7.95 mV, and the gap comes from the switch and diode drops that ngspice models and the formula
leaves out. The run is repeated at half the time step and the ripple agrees to 0.0022%.

![Output voltage of the buck converter rippling between about 5.068 and 5.077 volts](/img/buck-waveform.webp)

## How hot a MOSFET gets

```sh
wobbl circuit mosfet --set current=15 --set rds_on=8m --set frequency=20k \
  --set theta_ja=40 --set ambient=50
```

The junction settles at about 110 C from 1.51 W of loss, under the 150 C limit. Wobbl iterates
the junction temperature with the on resistance rising 0.6% per degree, which also catches thermal runaway.

## Bench drawings

Every run also draws the circuit wired up, in `images/setup.svg`, and its schematic, in
`images/schematic.svg`. The bench is a plausible build, such as the LED on a breadboard under a
supply that reads the simulated current, the buck converter board between a supply and an
electronic load with a scope on its output, or the MOSFET on a heatsink sized from its junction to
air resistance. Values, color bands and readings come from the run, so a different `--set` value
gives a different drawing. Any other netlist has its parts placed on a breadboard the way a person
would build it, with the measured current on every wire. The report shows both drawings, and the
[circuits example](/examples/circuits/) plays each bench from power on.

## Template parameters

Component values take SPICE style suffixes like `10k`, `4k7`, `22u` or `2.2meg`, and units
like `5V` are fine too. `max_ripple` (in volts), `cycles`, `duty`, `switching_time` (in
seconds), `ambient`, `theta_ja`, `tj_max` and `rds_tempco` take plain numbers.

| Template | Parameters, with defaults |
|---|---|
| `led` | `supply` 5 V, `resistor` 150 ohm, `led` red, `vf` and `imax` from the color, `rating` 0.25 W for the resistor |
| `rc` | `r` 10k, `c` 10u, `v` 5 |
| `buck` | `vin` 12, `vout` 5, `current` 1, `frequency` 500k, `inductor` 10u, `capacitor` 22u, `esr` 0.01, `rds` 0.05, `max_ripple` 1% of `vout`, `cycles` 300 |
| `mosfet` | `current` 10, `rds_on` 0.01, `vbus` 12, `frequency` 0 for DC, `duty`, `switching_time` 40 ns, `ambient` 25, `theta_ja` 62, `tj_max` 150, `rds_tempco` 0.006 |

## Verdicts

The LED check says ok, warm when the current is above 80% of the LED's rating or the resistor
burns more than 60% of its own, or fails. The buck check says ok or fails against `max_ripple`. The MOSFET check says ok
when the junction stays at least 25 C under its limit, warm when it is closer than that, and
hot past the limit. The RC check reports the time constant and is always ok.

## Any netlist

```sh
wobbl circuit board.cir --probe "v(out)" --probe "i(V1)"
```

Wobbl runs the netlist as written, minus any `.control` block, and reports the minimum,
maximum, mean and final value of each probe with a plot. A netlist with a `.tran` analysis is
run again at half the time step to check that the peaks have settled. You have to name at
least one probe, since Wobbl cannot guess which signals matter to you. The
[circuits example](/examples/circuits/) has all three templates above with their reports.
