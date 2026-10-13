---
title: "wobbl circuit"
description: "Check an LED, RC network, buck converter or MOSFET with ngspice, or run any SPICE netlist and probe its signals."
lead: "circuit runs one of four ready made checks, or your own netlist, in ngspice."
order: 5
---

```sh
wobbl circuit TEMPLATE --set KEY=VALUE [--set ...]
wobbl circuit NETLIST --probe SIGNAL [--probe ...]
```

## Examples

```sh
wobbl circuit led    --set supply=5V --set resistor=100 --set led=blue
wobbl circuit rc     --set r=10k --set c=10u --set v=5
wobbl circuit buck   --set vin=12 --set vout=5 --set current=2 --set frequency=500k \
                     --set inductor=10u --set capacitor=22u --set max_ripple=0.05
wobbl circuit mosfet --set current=15 --set rds_on=8m --set frequency=20k --set theta_ja=40
wobbl circuit board.cir --probe "v(out)" --probe "i(V1)"
```

## Options

| Option | What it does |
|---|---|
| `--set` | a template parameter as `key=value`, repeatable |
| `--probe` | a signal to read from a netlist, such as `v(out)` or `i(V1)`, repeatable and required for a netlist |
| `--question` | the question this run answers, shown in the report |
| `-o`, `--out` | the result folder, by default `wobbl-out/<template>` |
| `--json` | prints the result as JSON |

The target is read as a netlist when it is a file, and as a template name otherwise, so
`wobbl circuit led` still works with a folder called `led` next to you. Every template
parameter and its default is listed in [Circuits](/guides/simulations/circuits/), along with
the verdicts and the checks.
