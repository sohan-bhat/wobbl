---
title: "Heat"
description: "Steady and transient heat studies with convection, radiation, heat sources and held temperatures, solved with CalculiX."
lead: "A heat study tells you how hot the hottest point gets, and in a transient run, when it crosses your limit."
order: 2
---

## A chip on a heat sink

```sh
wobbl thermal heatsink.py -m 6063-t5 --power "15 W" \
  --heat-faces "bottom and box 17.5,17.5,-1,42.5,42.5,0.5" \
  --film "6 W/m2K @ all" --ambient "40 C" --radiation 0.85 --limit "85 C"
```

The 15 W goes in through the 25 mm square the chip touches. Still air in a 40 C cabinet takes
heat away at about 6 W/(m2 K) and the anodized surface radiates. With no `--duration` the run
is steady state, and the [heat sink example](/examples/heat-sink/) comes back like this.

![Temperature map of the aluminum heat sink seen from below, 70.3 C under the chip and 67.1 C at the fin tips](/img/heat-sink-temperature.webp)

```text
OK. Hottest point reaches 70.3 C (limit 85 C) at (30, 29.9, 0) mm at steady state.

Confidence: high. passed mesh convergence, energy balance, hand calculation, units and sanity
```

## A battery pack over an hour

```sh
wobbl thermal pack.py -m li-ion-cell --power "6 W" --film "5 W/m2K @ all" \
  --ambient "35 C" --radiation 0.9 --duration "60 min" --limit "60 C"
```

`--duration` makes it a transient run that starts at the air temperature, and Wobbl reports
when the limit is crossed. The [battery pack example](/examples/battery-pack/), a 2S2P pack in
a sealed box, passes 60 C after about 21 minutes and reaches 74.5 C at the hour, so the
verdict is hot.

![The battery pack temperature over the one hour discharge, crossing the 60 C limit just after 20 minutes](/img/battery-history.webp)

## Choosing the inputs

| Input | How to set it |
|---|---|
| heat made in the part | `--power "12 W"`, spread evenly through the volume |
| heat from a chip | `--power` together with `--heat-faces` on the faces it touches |
| cooling by air or liquid | `--film`, repeatable, with roughly 5 W/(m2 K) for still air in a box, 10 for open air, 25 for a small fan, 60 for strong forced air and 500 for water |
| air temperature | `--ambient`, 25 C unless you say otherwise |
| a held temperature | `--fixed-temp "40 C @ bottom"` |
| radiation | `--radiation 0.9` for painted or plastic surfaces, off by default |
| the limit | `--limit "60 C"`, or the material's own service limit |

A film can carry its own air temperature, as in `--film "25 W/m2K at 30 C @ top"`. Without
any `--film`, Wobbl assumes open air at 10 W/(m2 K) on every face. Radiation acts on the same
faces as the films. [wobbl thermal](/guides/cli/thermal/) lists every option.

## The verdict

| Verdict | When |
|---|---|
| ok | the hottest point stays at least 5 C under the limit |
| warm | it stays under the limit by less than 5 C |
| hot | it goes over the limit |

## How heat answers are checked

The hottest point has to settle within 2% between meshes, and two meshes are enough here
because temperature is a smooth field. In steady state, the heat going in has to match the
heat leaving by convection, radiation and any held temperature faces to within 1%. A lumped
hand calculation treats the part as one uniform block and is compared with the surface
average in steady state, or with the mean temperature over time in a transient. When
radiation is off but the part runs hot, the result says how much heat radiation could shed.
[Checks](/guides/results/checks/) has the details.
