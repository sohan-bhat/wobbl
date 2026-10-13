---
title: "wobbl thermal"
description: "Run a steady or transient Wobbl heat study with heat sources, convection, radiation and held temperatures."
lead: "thermal answers whether a part overheats, at steady state or over a given time."
order: 4
wide: true
---

```sh
wobbl thermal PART -m MATERIAL [--power P] [--film FILM] [options]
```

## Examples

```sh
# a 15 W chip on a heat sink in a 40 C cabinet, at steady state
wobbl thermal heatsink.py -m 6063-t5 --power "15 W" \
  --heat-faces "bottom and box 17.5,17.5,-1,42.5,42.5,0.5" \
  --film "6 W/m2K @ all" --ambient "40 C" --radiation 0.85 --limit "85 C"

# a battery pack in a sealed box over a one hour discharge
wobbl thermal pack.py -m li-ion-cell --power "6 W" --film "5 W/m2K @ all" \
  --ambient "35 C" --radiation 0.9 --duration "60 min" --limit "60 C"
```

## Options

| Option | What it does |
|---|---|
| `-m`, `--material` | the material, which is required here and sets conductivity, density and specific heat |
| `-p`, `--power` | heat made inside the part, such as `"12 W"`, spread evenly through the volume |
| `--heat-faces` | puts the power on these faces instead, such as the patch a chip touches |
| `--film` | convection, repeatable, such as `"5 W/m2K @ all"` or `"25 W/m2K at 30 C @ top"` |
| `--ambient` | the air temperature, 25 C by default |
| `--fixed-temp` | a held temperature, repeatable, such as `"40 C @ bottom"` |
| `--duration` | the length of a transient run, such as `"30 min"`, and without it the run is steady state |
| `--limit` | the temperature limit, by default the material's service limit |
| `--radiation` | the surface emissivity, such as `0.9` for painted or plastic surfaces, and 0 (the default) turns radiation off |
| `--quality` | `fast` or `normal` (the default) |
| `--question` | the question this run answers, shown in the report |
| `-o`, `--out` | the result folder, by default `wobbl-out/<part>` |
| `--json` | prints the result as JSON |

Without any `--film`, Wobbl assumes open air at 10 W/(m2 K) on every face, and radiation acts
on the same faces as the films. The verdict is ok, warm within 5 C of the limit, or hot past
it, and a transient run also says when the limit is crossed. [Heat](/guides/simulations/heat/)
has typical film values and how heat answers are checked.
