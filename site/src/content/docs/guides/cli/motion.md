---
title: "wobbl motion"
description: "Fly a drop, a throw or a rocket spec through gravity, air, wind, thrust and parachutes, checked against closed forms and an energy balance."
lead: "motion answers how something falls, flies or launches, for a body given as options or a rocket in a spec file."
order: 5.5
---

```sh
wobbl motion --diameter D --mass M [--height H] [--speed V --angle A] [--wind W]
wobbl motion SPEC.toml [--wind W]
```

## Examples

```sh
# a 9 cm granite rock dropped off a 300 m cliff
wobbl motion --name rock --diameter "9 cm" --density "2700 kg/m3" \
  --cd 0.6 --height "300 m"

# a baseball thrown at 40 m/s and 35 degrees
wobbl motion --name ball --mass "145 g" --diameter "7.4 cm" --speed "40 m/s" --angle 35

# a high power rocket from its spec, on a windier day than the spec says
wobbl motion rocket.toml --wind "15 mph"
```

## Options

| Option | What it does |
|---|---|
| `SPEC` | a TOML file for a rocket or a body, which [Motion](/guides/simulations/motion/) describes table by table |
| `-d`, `--diameter` | the diameter that meets the air, such as `9 cm` |
| `--mass` | the body's mass, such as `1 kg` or `145 g` |
| `--density` | the density of a solid round body, such as `2700 kg/m3`, in place of `--mass` |
| `--cd` | the drag coefficient on the frontal area, 0.47 for a sphere unless you give one |
| `--height` | the height above the ground at the start |
| `--speed` | the launch speed, such as `30 m/s` or `60 mph` |
| `--angle` | the launch angle above level, 45 degrees for a throw unless you give one |
| `--wind` | a steady wind blowing back toward the start, negative when it blows from behind, and with a spec it replaces the spec's wind |
| `--name` | the name in the report and the result folder, the spec's own name when there is one |
| `--question` | the question this run answers, shown in the report |
| `-o`, `--out` | the result folder, by default `wobbl-out/<name>` |
| `--json` | prints the result as JSON |

With a spec file only `--wind`, `--name`, `--question` and `--out` apply, since everything else
about the rocket lives in the spec. Every quantity takes its unit, and the result folder gets
`summary.txt`, `outcome.json`, `report.html`, the charts in `images/` and `trajectory.csv`, at
most 500 rows with one at every event. Motion runs on scipy, which every Wobbl install already
has, so it needs no solver download, and the examples each took about four seconds with their
checks.
