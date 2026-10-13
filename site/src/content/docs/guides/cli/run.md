---
title: "wobbl run"
description: "Run a Wobbl strength study with explicit material, supports and loads, and every option that controls it."
lead: "wobbl run is the strength study with everything spelled out, which is what you want for an answer you rely on."
order: 2
wide: true
---

```sh
wobbl run PART -m MATERIAL --fix SUPPORT --load LOAD [options]
```

`wobbl run` needs at least one `--load` and one `--fix`. Without a material it falls back to
aluminum 6061-T6 and says so in the assumptions.

## Examples

```sh
# a printed bracket screwed to a wall
wobbl run bracket.py -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6"

# a steel tab pulled sideways by the bolt through its hole
wobbl run tab.step -m mild-steel --fix "x=min" --load "1.5 kN +x bearing @ hole d=8.5"

# a cast mount with a belt pull off the part and a house rule of safety factor 2
wobbl run cast_motor_mount.step -m "custom:E=72400,nu=0.33,yield=165,process=cast" \
  --target-sf 2 --fix "z=min" --load "2000 N -0.8660254,0,-0.5 at 0,-30,66 @ holes d=6.6" \
  --budget 600

# a printed mount with its layers stacked along z, run without its ribs
wobbl run mount.py --param rib=0 -m petg --print-direction z --fix "bolt holes d=5.5" \
  --load "250 N down at 0,-15,31 @ normal +y and box -21,4,10,21,6,52 rigid"
```

## Options

| Option | What it does |
|---|---|
| `-m`, `--material` | a material key such as `pla` or `6061-t6`, a library material with one property changed such as `pla:yield=40`, or `custom:E=...,nu=...,yield=...`, as [Materials](/guides/inputs/materials/) explains |
| `-l`, `--load` | a load, repeatable, such as `200 N down @ top`, or `300 N -x bearing @ hole d=6.6` for a bolt pressing on one side of its hole, as [Loads](/guides/inputs/loads/) explains |
| `-f`, `--fix` | a support, repeatable, such as `bolt holes d=4.5` or `clamp x=min`, as [Supports](/guides/inputs/supports/) explains |
| `--target-sf` | the safety factor the part has to reach, by default the material's own |
| `--quality` | `fast` for a quick look, `normal` (the default) or `fine` for a tighter tolerance |
| `--budget` | seconds for the whole study, 300 by default |
| `--print-direction` | the axis printed layers stack along, such as `z`, so pull across the layers is judged against the weaker layer strength |
| `--param` | a parameter of a parametric `.py` part, such as `thickness=6`, repeatable |
| `--scale` | scales the geometry, `25.4` for a file in inches or `1000` for one in meters |
| `--body` | picks one body, counting from 1, from a file that holds several |
| `--mesh-size` | forces the global size of the first mesh in mm |
| `--question` | the question this run answers, shown in the report |
| `-o`, `--out` | the result folder, by default `wobbl-out/<part>` |
| `--json` | prints the result as JSON instead of the readable summary |
| `--no-images` | skips drawing the heatmaps |
| `-q`, `--quiet` | drops the progress lines |

## Time

A strength study on a real part takes one to seven minutes on a 2018 laptop. The budget is spent
by a cost model of each mesh and solve rather than by the clock, so the same input reaches the
same mesh level on every run, and when the budget runs out the last finished level comes back
with its checks. Large or thin walled parts need big meshes, so give them more time with
`--budget 600`, as the cast motor mount, enclosure and drone arm examples do.

## Output

The terminal shows the verdict, the peak stress, the safety factor, the target, the largest
deflection and the confidence with its reasons, then the checks, the fixes, the assumptions
and the file paths. [Reading a result](/guides/results/reading/) walks through it.
