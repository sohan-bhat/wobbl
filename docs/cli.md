# Command line reference

Every command prints a readable summary in the terminal and writes its full
result to a folder (default `wobbl-out/<part>`): `report.html`, `summary.txt`,
`outcome.json`, heatmaps in `images/` and the full field in `result.vtu` for
ParaView. Add `--json` to get the result on stdout instead.

## Ask in plain words

```sh
wobbl bracket.step "will this PLA bracket crack under 200 N on the top face?"
wobbl pack.step "will this 12 W battery pack overheat in a sealed box at 35 C?"
wobbl "will a blue LED on 5V with a 47 ohm resistor fry?"
```

Wobbl reads the load, material, supports and domain out of the question and
fills the gaps with the most common reading: bolted through the holes of the
most common size, load on the flat face farthest from them, pushing down,
aluminum 6061-T6 if no material is named. Every guess is printed and recorded
as an assumption. For anything that matters, use the explicit flags below.

## wobbl run

```sh
wobbl run PART -m MATERIAL --fix SUPPORT --load LOAD [options]
```

| Option | Meaning |
|---|---|
| `-m, --material` | `pla`, `petg`, `abs`, `asa`, `nylon`, `pa12`, `pc`, `hdpe`, `6061-t6`, `7075-t6`, `6063-t5`, `5052-h32`, `mild-steel`, `4140`, `ss304`, `ti-6al-4v`, `copper`, `brass`, `wood`, `plywood`, `fr4`, `li-ion-cell`, or `custom:E=3000,nu=0.35,yield=40,density=1200` (MPa, kg/m3). `pla:yield=40` overrides one property. |
| `-l, --load` | repeatable, see [loads](#loads) |
| `-f, --fix` | repeatable, see [supports](#supports) |
| `--target-sf` | required safety factor (default: the material's, 1.5 for metals, 2.5 for printed plastics) |
| `--quality` | `fast` (two meshes, 5% tolerance, for quick looks), `normal` (three or more meshes, 3%), `fine` (2%) |
| `--budget` | seconds for the whole study, default 300; when it runs out Wobbl returns the last finished mesh level with its checks |
| `--print-direction` | axis the print layers stack along, e.g. `z`; pull across layers is judged against the weaker layer strength |
| `--param k=v` | parameter of a parametric `.py` part |
| `--scale` | scale the geometry, e.g. `1000` for a file in meters, `25.4` for inches |
| `--body N` | pick one body from a multi body file |
| `--mesh-size` | force the first element size in mm |
| `-o, --out` | output folder |
| `--no-images`, `--json`, `-q` | skip rendering, print JSON, no progress lines |

Units are millimeters for geometry. Any common unit works in loads: `N`, `kN`,
`lbf`, `kgf`, `MPa`, `psi`, `ksi`, `bar`, `N*m`, `in-lb`, `g`.

## Loads

| Kind | Example | Meaning |
|---|---|---|
| force | `200 N down @ top` | total force spread evenly over the faces |
| force | `50 lbf -z @ x=max` | directions: `+x -x +y -y +z -z up down left right front back` or a vector `0,0.7,-0.7` |
| force | `20 N normal @ face 4` | pushes into each face along its normal (negative pulls out) |
| pressure | `0.2 MPa @ top` | pushes into the faces; negative pulls |
| bolt | `300 N -x @ hole d=6.6` | a pin or bolt bearing on a hole with a cosine pressure, shared by all holes picked |
| gravity | `3 g down` | self weight times a factor; needs no faces |
| remote | `2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6` | a force at a point off the part, carried by the faces |
| remote, rigid | `... @ holes d=6.6 rigid` | the attached part is rigid (a motor flange, a clamp) |
| moment | `0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25 rigid` | a torque about an axis through the point |

Remote loads and moments are distributing by default: the exact resultant is
spread over the faces without stiffening them. `rigid` ties the faces to the
point as one rigid body (CalculiX `*RIGID BODY`). Several rigid loads on the same
faces are merged into one attachment.

## Supports

| Kind | Example | Meaning |
|---|---|---|
| fixed | `x=min`, `holes d=4.5` | every point on the faces held still |
| fixed in some directions | `fixed-z faces 7`, `fixed-xy faces 3,4` | only those directions held |
| bolt | `bolt holes d=6.6` | the hole and the washer ring around it (outer radius twice the hole's) held still |
| roller | `roller y=0` | slides in its plane, cannot move off it (symmetry planes, sliding fits) |
| wall | `wall x=min` | rests on a rigid wall: can push on it, can lift off. Solved as contact |

Wobbl refuses to solve a part that is not held and names the free motion
("free to spin about z").

## Selecting faces

| Selector | Picks |
|---|---|
| `top` `bottom` `left` `right` `front` `back` | flat outer faces at the extremes (+z up, -y front) |
| `x=min` `z=max` `y=12.5` | faces lying in that plane |
| `x>100` `z<5` | faces whose center is past that coordinate |
| `normal +z`, `normal 0,0.7,0.7`, `facing down` | flat faces with that outward normal |
| `holes`, `hole d=4.5`, `holes d<6`, `hole near 55,0,10` | holes by size or position |
| `bosses`, `fillets`, `cylinders r=3`, `flats`, `curved` | faces by shape |
| `largest`, `smallest`, `area>100` | faces by size |
| `near 10,20,5` | the single closest face |
| `within 15 of 50,-25,35` | only the part of faces within 15 mm of the point (a patch) |
| `box 0,0,0,10,20,5` | only the part of faces inside the box |
| `face 12`, `faces 3,5,7` | ids from `wobbl inspect` |
| `A and B`, `A or B` | combine; region clauses apply first, so `top and within 14 of P` is the highest face near P |

## wobbl inspect

```sh
wobbl inspect part.step [--json]
```

Bounding box, volume, wall thickness, holes with diameters and centers,
fillet radii, sharp inside corners, and the largest faces with their ids and
normals. Run it first; it tells you which selectors will work.

## wobbl fix

```sh
wobbl fix wobbl-out/bracket            # list fixes with numbers
wobbl fix wobbl-out/bracket --apply    # apply the automatic ones and re-run each
wobbl fix wobbl-out/bracket --apply --only fillet,gusset
```

Fillet radius, wall thickness, a corner gusset, a stronger material and a
shorter lever arm, each with the number it needs. Applied automatically:
fillets and gussets on any STEP (OpenCASCADE edits; the edited STEP is saved),
thickness and fillet parameters of parametric parts, and material swaps.

## wobbl thermal

```sh
wobbl thermal pack.step -m li-ion-cell --power "12 W" --film "5 W/m2K @ all" \
  --ambient "35 C" --duration "45 min" --limit "60 C" --radiation 0.9
```

| Option | Meaning |
|---|---|
| `--power` | heat made inside the part (spread through the volume) |
| `--heat-faces` | put the power on these faces instead (a chip on a heat sink) |
| `--film` | convection, repeatable: `5 W/m2K @ all`, `25 W/m2K at 30 C @ top`. Still air in a box about 5, open air 10, a small fan 25, strong forced air 60, water 500 |
| `--fixed-temp` | held temperature: `40 C @ bottom` |
| `--duration` | transient run length; omit for steady state |
| `--limit` | temperature limit (default: the material's service limit) |
| `--radiation` | surface emissivity, 0.9 for painted or plastic surfaces |

## wobbl circuit

```sh
wobbl circuit led    --set supply=5V --set resistor=100 --set led=blue
wobbl circuit rc     --set r=10k --set c=10u --set v=5
wobbl circuit buck   --set vin=12 --set vout=5 --set current=2 --set frequency=500k \
                     --set inductor=10u --set capacitor=22u --set max_ripple=0.05
wobbl circuit mosfet --set current=15 --set rds_on=8m --set frequency=20k --set theta_ja=40
wobbl circuit board.cir --probe "v(out)" --probe "i(V1)"
```

LED colors: red, orange, yellow, green, blue, white, uv, ir, power-white,
power-red; override with `vf=` and `imax=`. Any netlist with a `.tran` is
re-run at half the time step to check convergence.

## Other commands

| Command | Does |
|---|---|
| `wobbl check [DIR]` | the checks, confidence and reasons of a finished run (latest by default) |
| `wobbl report [DIR] --open` | rebuild and open the HTML report |
| `wobbl materials` | the materials table |
| `wobbl validate` | run the validation suite (see [validation](validation.md)) |
| `wobbl setup` | install solvers into `~/.wobbl` and register agents |
| `wobbl doctor` | check every piece and print what is missing |
| `wobbl uninstall [--all]` | remove agent registrations (and `~/.wobbl` with `--all`) |
| `wobbl mcp` | run the MCP server on stdio |
