# Getting started

## 1. Install

```sh
pipx install "git+https://github.com/sohan-bhat/wobbl"     # or: uv tool install "git+https://github.com/sohan-bhat/wobbl"
wobbl setup
wobbl doctor
```

`wobbl setup` puts CalculiX and ngspice in `~/.wobbl` and registers Wobbl with
the coding agents it finds. Run `wobbl setup --no-agents` to skip that, or
`wobbl setup --agents claude,cursor` to choose.

## 2. Look at your part

```sh
wobbl inspect bracket.step
```

You get the size, wall thickness, holes with their diameters, fillet radii,
sharp inside corners and the largest faces with ids. These tell you how to
point at faces: `hole d=4.5`, `x=min`, `top and within 12 of 65,12,6`, `face 9`.
Files are read in millimeters; pass `--scale 25.4` for an STL drawn in inches.

## 3. Ask

```sh
wobbl run bracket.step -m pla --fix "bolt holes d=4.5" --load "40 N down @ top and within 12 of 65,12,6"
```

Choose supports that match how the part is really held: bolted holes, a face
clamped flat, a face that rests on a wall and can lift off. Wobbl refines the
mesh until the answer settles, checks itself and prints:

- the verdict (holds, marginal, fails) against the material's limit and your `--target-sf`
- the governing stress and where it is
- the confidence and the reasons for it
- every check, the assumptions, and fixes with numbers
- the paths of the heatmaps and the HTML report

## 4. Fix

```sh
wobbl fix wobbl-out/bracket --apply
```

Wobbl rounds the corner, adds a gusset or swaps the material, re-runs each
version and shows a before and after table. Edited STEP files are saved next to
the results so you can open them in your CAD tool.

## 5. Heat and circuits

```sh
wobbl thermal heatsink.step -m 6063-t5 --power "15 W" --heat-faces "bottom and box 17,17,-1,43,43,1" \
  --film "6 W/m2K @ all" --ambient "40 C" --radiation 0.85 --limit "85 C"
wobbl circuit buck --set vin=12 --set vout=5 --set current=2 --set capacitor=22u
```

## Parametric parts

Instead of a STEP file, a part can be a short Python file:

```python
PARAMS = {"thickness": 6.0, "fillet": 0.5}

def build(occ, thickness, fillet):
    arm = occ.addBox(0, 0, 0, 90, 25, thickness)
    ...
```

`occ` is OpenCASCADE through gmsh (`addBox`, `addCylinder`, `fuse`, `cut`,
`fillet`, `extrude`). Run it like any part, override with `--param fillet=3`,
and `wobbl fix --apply` changes the thickness and fillet parameters itself.
See `examples/*/` for complete parts.
