---
title: "Geometry"
description: "The CAD and mesh files Wobbl reads, how it repairs and prepares them, and how to write a parametric Python part."
lead: "Wobbl reads STEP, IGES, BREP and STL files, and parts written as a short Python file it can rebuild with new dimensions."
order: 1
---

| Format | Extensions | Notes |
|---|---|---|
| STEP | `.step`, `.stp` | read through OpenCASCADE, the best choice from any CAD tool |
| IGES | `.iges`, `.igs` | read through OpenCASCADE |
| BREP | `.brep`, `.brp` | OpenCASCADE's own format |
| STL | `.stl` | has to be watertight, and has no units, so Wobbl assumes millimeters |
| Parametric | `.py` | a Python file that builds the part from named dimensions |

OBJ and PLY files are refused with a request to export STEP or STL instead. Geometry is
always read in millimeters, so use `--scale 25.4` for a file drawn in inches and
`--scale 1000` for one in meters. A part under 1 mm or over 20 m across gets a note
suggesting exactly that.

## What happens when a file is read

Wobbl imports the file as it is and repairs it only when a solid is broken, because the import
time fixes can damage good solids. A file that holds only surfaces is sewn into a solid when
the surfaces close, and if no valid closed solid comes out you get an error asking for one.
Several solids that touch are bonded into one part, glued at their interfaces. Solids that do
not touch are an error, and `--body N` picks one of them. Faces far too small to matter for
strength, such as embossed text, logos and slivers, are removed, while inside fillets and
holes are always kept because they set the peak stress. Each of these steps leaves a note in
the result, so you can see what was changed.

STL is triangle soup, so Wobbl groups the triangles into patches by a 30 degree feature angle
before it turns them into a volume. If that fails, the STL is not watertight and needs a
repair, for example in PrusaSlicer, Meshmixer or Blender's 3D Print Toolbox.

## Parametric parts

A part can be a Python file with a `PARAMS` dictionary of dimensions in millimeters and a
`build` function that takes gmsh's OpenCASCADE kernel, with calls like `addBox`, `addCylinder`,
`fuse`, `cut` and `fillet`. This is the start of the shelf bracket from the examples.

```python
PARAMS = {
    "thickness": 6.0,      # both legs
    "width": 25.0,
    "height": 80.0,        # upright
    "depth": 90.0,         # arm, measured from the wall
    "fillet": 0.5,         # inside corner radius
    "hole": 4.5,           # screw clearance holes in the upright
}


def build(occ, thickness, width, height, depth, fillet, hole):
    t = thickness
    arm = occ.addBox(0, 0, 0, depth, width, t)
    upright = occ.addBox(0, 0, 0, t, width, height)
    body, _ = occ.fuse([(3, arm)], [(3, upright)])
    ...
```

Wobbl builds a STEP file from it and caches that file by its parameter values. Change a
dimension for one run with `--param`, which can be repeated.

```sh
wobbl run bracket.py --param fillet=6 -m pla --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6"
```

Parametric parts also let `wobbl fix --apply` change the part by changing its numbers. It
treats a parameter whose name contains `thick` or `wall`, or is `t` or starts with `t_`, as
the wall thickness, and one whose name contains `fillet` or `radius`, or is `r` or starts with
`r_`, as the corner radius. The [examples](/examples/) are complete parts to start from.
