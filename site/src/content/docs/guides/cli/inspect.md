---
title: "wobbl inspect"
description: "List the size, walls, holes, fillets, sharp inside corners and face ids of a part, to write loads and supports against."
lead: "inspect reads a part and lists everything you can point a load or a support at."
order: 3
---

```sh
wobbl inspect PART [--json] [--scale N] [--body N]
```

Run it before a study. It tells you which selectors will work on your part and catches a unit
mistake before you spend minutes solving.

```text
bracket.py  90 x 25 x 80 mm, volume 24.42 cm3, walls about 6 mm (thinnest 6 mm)
  Built from bracket.py with thickness=6, width=25, height=80, depth=90, fillet=0.5, hole=4.5.

HOLES
  d 4.5 mm at (3, 12.5, 25)  select with 'hole d=4.5' or 'face 10'
  d 4.5 mm at (3, 12.5, 65)  select with 'hole d=4.5' or 'face 11'

INSIDE FILLETS
  radii 0.5 mm

FACES
 id  kind      area mm2  center          normal or size
 8   plane     2250      45, 12.5, 0     +0.00, +0.00, -1.00
 5   plane     2088      48.3, 12.5, 6   +0.00, +0.00, +1.00
 ...
 3   cylinder  19.13     6.2, 12.5, 6.2  r 0.5 (fillet)
```

## What it lists

The first line gives the bounding box, the volume and the wall thickness, typical and thinnest,
followed by any notes about repairs, bonded bodies or a suspicious size. Then come the holes
with their diameters and centers, the inside fillet radii, the sharp inside corners with the
two faces that meet there, which are the likely stress raisers, and the largest faces with their
ids, kinds, areas, centers and normals. The terminal shows 40 faces at most, and `--json` lists
every one.

## Options

| Option | What it does |
|---|---|
| `--json` | prints everything as JSON, which is what agents get from `inspect_geometry` |
| `--scale` | scales the geometry first, such as `25.4` for a file in inches |
| `--body` | picks one body from a file that holds several |
