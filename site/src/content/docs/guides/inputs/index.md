---
title: "Inputs"
description: "How to give Wobbl a part, its material, the supports that hold it, the loads on it and the faces they act on."
lead: "A study is only as good as its setup, so these pages cover each input and the short text syntax Wobbl uses for it."
cards:
  - title: "Geometry"
    href: "/guides/inputs/geometry/"
    text: "STEP, IGES, BREP, STL and parametric Python parts, and what Wobbl does when it reads them."
  - title: "Loads"
    href: "/guides/inputs/loads/"
    text: "Forces, pressures, gravity, and forces or torques at a point off the part."
  - title: "Supports"
    href: "/guides/inputs/supports/"
    text: "Fixed faces, bolted holes, rollers and walls the part can lift off."
  - title: "Materials"
    href: "/guides/inputs/materials/"
    text: "The 23 built in materials, custom properties and printed parts."
  - title: "Faces"
    href: "/guides/inputs/faces/"
    text: "The selector language that points loads and supports at the right faces."
  - title: "Units"
    href: "/guides/inputs/units/"
    text: "Millimeters for geometry and any common unit everywhere else."
---

Loads and supports are written as short phrases, a magnitude or a kind followed by `@` and
the faces it acts on.

```sh
wobbl run bracket.py -m pla \
  --fix "bolt holes d=4.5" --fix "wall x=min" \
  --load "40 N down @ top and within 12 of 65,12.5,6"
```

The same phrases work in the MCP tools, so an agent writes them exactly the way you would.
Run [`wobbl inspect`](/guides/cli/inspect/) on a part first, since it lists the holes,
fillets and faces you can point at.
