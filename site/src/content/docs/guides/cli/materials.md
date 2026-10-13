---
title: "wobbl materials"
description: "Print Wobbl's built in material table with stiffness, strength limit, criterion, density, conductivity and service temperature."
lead: "materials prints the table of built in materials with the numbers Wobbl uses for each."
order: 9
---

```sh
wobbl materials
```

Each row gives the key you pass to `-m`, the name, how the material is made, its stiffness E in
MPa, its strength limit in MPa, whether it is judged by yield or by tensile strength, its
density in kg/m3, its thermal conductivity in W/(m K) and its maximum service temperature in C,
which heat studies use as their default limit. Agents get the same table from
`list_materials`. [Materials](/guides/inputs/materials/) shows the table with each material's
default safety factor target and how to give custom properties.
