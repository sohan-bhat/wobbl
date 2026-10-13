---
title: "Limits"
description: "What Wobbl does not model yet, from plasticity and fatigue to bolt preload and contact between parts, and what is planned next."
lead: "Wobbl tells you when its answer is not the whole story, and this page lists what it does not model yet."
order: 3
---

## Materials

Materials are linear elastic, so Wobbl reports whether yield is reached but not how the load
spreads once a spot yields, and buckling is not modeled. A part that bends a good fraction of its
wall is solved again with large deflection theory, but the material stays elastic. Each study uses one material, and touching bodies in an assembly are bonded together
as one part.

Printed parts are treated as solid and isotropic at 100% infill, so a part printed with less
infill is weaker than the result assumes. Pass the print direction and Wobbl judges the pull
across the layers against a weaker layer strength.

## Loads and supports

Loads are static and applied all at once, so fatigue, impact and vibration are not modeled yet.
Contact is limited to rigid walls the part can push on and lift off. Bolted joints are clamped
holes without preload, and contact, friction and preload between separate parts are not
modeled.

## Speed

Thin walls are slow. Shells 2 to 3 mm thick on parts around 100 mm across need large meshes of
tetrahedra and take several minutes, and shell elements are planned to fix that.

## Repeatability

The same study gives the same numbers on every run on one machine. Another machine with a
different build of gmsh, OpenCASCADE or CalculiX can move them slightly.

## What is next

Shell elements, fatigue estimates and bolt preload are planned for 0.2, CFD with OpenFOAM for
0.3, and motion with MuJoCo for 0.4.
