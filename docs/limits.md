# Limits

What Wobbl does not model yet, so you know when its answer is not the whole story.

- **Linear elastic materials.** Wobbl reports whether yield is reached, not what
  happens after. Large deflection is checked with a nonlinear re-solve.
- **Static loads.** Fatigue, impact and vibration are not modeled yet.
- **Rigid wall contact only.** A part can push on a wall and lift off it. Bolted
  joints are clamped holes, without preload.
- **One material per study.** Assemblies of touching bodies are bonded together.
- **Thin walls are slow.** 2 to 3 mm shells on 100 mm parts need large
  tetrahedral meshes and take several minutes. Shell elements are next.
- **Printed parts are idealized.** They are treated as isotropic at 100% infill,
  with a layer strength knockdown when you give the print direction.
- **Repeatable on one machine.** The same study gives the same numbers on every
  run. Another machine with different gmsh, OpenCASCADE or CalculiX builds can
  move them slightly.

Next: shell elements, fatigue estimates and bolt preload in 0.2, CFD with
OpenFOAM in 0.3, motion with MuJoCo in 0.4.
