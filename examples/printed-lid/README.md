# Printed lid

Will this long ABS lid warp or lift its corners when I print it on an open printer?

The lid is 220 mm by 60 mm, a 2.5 mm top with a 2 mm lip 12 mm high around it. It prints face down on a 100 C bed in
open air, solid, with the layers stacking along z. Long flat prints are where ABS lets go of the bed first.

```sh
./run.sh
```

| Result | Value |
|---|---|
| Verdict | FAILS, 3.81 mm off flat against a 0.8 mm tolerance, medium confidence |
| Corner lift | 3.81 mm once it comes off the bed |
| Pull on the bed | 339 N at the far corner while it prints, 15.8 MPa of stress in the part |
| Mesh convergence | 3.79 then 3.81 mm over two meshes, the last change 0.4% |
| Hand calculation | a free plate curved by the same shrinkage gives 3.62 mm, the solve 3.81 mm (1.05 times) |
| Fixes | an enclosure near 45 C brings it to about 2.86 mm, PETG to about 1.75 mm, a brim and a hot bed keep the corner stuck while it prints |

The warp model is fitted to the 48 ABS plates Armillotta et al. measured, with a median error of 9%. Printer and
filament settings come from a slicer file next to the part when there is one (Cura, Orca, Bambu, Creality, PrusaSlicer),
and otherwise from the defaults listed under the assumptions in `output/summary.txt`.

Open `output/simulation.html` to see the lid come off the bed in 3D, played from `output/scene.wobbl`.
