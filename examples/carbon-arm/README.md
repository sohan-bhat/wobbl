# Carbon fiber drone arm

Is this carbon fiber arm strong enough for 120 N at the motor?

The arm of a heavy lift drone is a carbon tube 400 mm long and 25 mm across, laid up from eight plies of
T300/5208 carbon epoxy as [0/±45/0]s, 1 mm in all. Its root is clamped 25 mm deep in the hub, and the motor
mount at the far end pushes up with 120 N, three times the motor's hover thrust, the kind of load a hard climb or
a gust puts on it.

```sh
./run.sh
```

| Result | Value |
|---|---|
| First ply failure | 5.17 times the load, by Tsai-Wu, against a target of 1.5, so it HOLDS |
| Where | the outer 0 degree ply on top of the tube, just past the clamp, where it is squeezed along its fibers |
| Hashin | 8.88 times the load, by fiber compression, at the same spot |
| Mesh convergence | 4.70, 5.01 then 5.17 times over three midsurface meshes, the last step 3% |
| Hand calculation | beam theory with the laminate's own moduli from lamination theory (104 GPa stretching, 118 GPa bending) gives 750 microstrain at a section 120 mm from the clamp; the shell model gives 751 |
| Force balance | reactions match the load to 4e-9% |
| Deflection | 3.79 mm at the motor |
| Mass | 48 g |

Wobbl puts the laminate on the midsurface of the tube's walls, solves it with CalculiX composite shells, and
reads the strain through the thickness at every point. Each ply's stresses along and across its fibers come
from that strain, and Tsai-Wu and Hashin turn them into the factor on the load at which the first ply cracks.
Tsai-Wu goes first here because the ±45 plies make the 0 degree plies carry a little stress across their
fibers, where the matrix is weak, on top of the squeeze along them. The clamp itself reads lower, 4.05 times,
but that is how an ideal rigid clamp holds a laminate, so Wobbl reports the value away from it and says so.

The layup has no 90 degree plies, which Wobbl flags: loads off the arm's axis, such as a crash, fall on the
matrix. The run also writes `output/playback.json`, the laminate loading from nothing to first ply failure:
the midsurface, the solved displacement, and for every ply its failure index at the applied load, which a viewer
scales by the load factor. The study is linear, so that is exact up to first ply failure.

To try a different stack, change `--layup`, for example `[0/±45/90]s`, or `--ply im7-8552`. The same study runs
through `wobbl run arm.py --layup ...` with the same loads and supports.
