# Motor mount

Do the ribs on a printed NEMA 17 mount earn their keep?

This NEMA 17 stepper mount is printed in PETG with 5 mm walls and bolted down through four
holes. The motor's face presses on the upright, its pulley sits 15 mm behind it, and the belt
pulls 250 N straight down. The same part is run with its two 4 mm ribs and without them
(`--param rib=0`). Printed layers stack along z, so pull across the layers is judged against
the weaker layer strength.

```sh
./run.sh
```

| Version | Peak stress | Safety factor (target 2.5) | Verdict |
|---|---|---|---|
| With ribs | 11.0 MPa | 4.3 | holds |
| Without ribs | 41.3 MPa on the 3 mm fillet at the upright's base | 1.14 | marginal, since it survives once with no margin for a printed part |

The ribs cut the peak stress by nearly a factor of four. Wobbl also tells you what the answer
leans on.

- The motor face is modeled as rigid. If the flange were flexible enough to spread the load
  evenly, the ribbed mount would see 18.8 MPa (safety factor 2.50, a hair under the target)
  and the bare one 73.6 MPa, each converged over three meshes refined around its own peak.
  For the ribbed version that sits right at the target, so the result is marked medium
  confidence with that reason.
- The edges of the bolted washer rings and of the rigid motor face read up to 54 MPa and climb
  as the mesh refines. Those are artifacts of perfectly rigid supports, so Wobbl lists them and
  reports the stress away from them.
- The fillet peak still moves between meshes, through 11.1, 10.3 and 11.0 MPa with ribs and
  45.4 and 41.3 MPa without, so Wobbl reports the last change, 6.6% and 9.7%, as the likely
  error. With ribs the verdict holds even at the highest value seen, and without them every
  mesh's value is marginal.
