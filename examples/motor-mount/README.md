# Motor mount: do the ribs earn their keep?

A NEMA 17 stepper mount printed in PETG, 5 mm walls, bolted down through four
holes. The motor's face presses on the upright; its pulley sits 15 mm behind it
and the belt pulls 250 N straight down. The same part is run with its two 4 mm
ribs and without them (`--param rib=0`). Printed layers stack along z, so pull
across the layers is judged against the weaker layer strength.

```sh
./run.sh
```

| Version | Peak stress | Safety factor (target 2.5) | Verdict |
|---|---|---|---|
| With ribs | 10.7 MPa | 4.4 | holds |
| Without ribs | 43.8 MPa on the 3 mm fillet at the upright's base | 1.07 | marginal: it survives once, with no margin for a printed part |

The ribs cut the peak stress by a factor of four.

Wobbl also tells you what the answer leans on:

- **How the motor carries the load.** The motor face is modeled as rigid. If
  the flange were flexible enough to spread the load evenly, the ribbed mount
  would see 19.5 MPa (safety factor 2.4) and the bare one 70 MPa. For the ribbed
  version that sits right at the target, so the result is marked medium
  confidence with that reason.
- **Where the idealization is extreme.** The edges of the bolted washer rings
  and of the rigid motor face read up to 52 MPa and climb as the mesh refines.
  Those are artifacts of perfectly rigid supports; Wobbl lists them and reports
  the stress away from them.
