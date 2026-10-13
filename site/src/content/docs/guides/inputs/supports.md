---
title: "Supports"
description: "How to hold a part in a Wobbl strength study with fixed faces, bolted holes, rollers and walls it can lift off."
lead: "Hold the part the way the real one is mounted, because the supports decide the answer as much as the load does."
order: 3
---

```sh
--fix "bolt holes d=4.5" --fix "wall x=min"
```

A support is an optional kind followed by the faces it holds. With no kind it is fixed, so
`--fix "x=min"` holds every point of the faces at the minimum x still. Pass `--fix` as many
times as you need.

| Kind | Example | What it means |
|---|---|---|
| fixed | `x=min`, `holes d=4.5` | every point on the faces held still |
| fixed in some directions | `fixed-z faces 7`, `fixed-xy faces 3,4` | only those directions held, the others free |
| bolt | `bolt holes d=6.6` | the hole and the washer ring around it held still, with an outer radius twice the hole's |
| roller | `roller y=0` | slides in its plane but cannot move off it, for symmetry planes and sliding fits |
| wall | `wall x=min` | rests on a rigid wall that it can push on and lift off, solved as contact |

The words `fix`, `bolted`, `slider`, `symmetry`, `rest` and `support` also work, as
`fixed`, `bolt`, `roller`, `roller`, `wall` and `wall` respectively. Only fixed supports can
hold some directions and not others.

## Picking the right kind

A fixed face is perfectly rigid, which is stiffer than any real mount, so the stress right at
its edge climbs as the mesh gets finer. Wobbl knows this and reports the stress a short
distance away, listing the edge peak separately as an artifact of the support. A bolted hole
is the better choice for screws and bolts, because only the hole and the washer ring are
held. At bolted and fixed holes Wobbl also runs a bearing check, which divides the force
through each hole by its projected area and compares that with the bearing limit.

A wall is for faces that rest against something without being attached, like the back of a
shelf bracket against the wall it hangs on. The part can push into the wall and lift away
from it, which Wobbl solves as contact. A coarse first solve finds the patch in contact, every
mesh starts from the patch the last one found, and the loop stops when fewer than 2% of the
contact nodes change state.

## A part that is not held

Wobbl refuses to solve a part that can still slide or spin, and names the motion that is free.

```text
the fixtures leave the part free to spin about z. Add a fixture that stops it, for example fix a hole or a face that is not parallel to the existing ones
```

Two rollers on parallel faces, for example, still let the part slide along them. Add a
support that stops the motion it names.
