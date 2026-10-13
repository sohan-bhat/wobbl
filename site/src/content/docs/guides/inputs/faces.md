---
title: "Faces"
description: "The face selector language that points Wobbl loads, supports, heat inputs and films at the right faces of a part."
lead: "Everything after the @ in a load or support picks faces by what they are and where they are, not by fragile ids."
order: 5
---

```text
top   x=min   z>5   normal +z   hole d=4.5   holes d<6   fillets
face 12   within 15 of 50,-25,35   top and within 12 of 65,12.5,6
```

Run `wobbl inspect` first and it shows which of these exist on your part. When a selector
matches nothing, the error lists the hole sizes and the largest flat faces that are there.

## Every selector

| Selector | Picks |
|---|---|
| `top` `bottom` `left` `right` `front` `back` | the flat outer faces at that extreme, with z up and the front facing minus y |
| `x=min` `z=max` `y=12.5` | faces lying in that plane |
| `x>100` `z<5` `y>=0` | faces whose center is past that coordinate |
| `normal +z`, `normal 0,0.7,0.7`, `facing down` | flat faces whose outward normal points that way |
| `holes`, `hole d=4.5`, `holes d<6`, `hole near 55,0,10` | holes, by diameter or by the one closest to a point |
| `bosses`, `fillets`, `cylinders r=3`, `flats`, `curved` | faces by shape, with `d` or `r` and `=`, `<`, `>`, `<=` or `>=` |
| `largest`, `largest flat`, `smallest`, `area>100` | faces by size in mm2 |
| `near 10,20,5` | the single face closest to a point |
| `within 15 of 50,-25,35` | only the part of faces within 15 mm of the point, a round patch |
| `box 0,0,0,10,20,5` | only the part of faces inside the box, given as two corners |
| `face 12`, `faces 3,5,7` | face ids from `wobbl inspect` |
| `all` | every face |

A size like `hole d=4.5` matches within 5% or 0.1 mm, whichever is larger, so a hole modeled
at 4.52 mm still counts.

## Combining them

`and` keeps faces that match both sides and `or` keeps faces that match either. The patch
clauses, `within` and `box`, are applied first, so `top and within 14 of 210,0,25` means the
highest flat face near that point, cut down to a 28 mm circle. That is the usual way to put a
load on a small pad of a larger face.

```sh
--load "40 N down @ top and within 12 of 65,12.5,6"
--fix "bolt holes d=3.2 and x<0"
--heat-faces "bottom and box 17.5,17.5,-1,42.5,42.5,0.5"
```

## Ids versus names

Face ids come from the CAD file and change whenever the part changes, so a selector by name or
position keeps working after an edit while an id may not. When `wobbl fix --apply` changes the
geometry, Wobbl carries each selector over to the new faces by itself.
