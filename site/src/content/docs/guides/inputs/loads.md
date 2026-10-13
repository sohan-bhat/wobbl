---
title: "Loads"
description: "How to write forces, pressures, gravity, remote forces and torques for a Wobbl strength study."
lead: "A load is a magnitude with its unit, a direction, and after an @ the faces it acts on."
order: 2
---

```text Flag
--load "200 N down @ top"
```

Pass `--load` as many times as you need, and Wobbl adds the loads together. Each one is
written the same way, in this order.

```text Syntax
MAGNITUDE UNIT  DIRECTION  [bearing]  [at X,Y,Z]  @ FACES  [rigid]
```

## Kinds of load

Wobbl tells the kinds apart by the unit, by whether there is a point, and by the word `bearing`.

| Kind | Example | What it means |
|---|---|---|
| force | `200 N down @ top` | the total force, spread evenly over the faces by area |
| force along the normal | `20 N normal @ face 4` | pushes into each face along its own normal, and a negative value pulls out |
| force into holes | `300 N -x @ hole d=6.6` | the force spread evenly over the hole faces, with the peak where it meets the hole edge reported separately |
| bearing load | `300 N -x bearing @ hole d=6.6` | a bolt or pin pressing on the side of the hole the force points to, shared equally by the holes picked |
| pressure | `0.2 MPa @ top` | pushes into the faces, and a negative value pulls |
| gravity | `3 g down` | the part's own weight times the factor, with no faces needed |
| remote force | `2000 N -0.87,0,-0.5 at 0,-30,66 @ holes d=6.6` | a force at a point off the part, carried by the faces |
| moment | `0.8 N*m +z at 210,0,25 @ top and within 14 of 210,0,25` | a torque about an axis through the point |

Gravity needs no faces and points down unless you say otherwise. The `g` means gravity only
when there is no `@`, because with faces the same letter reads as grams of force. A moment
needs an axis, and when you leave out the point it turns about the center of its faces.

## Bolts and pins in holes

A bolt or a pin that pulls on a part presses on one side of its hole, and the far side carries
nothing. Write `bearing` after the direction, or at the very end after the faces, and Wobbl
puts the force on the loaded half of each hole with a cosine distribution that peaks where the
bolt pushes hardest. When you pick several holes they share the force equally, and the result
lists this in its assumptions.

```text Flags
--load "300 N -x bearing @ hole d=6.6"
--load "1.2 kN -y @ holes d=8 bearing"
```

Without `bearing` the same string spreads the force evenly over the whole hole, all the way
around, which suits a flange or a press fit better than a single bolt. A bearing load needs
hole faces and a force unit, so Wobbl refuses it with a pressure, a moment, gravity or a point
off the part.

## Directions

A direction is one of `+x -x +y -y +z -z`, a word, or a vector like `0,0.7,-0.7` that does not
need to be unit length. The words follow the usual view where z is up and the front of the
part faces minus y.

| Word | Direction |
|---|---|
| `up`, `down` | `+z`, `-z` |
| `right`, `left` | `+x`, `-x` |
| `back`, `forward` | `+y` |
| `front`, `backward` | `-y` |
| `normal` | into each face along its normal (forces only) |

## Loads that act off the part

A remote force or a moment acts at a point that is not on the part, like a belt pull at a
pulley or the torque of a motor on its flange. It reaches the part through the faces after the
`@`, and there are two ways it can get there.

By default the load is distributing, which means the exact force and moment are spread over
the faces without making them any stiffer, the way an RBE3 element does. That is close to a
bolt group sharing a load or a flexible part pressing on the faces. Add `rigid` at the end and
the faces are tied to the point as one rigid body instead, through a CalculiX
`*RIGID BODY`, which suits a stiff motor flange or a clamp. Several rigid loads on the same
faces merge into one attachment, which is how the drone arm example puts both thrust and
torque on its motor pad.

```text Flags
--load "25 N up at 150,0,15 @ top and within 15 of 150,0,15 rigid" \
--load "0.5 N*m +z at 150,0,15 @ top and within 15 of 150,0,15 rigid"
```

Either way, Wobbl also solves the other idealization and reports how far apart the two are,
and the confidence drops when the verdict depends on the choice.

## Units

Any common unit works. Forces take `N`, `kN`, `lbf`, `kip` and `kgf`, pressures take `MPa`,
`kPa`, `Pa`, `GPa`, `psi`, `ksi` and `bar`, moments take `N*m`, `N*mm`, `in-lb` and `ft-lb`,
and gravity takes `g` or `m/s2`. A load in imperial units makes the whole result print in
psi and inches, with the metric values next to them. [Units](/guides/inputs/units/) has the
full list.
