---
title: "Motion"
description: "Drops, throws and rocket flights under gravity, air, wind, thrust and parachutes, integrated with scipy and checked against closed forms."
lead: "A motion study tells you how something falls, flies or launches, from a rock's impact speed to a rocket's apogee, stability and landing."
order: 4
---

## A rock off a cliff

```sh
wobbl motion --name rock --diameter "9 cm" --density "2700 kg/m3" \
  --cd 0.6 --height "300 m"
```

A body is a diameter with a mass, or with a density for a solid ball, and a drag coefficient on
its frontal area. The [falling rock example](/examples/falling-rock/) is a fist sized lump of
granite taken as a sphere 9 cm across, so it weighs 1.03 kg, dropped from rest off a 300 m cliff.
A tumbling rock averages a drag coefficient of about 0.6, and the air thins slightly with height
as the 1976 standard atmosphere has it.

```text
OK. Hits the ground 8.72 s after release at 56.8 m/s (205 km/h). Without air it would land after 7.82 s at 76.7 m/s, so drag takes 26% off the impact speed.

Confidence: high. passed step convergence, hand calculation, vacuum drop, energy balance, units and sanity
```

![Altitude and speed of the falling rock over 8.72 s, with the fall without air dashed and landing at 7.82 s](/img/falling-rock-flight.webp "The falling rock, with the same drop without air dashed.")

`--speed` and `--angle` turn the drop into a throw, as in
`wobbl motion --name ball --mass "145 g" --diameter "7.4 cm" --speed "40 m/s" --angle 35`, and
the answer says how far away it lands, when, how high it rises and how far it would go without
air. `--wind` adds a steady wind blowing back toward the start, negative when it blows from
behind.

## A rocket on an L motor

```sh
wobbl motion rocket.toml
```

A rocket is a spec file with its airframe, motor, launch rail and parachutes, every quantity
written with its unit. The [rocket launch example](/examples/rocket-launch/) is a level 2 high
power rocket, a 98 mm airframe 2.1 m long on a 75 mm L class motor of 3773 N s. It weighs
7.95 kg on the pad, leaves an 8 ft rail tilted 5 degrees into a 10 mph wind, opens an 18 inch
drogue at apogee and a 60 inch main at 150 m on the way down.

```toml rocket.toml
[launch]
rail_length = "2.44 m"     # an 8 ft rail
rail_angle = "5 deg"       # tilted from vertical, leaning into the wind
wind = "10 mph"            # blowing toward the pad from the side the rail leans to

[drogue]
diameter = "46 cm"         # an 18 inch drogue, opened at apogee
cd = 1.5

[main]
diameter = "1.52 m"        # a 60 inch main
cd = 1.5
deploy = "150 m"           # above the ground, on the way down

[targets]
apogee = "11000 ft"
landing_speed = "6.1 m/s"  # about 20 ft/s
```

```text
OK. Reaches 3439 m (11284 ft) at Mach 1.27 and lands 29.3 m upwind of the pad at 5.72 m/s, with a 1.88 caliber margin and 30.6 m/s off the rail.

Confidence: high. passed step convergence, delta v budget, parachute descent, energy balance, units and sanity
```

![Altitude and speed of the rocket over its 203 s flight, colored by phase from the rail to the main parachute](/img/rocket-flight.webp "The rocket launch, by phase from the rail to the main parachute.")

Off the rail the rocket turns into the wind it feels, so it climbs upwind, reaches 757 m from the
pad at apogee and drifts back under its parachutes. The trajectory, with the speed, Mach number
and dynamic pressure at every row, is in `trajectory.csv` next to the report.

## The spec file

| Table | Keys |
|---|---|
| `[rocket]` or `[body]` | `mass`, or `density` for a solid ball, `diameter`, `cd` as a number or a table against Mach, and for a rocket `length` and `cg`, its balance point from the nose tip with the motor loaded |
| `[nose]` | `shape`, one of cone, ogive, elliptical, parabolic or haack, and `length` |
| `[fins]` | `count`, `root`, `tip`, `span`, `sweep` and `position` |
| `[motor]` | `name`, `thrust` as pairs of seconds and newtons, `eng` for a RASP file from thrustcurve.org or `design` for a motor designed with `wobbl motor`, and `loaded_mass`, `propellant` and `length` |
| `[launch]` | `rail_length`, `rail_angle` from vertical or `angle` above level, `height`, `speed`, `wind` and `site_altitude` |
| `[drogue]`, `[main]` | `diameter`, `cd` and `deploy`, either apogee or a height above the ground |
| `[targets]` | `apogee`, `landing_speed`, `rail_exit_speed` and `stability` in calibers |
| `[world]` | `air`, standard, vacuum or a density, and `gravity`, falls off or uniform |

The `[nose]` and `[fins]` tables, with the balance point, give the stability margin by Barrowman's
method and the rate the rocket turns into the wind. A spec without them flies, but its rocket
points into the wind at once. An agent can pass the same TOML as text to the `solve_motion` tool.

## What the model includes

The body is a point mass in a vertical plane, so it does not spin, tumble or move sideways.
Gravity falls off with the square of the distance to the Earth's center, and the air follows the
1976 standard atmosphere from the site's altitude. Drag acts against the air rushing past, with
the drag coefficient held constant, read from a table against Mach, or for a rocket raised
through the transonic range by a generic curve for a slender finned body. A rocket rides its rail
until it leaves it, burns propellant in proportion to thrust so all of it is gone at burnout, and
turns into the relative wind at its weathercock frequency. Each parachute opens over the time it
takes to travel eight of its diameters, and the drogue stays on under the main.

## The verdict

A drop or a throw always says ok, since it reports what happens. A rocket's verdict comes from
its rocketry checks.

| Verdict | When |
|---|---|
| ok | every rocketry check passes |
| marginal | the rocket leaves the rail between 9 and 15 m/s, lands faster than its limit but under 1.5 times it, or misses its apogee target by more than 5% |
| fails | the stability margin is under 1 caliber, the rocket leaves the rail under 9 m/s, or it lands at more than 1.5 times its limit or with no parachute |

`[targets]` changes the limits. When a rocket fails or is marginal, the fixes say how much nose
weight or fin span brings the margin to 1.5 calibers, how long a rail it needs, or how big a main
lands it under the limit.

## How motion answers are checked

| Check | What it compares |
|---|---|
| Step convergence | the whole flight again with a tolerance ten times tighter, where the key numbers have to move by under 0.1% |
| Energy balance | the work done by thrust, drag and gravity, less what the burned propellant carried off, against the change in kinetic energy, within 0.1% |
| Hand calculation | for a drop from rest in still air, the quadratic drag solution with the air's density halfway down, within 1% |
| Vacuum drop or flight | the same body without air and with gravity at its ground value, against the closed form for a fall or a throw |
| Delta v budget | for a rocket, Tsiolkovsky's ideal delta v less what gravity, drag and steering took, against the speed at burnout, within 0.5% |
| Parachute descent | the sink rate under each parachute against its terminal speed `sqrt(2 m g / (rho Cd A))`, within 1% |
| Units and sanity | that the mass, density, speeds and burned propellant are physical |

A rocket also gets its stability margin, its speed off the rail, its apogee against the target,
its landing speed and, as information, its peak dynamic pressure, peak acceleration and drift.
[Checks](/guides/results/checks/) covers them, and [validation](/validation/) flies six cases
with exact answers through the same code.
