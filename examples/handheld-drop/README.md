# Handheld drop

Will this handheld survive a 1.2 m drop onto concrete with its 45 g battery?

A sealed polycarbonate housing, 100 x 55 x 22 mm with 2.5 mm walls and rounded vertical corners, holds its
battery between two ribs on the inside of its floor. The battery is a 45 g mass spread over the floor and ribs
under it, and the accelerometer sits there too, on the battery mount. 1.2 m is the height most handheld drop
specifications start from.

```sh
./run.sh
```

RESULTS

Wobbl gives the housing the speed it reaches after 1.2 m, 4.85 m/s, at the moment it first touches the floor, and
marches its motion through the impact: lumped mass, explicit time steps of a fraction of a microsecond, and the
floor pushing back on every surface node that goes below it. No landing was named, so it screens every face and
corner landing on a linear tet copy of the mesh, using the housing's mirror symmetry to skip the landings that
are the same, runs the worst ones again on quadratic tets, and refines the mesh around the peak of the worst.
Each run is checked by momentum (the floor's impulse against the change of momentum) and by energy.

The run also writes `output/playback.json` and `output/scene.wobbl`: the housing falling, landing on its worst
corner and the stress flashing through it, with the accelerometer's reading at the battery mount, frame by
frame, from the solved motion.
