# Cold plate

Does a liquid cooled plate keep 150 W of electronics under 60 C on 2 L/min of water?

The plate is a 120 x 70 x 16 mm aluminum block with an 8 mm channel drilled in a U, both ports on one end.
Nothing in the question says where the water goes in. Wobbl finds the two ports as holes in the end face whose far
side is empty, caps them, follows the channel between them, and takes the larger one as the inlet. It then:

- meshes the water inside the channel with snappyHexMesh and solves its flow with OpenFOAM on two grids,
- carries the walls' temperature into the water to get each channel wall's heat transfer coefficient, with a
  thermal wall function for water's thin thermal sublayer,
- puts those coefficients on the channel faces of the plate, still air on the outside, and the 150 W on the bottom,
  and solves the plate's temperature with CalculiX on a refined mesh ladder.

```sh
./run.sh
```

The report in `output/` gives the hottest point, the pressure the pump must supply, the mean heat transfer
coefficient against Gnielinski's correlation for a straight tube, and every check. `output/playback.json`
holds the plate's surface with each face's heat transfer coefficient and the water's speed in the channel's middle plane for the site's 3D player.
