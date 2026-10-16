# Studio light

`studio.rgbe` is the image based light the example pages' 3D views are lit with (src/scripts/play-look.js).

- Source: Studio Small 08 by Sergej Majboroda, Poly Haven, https://polyhaven.com/a/studio_small_08
- License: CC0 1.0 (public domain), https://polyhaven.com/license
- Made from the 1k Radiance HDR (`studio_small_08_1k.hdr`), box filtered to 512 by 256 and stored as RGBE in four
  planes (red, green, blue, exponent), each row as the change from the pixel before, gzipped. The first four
  bytes are the width and height as little endian 16 bit numbers. About 200 kB.
