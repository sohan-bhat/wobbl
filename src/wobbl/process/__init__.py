"""Manufacturing: can the part be made the way it is drawn?

Strength and heat check a finished part. These studies check the process that
makes it: a 3D print warping off the bed, a casting freezing with a shrinkage
pocket inside, an injection molded part filling short, a sheet metal bend
springing back. Each one runs a real solver (CalculiX for the print, Wobbl's
own enthalpy and Hele Shaw solvers on gmsh meshes for casting and molding),
checks itself against closed form solutions and reports a verdict, a
confidence level and fixes, in the same outcome.json as every other domain.

Process simulation is an estimate. Every result says which inputs it is most
sensitive to, and none of them is reported with high confidence when it rests
on calibrated or uncertain material data.
"""
