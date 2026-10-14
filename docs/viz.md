# The 3D view of every run

Every Wobbl run ends with an interactive 3D simulation of what happened, set in the user's situation. The run
writes `scene.wobbl` next to its `outcome.json`, and the viewer plays it inside `report.html`, offline, as one
file. `wobbl view <result folder>` opens it, `wobbl simulate` opens `simulation.html` on its own, and every MCP
result carries its path as `view`.

## For domain agents: write your scene directly

Build the scene where your run has its data in memory, and save it before the report is written. Every report
writer then embeds it.

```python
from wobbl.viz import Scene
from wobbl.viz import props as P

s = Scene("Bracket under 40 N", domain="strength", question=question, length="mm")
s.timeline(load_n, unit="N", label="load", seconds=6, between="exact")       # one value per frame
part = s.mesh(X, tris, keep=[peak_node])               # 3 or 6 node triangles; decimated, real vertices only
part.color_by("von Mises", vm, unit="MPa", scale=ramp, range=(0, allow), limit=allow, limit_label="yield")
part.deform(U, scale=ramp, exaggerate=40)              # the view says it is drawn 40 times larger
P.slab(s, wall_face_center, toward_part, 200, 160, 12, name="wall")         # props sit at the real supports
P.arrow(s, load_point, direction, 30, ramp, "{load}")  # braces fill in a readout's live value
s.readout("load", load_n, unit="N")
s.readout("peak stress", vm.max() * ramp, unit="MPa", limit=allow)          # red above the limit
s.event(36.8, "first yield", "at the inside fillet", kind="fail")
s.label("yields here", peak_xyz, t0=36.8, kind="fail")
s.verdict("FAILS", "It cracks on the inside fillet at about 92% of the 40 N load.", tone="fail")
s.save(out_dir)                                        # out_dir/scene.wobbl, under about 1 MB
```

The API holds these pieces.

- `timeline(t, unit, label, seconds, between)` takes the saved frames' times, loads or cycles. `between` is
  `interpolated` (the viewer blends and labels it), `exact` (in between is exact, as for a linear model scaled
  by load) or `step` (each frame held).
- `mesh(X, tris)` with `.color_by(field, values, scale=None)`, `.deform(U, scale=None, exaggerate=1)`,
  `.move(pos, quat)` for a rigid pose per frame and `.reveal(values, thresholds)` for a print rising or a mold
  filling. Values are per vertex, either for every frame (frames x vertices) or once with a scale per frame.
- `body(shape, size, pos, quat)` for rigid primitives (sphere, box, cylinder, capsule, cone, plane, hex,
  ellipsoid, torus; sizes as in MuJoCo), with `.light(level)` for a glow per frame and `.color_by(field, values)`
  for one value per frame.
- `atoms(elements, positions, bonds=None, cell=None)` for molecules and crystals, positions fixed or per frame.
- `points(positions, values=None)` for particles, `lines(paths, drawn=, at=, fade=, flow=)` for trajectories that
  trace themselves, crack fronts that appear by time, field lines, and wires with current moving along them.
- `arrows(origins, vectors, scale=None, labels=None)`, `ground(z, size, grid)`, `label(text, at, t0, t1)`,
  `readout(name, values, unit, limit)`, `event(t, name, detail, kind)`, `note(text)`, `verdict(word, line, tone)`
  and `camera(target, direction, distance, follow=, presets=)`.

Every per frame array has the timeline's frame count first. The API thins frames and decimates meshes until the
gzipped file fits its budget, so a scene cannot grow past about 1 MB. Props (anything made with `prop=True`, and
everything in `wobbl.viz.props`) draw with faint stripes and are listed as illustrative. They never take part in
the analysis, and they belong where the run's own supports and loads are.

Truthfulness rules the viewer enforces and your scene should keep: every frame comes from the run's own data,
blended frames are labeled, exaggerated motion says its scale, and anything drawn for the eye says so in a note.

## Runs that do not write a scene yet

`wobbl.viz.adapters` builds one from what the run already writes: mechanism, drop, vibration, buckling, fatigue
and plastic `playback.json`, the chemistry playbacks (solid motor, flame, cell runaway, nozzle), crack growth,
casting and molding, field slices, flow grids, `fields.npz` of heat runs, `trajectory.csv` of motion runs and
`board.json` of circuit runs. A domain that writes its scene directly needs none of this.

Some domains still need to write a little more for a fully truthful scene.

- Strength shells (`analyze_shell`) write no scene yet. The shell mesh, the stress per node and the
  displacement would do, through `Scene.mesh` as the solid path does in `viz/adapters/strength.py`.
- Vibration, buckling and fatigue playbacks carry no supports or loads, so their scenes have no wall or bolts.
  Adding the fixture faces' triangles (or the strength run's selections) to playback.json would place them.
- A field slice's parts are flat shapes. Their thickness or revolve axis in the playback would let the view
  extrude or revolve them.
- A flow playback without frames is steady, so particles move through the one field. Unsteady runs should keep
  writing frames, which the flow adapter can then follow.

## For the site: show any run's scene on a page

The viewer is one dependency free script of about 50 kB (39 kB minified, 16 kB gzipped). Build it with
`python scripts/export_site_viewer.py`, which writes `site/public/viewer/wobbl-view.js` and copies each example's
`scene.wobbl` to `site/public/data/scenes/<example>.wobbl`. Then a page needs only this.

```html
<script src="/wobbl/viewer/wobbl-view.js" defer></script>
<div data-wobbl-scene="/wobbl/data/scenes/shelf-bracket.wobbl"></div>
```

or from a module, `WobblView.mount(element, { src, autoplay })`, which resolves to `{ play, pause, seek,
dispose, frames }`. The element gets its own shadow root, so the site's styles do not reach inside it, and it
reads the site's Archivo and IBM Plex Mono when they are loaded. It plays only while on screen, starts on the
result without playing under reduced motion, and keeps its readouts and timeline when WebGL is missing.
