---
title: "MCP tools"
description: "All 13 tools of the Wobbl MCP server, their parameters and the compact JSON each one returns."
lead: "The MCP server gives an agent 13 tools, and every one returns compact JSON with numbers, statuses, one line summaries and file paths."
order: 5
wide: true
---

Agents start the server with `wobbl mcp`, which `wobbl setup` registers for them. Results stay
small enough to read in one go, and when an agent needs to see the part it opens the PNG
heatmaps or the HTML report from the paths it gets back. Studies run from the tools write
their folders to `~/.wobbl/results/`. When something goes wrong a tool returns an `error`,
often with a `hint` that says what to change, instead of failing the call.

## The tools

| Tool | Parameters | Returns |
|---|---|---|
| `inspect_geometry` | `path`, `scale` | bounding box, volume, wall thickness, holes, fillets, sharp inside corners, up to 40 of the largest faces with ids, notes and example selectors that work on this part |
| `mesh` | `path`, `size` | node and element counts, the element size and the mesh time, to check that a file meshes at all |
| `new_study` | `path`, `material`, `question` | a `study_id` for building a study one step at a time |
| `set_material` | `study_id`, `material` | the material's properties as Wobbl will use them |
| `apply_load` | `study_id`, `load` | how the load was read and every load so far |
| `add_fixture` | `study_id`, `fixture` | how the support was read and every support so far |
| `solve_structural` | `study_id`, or `path` with `material`, `loads` and `fixtures`, plus `target_sf`, `quality`, `budget`, `question` and `print_direction` | the verified strength result |
| `verify` | `result_dir` | the confidence and its reasons, every check, the assumptions and the peaks that were excluded |
| `render` | `result_dir` | the paths of the heatmaps and the report |
| `suggest_fix` | `result_dir`, `apply`, `which` | the fixes with numbers, or for a strength result with `apply=true` the before and after of each one, while heat and circuit fixes come back with a note to solve again with the new value |
| `solve_thermal` | `path`, `material`, `power`, `heat_faces`, `film`, `ambient`, `fixed_temp`, `duration`, `limit`, `radiation` | the steady or transient heat result |
| `solve_circuit` | `template` or `netlist`, and `params` | the circuit result |
| `solve_motion` | `spec`, or `diameter` with `mass` or `density`, plus `cd`, `height`, `speed`, `angle`, `wind`, `name` and `question` | the flight with its events, checks, fixes and trajectory |
| `list_materials` | none | every material with its stiffness, limit, criterion, density, conductivity and service temperature |

Loads, supports, films and selectors use exactly the syntax of the command line, so the
[inputs guides](/guides/inputs/) apply word for word.

## solve_structural

This is the tool agents use most. The quickest way to call it is with everything at once.

```text Tool call
solve_structural(path="bracket.py", material="pla",
                 fixtures=["bolt holes d=4.5", "wall x=min"],
                 loads=["40 N down @ top and within 12 of 65,12.5,6"])
```

It can also take a `study_id` built with `new_study`, `set_material`, `apply_load` and
`add_fixture`, which saves each step under `~/.wobbl/studies/` so an agent can build a study
over several turns. When the material, the loads or the supports are missing, Wobbl reads them
from `question` the same way it reads a plain question on the command line and lists each
guess in the assumptions.

The result holds `verdict`, `headline`, `governing_stress_MPa`, `where`, `location_mm`,
`measure`, `safety_factor`, `target_safety_factor`, `max_deflection_mm`, `confidence`,
`confidence_reasons`, `checks`, `assumptions`, `fixes`, `images`, `report`, `result_dir`,
`mesh_levels` and `time_s`. The `result_dir` is what `verify`, `render` and `suggest_fix` take.

## solve_thermal and solve_circuit

`solve_thermal` takes one `film` and one `fixed_temp` as text, like `"5 W/m2K @ all"` and
`"40 C @ bottom"`, and a `duration` makes it transient. It returns the verdict, the headline,
`max_temperature_C`, `hottest_point_mm`, `limit_C`, `time_over_limit_s` for a transient that
crosses its limit, and the usual confidence, checks, assumptions, fixes, images and report.

`solve_circuit` takes a template name, `led`, `rc`, `buck` or `mosfet`, with its parameters in
`params`, as in `params={"supply": "5V", "resistor": "47", "led": "blue"}`. For your own
circuit, pass the netlist as a file path or as the text itself, and list the signals to read in
`params.vectors`. It returns the verdict, the headline, the main figures in `metrics`, the raw
`values` and the usual checks and files.

## solve_motion

`solve_motion` flies a round body from its quantities, as in
`solve_motion(name="rock", diameter="9 cm", density="2700 kg/m3", cd=0.6, height="300 m")`, or a
rocket from `spec`, a TOML file path or the TOML text itself, with the tables that
[Motion](/guides/simulations/motion/) describes. With a spec only `wind` can change, since it is
the one thing that differs from day to day. It returns the verdict, the headline, the main figures
in `metrics`, the raw `values`, the `events` from liftoff or release to the ground, the usual
confidence, checks, assumptions, fixes, images and report, and the path of `trajectory.csv`.

## The server's own instructions

The server tells every agent the same flow when it connects. Inspect the part first, solve
with explicit loads and supports, read `confidence`, `confidence_reasons` and `checks` before
quoting any number, never quote a low confidence result as final or a peak listed as an
excluded artifact, open the stress image to see where the peak is, and call `suggest_fix` with
`apply=true` when the verdict is fails or marginal. [The skill](/guides/agents/skill/) teaches
the same thing in more depth.
