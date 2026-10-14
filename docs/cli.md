# Command line reference

The full reference lives on the website, at
[sohan-bhat.github.io/wobbl/guides/cli](https://sohan-bhat.github.io/wobbl/guides/cli/). This page
keeps the shape of the command line and how new physics joins it.

## The commands

`wobbl simulate THING "question"` runs any physics on any thing. The thing is a file (a STEP, STL,
BREP, IGES or parametric Python part, a SPICE netlist or KiCad schematic, a mechanism spec, a URDF or
MJCF model, a rocket or motor spec), a folder (a finished result or an OpenFOAM case), a name, a
formula, SMILES or an equation, or a sentence that describes it. `wobbl part.step "question"` and
`wobbl "question"` do the same without the word simulate.

| Option | What it does |
|---|---|
| `--physics stress,heat` | runs these physics instead of the ones read from the question |
| `--domain part` and `--kind buck` | say what the thing is when Wobbl reads it wrongly |
| `-m`, `--load`, `--fix` | the material, loads and supports of a part, repeatable |
| `--NAME VALUE` or `--set NAME=VALUE` | any option a physics reads, such as `--power "12 W"` or `--rpm 6000` |
| `--plan` | prints the route, what was read from the question and every guess, without solving |
| `--list` | lists every domain, its physics and their options |
| `--estimate`, `--background`, `--remote NAME` | predicts the cost, or runs it as a job here or on a remote |
| `--quick`, `--json`, `--no-open`, `-o` | skips pictures and fixes, prints JSON, keeps the view closed, picks the folder |

Every run prints each guess with the option that overrides it and ends with the explicit command
that repeats it with nothing guessed, then opens the 3D view.

The general verbs work on any thing or result. `inspect` describes a part's faces, holes and walls,
`view` opens a result's 3D view, `check` and `report` read a finished run again, `fix` shows and
applies fixes, `uncertainty`, `credibility` and `test-plan` build credibility evidence, `estimate`
and `jobs` handle big runs, and `materials`, `validate`, `setup`, `doctor` and `mcp` look after the
install. The commands of earlier releases, such as `run`, `thermal`, `circuit`, `motion`,
`mechanism`, `flow`, `buckling` and `plastic`, still work as hidden aliases, and the MCP server keeps
its `solve_*` tools as deprecated aliases of `simulate`.

## Adding physics

A package registers its physics in a `domain.py` of its own, which Wobbl imports on first use, so
the new physics reaches `wobbl simulate`, `--list`, the MCP `simulate` tool and the routing test with
no new command or tool.

```python
from ..simulate import Analysis, Option, register
from ..structural.domain import PART_KINDS, material


def drop(job):
    height = job.value("height", "1 m", "drop height not given")    # given, or assumed with its override
    mat = material(job, "abs")
    job.situate(environment={"floor": "concrete"}, loads=[f"dropped from {height}"])   # for the 3D view
    job.ready()            # under --plan the run stops here, after every guess is recorded
    return run_drop(job.thing, mat, height, job.out, job.log)       # writes outcome.json in job.out


register(Analysis(
    "part", "drop", "Drop onto a floor: peak stress and whether it cracks", drop,
    kinds=PART_KINDS, dry=True, covers=("stress",),
    keywords=(r"(?<!pressure )\bdrop(?:ped|s)?\b",),
    options=(Option("height", "Drop height", "1 m"),)))
```

The router picks the analysis when a clause of the question matches its keywords, reads the loads of
that clause through `job.focus`, runs what it `needs` first and drops what it `covers` in the same
clause (or the whole sentence with `covers_scope="sentence"`, or the whole question with
`covers_all=True`). Each run writes `request.json` beside `outcome.json`, so
`wobbl.simulate.get("part", "drop").run(spec, out_dir, quick=True)` and
`wobbl.simulate.rerun(result_dir, out_dir, changes)` repeat it with nothing guessed, and
`job.criterion(...)` adds pass and fail lines that uncertainty studies read. A new kind of thing
registers a `Domain` with the file suffixes, words and recognizer that pick it.
