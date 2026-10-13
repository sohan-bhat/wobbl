---
title: "wobbl validate"
description: "Run Wobbl's 19 validation cases with known answers on your own machine, all of them or a chosen few."
lead: "validate runs the 19 cases with known answers through the same pipeline your studies use."
order: 10
---

```sh
wobbl validate                       # all 19 cases, about 30 minutes
wobbl validate --only lame,led       # a few of them
```

The full run took 29 minutes on a 2018 six core laptop. Each case builds its part, meshes it,
refines until the answer settles and runs its checks, and the number is then compared with a
textbook formula, a NAFEMS benchmark or a converged reference model. The command prints how
many cases landed within their tolerance and exits with an error when any did not, so it works
in a CI job.

## Options

| Option | What it does |
|---|---|
| `--only` | a comma list of case ids |
| `-o`, `--out` | the folder for the case results, `wobbl-out/validation` by default |
| `--markdown` | also writes the results table to this Markdown file |

The output folder gets one subfolder per case, a `results.json` with every number and a
`results.md` table.

## Case ids

| Group | Ids |
|---|---|
| Textbook formulas | `cantilever-tip`, `cantilever-stress`, `plate-hole`, `lame`, `torsion`, `self-weight` |
| Industry benchmarks | `nafems-le1`, `nafems-le10` |
| Real parts | `cast-motor-mount`, `enclosure-shell`, `drone-arm-truss`, `bracket-sharp`, `bracket-fillet` |
| Heat | `slab`, `lumped`, `sudden-face` |
| Circuits | `led`, `rc`, `buck` |

[Validation](/validation/) shows the published results, and
[How the cases are set up](/validation/cases/) describes each one.
