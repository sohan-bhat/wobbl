---
title: "wobbl fix"
description: "List the fixes for a failing part, or apply them with --apply and solve each edited version again."
lead: "fix lists the suggested changes for a finished strength run, and with --apply builds them and solves each one again."
order: 8
---

```sh
wobbl fix [RESULT_DIR] [--apply] [--only KINDS]
```

With no folder it uses the newest result under `wobbl-out/`.

```sh
wobbl fix wobbl-out/bracket                            # list the fixes
wobbl fix wobbl-out/bracket --apply                    # build each one and run the study again
wobbl fix wobbl-out/bracket --apply --only fillet,gusset
```

## Options

| Option | What it does |
|---|---|
| `--apply` | edits the geometry or the material for every fix marked auto and runs the study again for each |
| `--only` | a comma list of the kinds to apply, from `fillet`, `thicken`, `gusset` and `material` |

Without `--apply` it prints each fix with the number it needs and marks the ones it can apply
by itself. With it, Wobbl prints a table of the peak stress, safety factor and verdict before
and after each fix, writes each version to `fix-<kind>/` inside the result folder and updates
the report with a before and after pair. A part that already meets its target gets no fixes.
[Fixing parts](/guides/fixing/) covers what each kind does and what it can edit.
