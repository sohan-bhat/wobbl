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
| `--apply` | builds every fix marked auto and checks each one against the original |
| `--only` | a comma list of the kinds to apply, from `fillet`, `thicken`, `gusset` and `material`, where `thicken` works on parametric parts |

Without `--apply` it prints each fix with the number it needs and marks the ones it can apply
by itself. With it, Wobbl adds fillets and gussets to the part, changes only the thickness and
fillet parameters of a parametric part, and swaps the material. It prints a table of the peak
stress, safety factor and verdict before and after each fix, writes each solved version to
`fix-<kind>/` inside the result folder and updates the report with a before and after pair. A
material swap is scaled from the first solve when only the stiffness changes and solved again
otherwise, and a line under the table says which. A part that already meets its target gets
no fixes, and an unknown kind in `--only` stops the command before anything runs.
[Fixing parts](/guides/fixing/) covers what each kind does and what it can edit.
