---
title: "wobbl check"
description: "Show the checks, the confidence level and its reasons for a finished Wobbl run."
lead: "check prints the checks and the confidence of a finished run again, without solving anything."
order: 6
---

```sh
wobbl check [RESULT_DIR] [--json]
```

With no folder it picks the newest result under `wobbl-out/` in the current directory.

```sh
wobbl check wobbl-out/bracket
wobbl check --json
```

It prints the part name and the confidence level, a table with one line per check and its
status, and the reasons behind the level. `--json` prints the confidence, the reasons and every
check with its details instead, which is handy in a script that should stop on low confidence.
[Checks](/guides/results/checks/) explains what each one tests.
