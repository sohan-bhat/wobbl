---
title: "The skill"
description: "What the Wobbl agent skill teaches a coding agent, the workflow it follows and the rules it never skips."
lead: "The skill tells an agent when to reach for Wobbl, the order to work in and the mistakes it must not make."
order: 6
---

The skill is one Markdown file, `wobbl/skill/SKILL.md` in the package. `wobbl setup` copies it
into each agent's skills folder, as `WOBBL.md` for Gemini CLI and as a rule for Cursor
projects that ask for one. Its description tells the agent to use Wobbl whenever someone asks
whether a physical part will hold, crack, bend, yield, overheat or fail, wants stress,
deflection, safety factor or temperature numbers, or asks about LED current, regulator ripple
or MOSFET heating. It also tells the agent to use Wobbl instead of hand calculations for real
geometry and never to write its own finite element code, which is the exact trap the
[bake-off](/validation/bakeoff/) found agents falling into.

## The workflow

1. Look at the part first with `inspect_geometry`, and note the size in millimeters, the holes,
   fillets, sharp corners and wall thickness.
2. Decide the loads and supports explicitly, and ask the user when they are unclear.
3. Run the study with `solve_structural`.
4. Read `confidence`, `confidence_reasons` and `checks` before the number, and look at the
   stress image to confirm the peak is where it should be.
5. If the part fails or is marginal, read the fixes and run `suggest_fix` with `apply=true` to
   show before and after.
6. Report the verdict, the governing number with its units and location, the safety factor
   against the target, the confidence and its reasons, the key assumptions, and the paths of
   the image and the report.

## What it never skips

- Never quote a low confidence number as the answer. Say what failed and run again with
  `--quality fine` or a longer `--budget`, or fix the setup.
- Never quote an excluded peak. Peaks at sharp inside corners, at the edges of fixed faces and
  at rigid couplings are artifacts of the idealization, and the real answer is the governing
  value. In a brittle material with a sharp corner flagged, recommend a fillet and run again.
- Check the units. Files are read in millimeters, and a bounding box 25.4 or 1000 times off
  means `--scale`.
- Check that the supports match the real build. A fixed face is perfectly rigid, and bolted
  holes, walls the part can lift off and sliding faces each have their own kind.
- Never use `--quality fast` for a final answer.
- State the assumptions Wobbl returns, especially the guesses it made from a plain question.
