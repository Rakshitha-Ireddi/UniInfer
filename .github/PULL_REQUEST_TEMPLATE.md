## What this PR does
<!-- One paragraph. Link the issue: Closes #… -->

## How it was verified
- [ ] `pytest -q` passes
- [ ] `ruff check .` passes
- [ ] (runtime changes) `python experiments/oracle/sweep.py --check` → `EQUIVALENCE PASS` (paste output)
- [ ] (new results) script + `*_meta.json` committed; realized cost reported

## Results (if any)
<!-- Table or figure, compared at equal realized cost against static and task-aware baselines. -->

## Checklist
- [ ] Tests for new code (CPU)
- [ ] `CHANGELOG.md` updated under *Unreleased*
- [ ] ADR added for significant design decisions
