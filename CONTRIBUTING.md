# Contributing to UniInfer

Thanks for your interest! UniInfer is a research codebase, so contributions are not only code. Ideas,
reproductions, negative results and critiques of our methodology are all valuable.

## Ways to contribute

| You want to… | Do this |
|---|---|
| Report a bug or a number you can't reproduce | Open a **Bug report** issue |
| Propose a research direction or experiment | Open a **Research idea** issue |
| Add a feature, operator, model or dataset | Open a **Feature request**, then a PR |
| Ask a question | GitHub Discussions |

For anything bigger than a small fix, **open an issue first** so we can agree on the approach.

## Development setup

```bash
git clone https://github.com/Rakshitha-Ireddi/UniInfer.git && cd UniInfer
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # CPU is enough for tests
pip install -e ".[dev,experiments]"
pytest -q && ruff check .
```

For GPU work use `scripts/setup_env.sh` (see README).

## Pull request checklist

- [ ] `pytest -q` and `ruff check .` pass (CI runs both).
- [ ] New code has unit tests that run on CPU.
- [ ] If you touched `runtime/`: `python experiments/oracle/sweep.py --check` prints `EQUIVALENCE PASS`
      (paste the output in the PR).
- [ ] If you touched attention capture or an attention-guided operator: `sweep.py --check-attn` prints
      `ATTENTION PASS`. If you touched `runtime/escalation.py`: `experiments/escalation/run.py --check` prints
      `ESCALATION PASS`.
- [ ] New results come with the script that produced them, a `*_meta.json`, and the realized cost.
- [ ] Significant design choices come with an ADR (`docs/adr/template.md`).
- [ ] Update `CHANGELOG.md` under *Unreleased*.

## Research standards

These are what make results here trustworthy. Please hold your PRs to them:

1. **Compare at equal realized cost.** Report the cost you actually spent, not the target.
2. **Choose everything on training data.** Configs, thresholds, λ and model selection must never see
   test requests.
3. **No credit for luck.** Upper bounds use luck-free (capped) scores (ADR-0006).
4. **Report negative results.** "This signal did not help" is a useful contribution.
5. **Pin everything.** Model revision, library versions, GPU and seed.

## Adding a compression operator

1. Add a pure function to `src/uniinfer/compression/` that takes tensors and returns a keep-mask or a new
   cache. Never mutate inputs. Export it from `uniinfer.compression`.
2. Add CPU tests: budget respected, protected tokens kept (sinks and the 16-token observation window), input
   unchanged, kept tokens in position order.
3. Wire it in behind a name:
   - **KV operator:** add a branch to `QwenVLRunner.compress()` and the name to `KV_METHODS`. If it needs
     attention scores, request query capture (see how `snapkv` and `pyramid` do it).
   - **Visual operator:** add the name to `METHODS` in `visual_tokens.py`; `keep_mask()` passes attention
     scores for attention-guided methods.
   - Add a descriptor flag in `config_descriptors` (`experiments/oracle/common.py`). Flags are added only
     when the method is present, so older results stay byte-identical.
4. Run it: `python experiments/oracle/sweep.py --kv-methods l2norm,<name>` (or `--v-methods`). Choose ratios
   with `--kv-ratios` / `--v-ratios`.
5. Cost is the KV length averaged over layers (`kv_tokens`). If layers or heads keep different counts, make
   sure the realized cost reflects that.
6. Compare with existing operators on the **same requests** with paired CIs (see
   `experiments/baselines/pyramid.py` for a minimal example).

## Code style

Python ≥ 3.10, type hints on public functions, docstrings that explain *why*, and ruff for lint and imports.
Keep library code (`src/`) free of experiment-specific logic.

By contributing you agree that your contributions are licensed under Apache-2.0 and that you will follow
the [Code of Conduct](CODE_OF_CONDUCT.md).
