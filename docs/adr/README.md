# Architecture Decision Records

Each ADR records one significant decision: the context, what we chose, the alternatives, and what it costs us.
ADRs are never deleted. A reversed decision gets a new ADR that marks the old one **Superseded**.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-narrow-research-scope.md) | Start with request-adaptive visual-token and KV budgets; defer weight quantization | Accepted |
| [0002](0002-backbone-and-benchmarks.md) | Qwen2.5-VL-3B on TextVQA, DocVQA and MMBench, in HF Transformers | Accepted |
| [0003](0003-custom-prefill-decode-loop.md) | Own the prefill/decode loop instead of `generate()` | Accepted |
| [0004](0004-compression-operators.md) | Initial compression operators: redundancy pruning and L2-norm KV eviction | Accepted |
| [0005](0005-cost-model-and-budgeted-selection.md) | Cost = relative KV memory; average budget; Lagrangian selection | Accepted |
| [0006](0006-evaluation-protocol.md) | Evaluation protocol: oracle first, request-grouped CV, luck-free claims | Accepted |
| [0007](0007-repository-and-reproducibility.md) | Repository layout, committed results, environment | Accepted |
| [0008](0008-confidence-cascade.md) | Confidence cascades, with failed attempts paid for | Accepted |
| [0009](0009-attention-guided-baselines.md) | Attention-guided baselines (SnapKV, FastV-style) via attention reconstruction | Accepted |

New ADRs: copy [`template.md`](template.md), take the next number, and open a PR.
