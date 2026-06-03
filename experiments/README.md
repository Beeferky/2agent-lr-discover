# Experimental Data

Reproducibility data for the 2-Agent LR Discovery framework, captured across 40 agent rounds + 8 cosine baseline runs on LLaMA-60M / C4 / 10000 steps.

## Files

### Agent (40 rounds)

| File | Description |
|------|-------------|
| `final_results_log_agent_60m.jsonl` | One line per completed round: `{round_id, final_eval_ppl, peak_lr_seen, min_lr_seen, total_decisions, crash_count, ...}` (40 entries) |
| `decisions_log_agent_60m.jsonl` | One line per Controller LR decision: `{run_start_iso, step, loss, old_lr, new_lr, trend, reasoning}` (~1230 entries) |
| `rollback_log_agent_60m.jsonl` | One line per SafetyGuard crash rollback: `{run_start_iso, crash_step, lr_at_crash, rolled_back_to_step, new_lr, crash_reason}` (9 entries) |

The three jsonl files are joined by `run_start_iso` to reconstruct each round's full trajectory.

### Cosine Baseline Sweep (8 peak LRs)

Each `cosine_lrXe-Y_60m_final.json` is the output of one cosine-decay training with that peak LR:

```json
{
  "final_eval_loss": 3.4191,
  "final_eval_ppl":  30.5415
}
```

Sweep results:

| peak LR | final ppl | Notes |
|---|---|---|
| 1e-3 | 33.13 | Below sweet spot |
| **1.2e-3** | 32.28 | Near agent's converged region |
| **2.5e-3** | **30.54** | Same peak as agent R28 (32.85) — shows shape matters |
| **3e-3** | **30.37** | Sweet spot |
| 3.5e-3 | 30.41 | Near sweet spot |
| 4.5e-3 | 30.37 | Tied best |
| 5e-3 | 30.44 | Near sweet spot |
| **1e-2** | **215.08** | **Diverged (training unstable)** |

## Headline Comparison

Agent achieved its best result (R39: ppl 31.18, peak 1e-2) **in the region where cosine diverges**. See [`docs/日志2_40轮后的发现.md`](../docs/日志2_40轮后的发现.md) for the full analysis including attribution-failure case study (R28).

## Quick Inspection

```bash
# Peek at final results for all 40 agent rounds
cat experiments/final_results_log_agent_60m.jsonl | \
  jq -c '{round: input_line_number, ppl: .final_eval_ppl, peak: .peak_lr_seen}'

# Count rounds with crashes
cat experiments/final_results_log_agent_60m.jsonl | jq '.crash_count' | sort | uniq -c
```
