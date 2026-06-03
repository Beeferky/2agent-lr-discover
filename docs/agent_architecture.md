# LLM-Driven LR Discovery — 2 Agents + 1 Rule Set

Single-process orchestrator (`agent.py`) runs one round and talks to the training
subprocess (`main.py`, torchrun) via `training_state.json` (read) and `lr_command.txt`
(write). Only **Memory** persists across rounds (ckpts wiped each round). No LR schedule
is fed to the LLMs — the system self-discovers it (D2Z-like) from its own trial outcomes.

```
╔════════════════════════════════════════════════════════════════════╗
║   LLM-DRIVEN LR DISCOVERY — 2 AGENTS + 1 RULE SET (single process)   ║
╚════════════════════════════════════════════════════════════════════╝

  FIXED CONFIG (not agents):
    warmup = 1000 steps  |  K = 300 steps (ablate {100,300,500})
    init LR = 3e-4       |  LR jump constraint = [0.5x, 2x] of current

  ─────────────────────────  ROUND START  ──────────────────────────
   ┌────────────────────────┐  strategy_brief (retrieved experience)
   │ [Agent] MEMORY (STRONG)│ ───────────────────────────────┐
   │ cross-round strategy   │   "what past rounds tried &      │
   │ retriever; reads logs, │    what ppl they got"            │
   │ distills (NO outside   │                                  │
   │  answer)               │                                  ▼
   └────────────────────────┘                      (injected into Controller prompt)

  ─────────────────  TRAINING LOOP (every K=300 steps)  ─────────────
        training_state.json (step, loss)
                  │
                  ▼
        ┌────────────────────────┐  proposed LR   ┌──────────────────────────┐
        │ [Agent] CONTROLLER FAST│ ─────────────▶ │ [Rules] SafetyGuard       │
        │ decides next LR from   │                │  • LR jump clamp [0.5x,2x]│
        │ current loss + Memory  │                │    (pure code, no LLM)    │
        │ brief                  │                └────────────┬─────────────┘
        └────────────────────────┘                 final LR    │
                                                                ▼
                                                        lr_command.txt
                                                                │ main.py reads every
                                                                ▼ step, applies to all GPUs

        ┌──────────────────────────────────────────────────────────────┐
        │ [Rules] SafetyGuard — three rules, pure code, no LLM:          │
        │   ① LR-jump clamp: proposed LR ∈ [0.5x, 2x] of current         │
        │   ② Crash rollback: subprocess exits non-zero →                │
        │      pick healthy ckpt (>=ROLLBACK_BACK_STEPS before crash) +  │
        │      restart LR ×0.5. ckpts saved every K steps within round.  │
        │   ③ NaN / Inf / loss > EMERGENCY_LOSS_MAX during training →    │
        │      terminate subprocess → falls into rule ② automatically.   │
        │      (small loss spikes are recorded only, not catastrophic)   │
        └──────────────────────────────────────────────────────────────┘

  ──────────────────────────  ROUND END  ───────────────────────────
        write final_results_log (warmup/K/init_lr/ppl) + decisions_log
                          │
                          └── next round's MEMORY reads it → self-improve
        (ckpts deleted at next round via clean_run; only Memory persists)
```

## Components

| Name | Kind | Model | When | Job |
|------|------|-------|------|-----|
| **Controller** | Agent (LLM) | FAST | every K steps | decide next LR from current loss + Memory brief |
| **Memory** | Agent (LLM) | STRONG | round start | cross-round strategy retriever (distills past logs, no external answer) |
| **SafetyGuard** | Rules (no LLM) | — | every decision / on crash / on NaN | (1) LR-jump clamp [0.5x,2x]; (2) crash rollback to healthy ckpt + LR×0.5; (3) NaN/Inf/catastrophic loss → terminate subprocess → triggers rule (2) |

Fixed config (in `agent.py`): `LR_INIT=3e-4`, `WARMUP_STEPS=1000`, `K=DECISION_EVERY=300`
(override via env `K=100/500` for ablation), `LR_JUMP_LOW/HIGH=0.5/2.0`.
Agent models in `AGENT_MODELS` (`MODEL_STRONG`/`MODEL_FAST`, default `qwen3.6-plus-2026-04-02`).
