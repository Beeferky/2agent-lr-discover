#!/usr/bin/env python3
"""
Generate figures for README/paper from experimental data in `experiments/`.

Outputs to `docs/figures/`:
  1. cross_round_learning.png   — mean ppl progression across 4 phases
  2. head_to_head.png            — per-peak-LR comparison: cosine vs agent
  3. lr_schedule_evolution.png   — LR trajectory shape evolution across rounds
  4. exploration_landscape.png   — agent's (peak_lr, ppl) scatter vs cosine sweep curve

Usage:
  python scripts/plot_figures.py

Reads:
  experiments/final_results_log_agent_60m.jsonl
  experiments/decisions_log_agent_60m.jsonl
  experiments/cosine_baselines/*.json
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

AGENT_COLOR  = "#2C7BB6"   # blue
COSINE_COLOR = "#D7191C"   # red
NEUTRAL      = "#666666"
PHASE_SHADE  = ["#F7F7F7", "#EAEAEA", "#F7F7F7", "#EAEAEA"]


# ── Data loaders ───────────────────────────────────────────────────────
def load_agent_results():
    """Return list of dicts from final_results_log."""
    path = EXP / "final_results_log_agent_60m.jsonl"
    return [json.loads(l) for l in path.open() if l.strip()]


def load_agent_decisions():
    """Return {run_start_iso: [(step, new_lr, loss), ...]} sorted by step."""
    by_iso = defaultdict(list)
    for line in (EXP / "decisions_log_agent_60m.jsonl").open():
        if not line.strip():
            continue
        d = json.loads(line)
        by_iso[d["run_start_iso"]].append(
            (d.get("step", 0), d.get("new_lr", 0.0), d.get("loss", 0.0))
        )
    for iso in by_iso:
        by_iso[iso].sort(key=lambda t: t[0])
    return by_iso


def load_cosine_baselines():
    """Return list of (peak_lr, final_ppl) for cosine sweep."""
    out = []
    for f in sorted((EXP / "cosine_baselines").glob("*.json")):
        # filename like 'cosine_lr1.2e-3_60m_final.json'
        name = f.stem
        lr_str = name.replace("cosine_lr", "").replace("_60m_final", "")
        try:
            lr = float(lr_str)
        except ValueError:
            continue
        ppl = json.load(f.open())["final_eval_ppl"]
        out.append((lr, ppl))
    return sorted(out)


# ── Figure 1: cross-round learning ─────────────────────────────────────
def figure_cross_round_learning(rows):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    rounds = np.arange(1, len(rows) + 1)
    ppls = np.array([r["final_eval_ppl"] for r in rows])

    # Fix axis scale first so transAxes works properly
    ax.set_yscale("log")
    ax.set_xlim(0.5, 41.5)
    ax.set_ylim(22, 400)

    # Background phase shading
    phase_bounds = [(1, 10), (11, 20), (21, 30), (31, 40)]
    phase_names = ["Phase 1\nExplore", "Phase 2\nConsolidate",
                   "Phase 3\nRefine", "Phase 4\nBreakthrough"]
    for (lo, hi), color, name in zip(phase_bounds, PHASE_SHADE, phase_names):
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.6, zorder=0)
        mid = (lo + hi) / 2
        # Place phase label in top 5% of axes (transAxes y-coords are 0..1)
        ax.text(mid, 0.96, name,
                ha="center", va="top", fontsize=9.5, color="#555",
                transform=ax.get_xaxis_transform())

    # Plot ppl markers
    ax.plot(rounds, ppls, "o-", color=AGENT_COLOR, lw=1.5,
            markersize=6, label="Agent final ppl per round", zorder=3)

    # Phase mean horizontal segments (exclude R40 catastrophe from mean)
    for i, (lo, hi) in enumerate(phase_bounds):
        mask = (rounds >= lo) & (rounds <= hi)
        ppl_in = ppls[mask]
        ppl_for_mean = ppl_in[ppl_in < 200]
        m = ppl_for_mean.mean() if len(ppl_for_mean) > 0 else float("nan")
        ax.hlines(m, lo - 0.4, hi + 0.4,
                  color="#222", lw=2, linestyle="--",
                  label="Phase mean (excl. R40)" if i == 0 else None,
                  zorder=2)
        ax.text(hi + 0.3, m, f" {m:.1f}",
                va="center", ha="left", fontsize=9, color="#222")

    # Cosine baseline reference
    ax.axhline(30.37, color=COSINE_COLOR, lw=1.2, linestyle=":",
               label="Cosine baseline best (30.37)", zorder=1)

    # Annotate R39 breakthrough
    ax.annotate("R39 (peak 1e-2)\nppl 31.18 — breakthrough",
                xy=(39, ppls[38]), xytext=(32, 26),
                fontsize=9.5, color="#1A4F86", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1A4F86", lw=1.0))

    # Annotate R40 catastrophe
    ax.annotate(f"R40 (same config)\nppl {ppls[39]:.0f} — diverged",
                xy=(40, ppls[39]), xytext=(33, 250),
                fontsize=9.5, color="#A00000", ha="center",
                arrowprops=dict(arrowstyle="->", color="#A00000", lw=1.0))

    ax.set_xlabel("Round")
    ax.set_ylabel("Final eval perplexity (log scale)")
    ax.set_title("Cross-Round Learning: Agent Improves Across 40 Self-Discovery Iterations")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_yticks([25, 30, 40, 60, 100, 200, 300])
    ax.legend(loc="lower left", fontsize=9.5)

    fig.tight_layout()
    out = OUT / "01_cross_round_learning.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ── Figure 2: head-to-head bar chart ───────────────────────────────────
def figure_head_to_head(rows, cosine):
    """Cosine vs Agent best at matching peak LRs."""
    # Group agent rounds by peak_lr (rounded to 1 sig fig for grouping)
    agent_by_peak = defaultdict(list)
    for r in rows:
        peak = r["peak_lr_seen"]
        agent_by_peak[peak].append(r["final_eval_ppl"])

    # Peaks of interest: ones where cosine baseline exists
    cosine_dict = dict(cosine)
    peaks = sorted(cosine_dict.keys())

    # Find closest agent peak for each cosine peak (within factor 1.3)
    pairs = []
    for cp in peaks:
        # Find agent rounds with peak within ~1.3x
        candidates = [(ap, min(ppls)) for ap, ppls in agent_by_peak.items()
                      if 0.77 < ap / cp < 1.3]
        if candidates:
            agent_peak, agent_best = min(candidates, key=lambda t: t[1])
            pairs.append((cp, cosine_dict[cp], agent_best))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pairs))
    w = 0.38
    cosine_ppls = [p[1] for p in pairs]
    agent_ppls = [p[2] for p in pairs]
    def _lr_label(v):
        s = f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")
        return s
    labels = [_lr_label(p[0]) for p in pairs]

    bars_c = ax.bar(x - w/2, cosine_ppls, w, color=COSINE_COLOR, label="Cosine baseline", alpha=0.85)
    bars_a = ax.bar(x + w/2, agent_ppls,  w, color=AGENT_COLOR,  label="Agent best",     alpha=0.85)

    # Annotate values on top
    for bars in (bars_c, bars_a):
        for b in bars:
            h = b.get_height()
            label = f"{h:.0f}" if h > 100 else f"{h:.2f}"
            ax.text(b.get_x() + b.get_width()/2, h * 1.04, label,
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Peak LR")
    ax.set_ylabel("Final eval perplexity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_title("Head-to-Head: Same Peak LR, Cosine vs Agent Best")
    ax.legend(loc="upper left")

    # Highlight the 1e-2 column with a callout
    if any(p[0] >= 0.009 for p in pairs):
        ax.annotate("Agent wins where\ncosine diverges",
                    xy=(len(pairs) - 1, agent_ppls[-1]),
                    xytext=(len(pairs) - 2.2, 90),
                    fontsize=10, color="#222",
                    arrowprops=dict(arrowstyle="->", color="#222", lw=1.0))

    fig.tight_layout()
    out = OUT / "02_head_to_head.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ── Figure 3: LR schedule evolution across selected rounds ─────────────
def figure_lr_schedule_evolution(rows, decisions):
    """Plot LR(step) for 4 representative rounds showing how shape evolved."""
    # Pick representative rounds: early conservative, mid plateau, late refined, breakthrough
    picks = [
        (3, "R3 (P1: stuck at init 3e-4)", "#9DA3A8"),
        (10, "R10 (P1: found 1e-3)", "#7EA6C9"),
        (30, "R30 (P3: 1.2e-3 + high min)", "#3F7DB5"),
        (39, "R39 ⭐ breakthrough (peak 1e-2)", "#D7191C"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for round_idx, label, color in picks:
        if round_idx > len(rows):
            continue
        iso = rows[round_idx - 1]["run_start_iso"]
        traj = decisions.get(iso, [])
        if not traj:
            continue
        steps = [s for s, lr, _ in traj]
        lrs = [lr for s, lr, _ in traj]
        ax.plot(steps, lrs, "o-", color=color, lw=1.8, markersize=4, label=label, alpha=0.9)

    # Cosine baseline reference (peak 3e-3, the optimal) as dashed line
    base_steps = np.linspace(1000, 10000, 200)
    cos_lr = 3e-3 * 0.5 * (1 + np.cos(np.pi * (base_steps - 1000) / 9000))
    cos_lr = np.maximum(cos_lr, 3e-4)
    ax.plot(base_steps, cos_lr, "--", color="#999", lw=1.5,
            label="Cosine baseline (peak 3e-3)", zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR Schedule Shape Evolution Across Self-Discovery Iterations")
    ax.set_xlim(800, 10100)
    ax.set_ylim(5e-6, 2e-2)
    ax.legend(loc="upper right", fontsize=9.5)

    fig.tight_layout()
    out = OUT / "03_lr_schedule_evolution.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ── Figure 4: exploration landscape ─────────────────────────────────────
def figure_exploration_landscape(rows, cosine):
    """Scatter of (peak_lr, ppl) for agent rounds, overlaid on cosine sweep curve."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Cosine sweep line
    cos_peaks = np.array([c[0] for c in cosine])
    cos_ppls  = np.array([c[1] for c in cosine])
    ax.plot(cos_peaks, cos_ppls, "o-", color=COSINE_COLOR, lw=2,
            markersize=8, label="Cosine baseline sweep")

    # Agent scatter, colored by round number
    peaks = [r["peak_lr_seen"] for r in rows]
    ppls = [r["final_eval_ppl"] for r in rows]
    round_nums = list(range(1, len(rows) + 1))

    sc = ax.scatter(peaks, ppls, c=round_nums, cmap="viridis", s=70,
                    edgecolor="white", linewidth=0.8, alpha=0.9,
                    label="Agent rounds (color = round #)", zorder=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Round #", fontsize=10)

    # Annotate key rounds
    annotations = [
        (39, "R39 ⭐", (60, -25)),
        (40, "R40 💥", (-10, -45)),
        (28, "R28 (plateau)", (-50, -25)),
    ]
    for round_idx, label, offset in annotations:
        if round_idx <= len(rows):
            ax.annotate(label, xy=(peaks[round_idx - 1], ppls[round_idx - 1]),
                        xytext=offset, textcoords="offset points",
                        fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#222", lw=0.8))

    # Highlight the "dead zone" 3-5e-3 where agent never explored
    ax.axvspan(2.8e-3, 5e-3, color="#FFD580", alpha=0.25, zorder=1)
    ax.text(3.8e-3, 200, "agent\ndead zone\n(cosine's optimum)",
            ha="center", fontsize=9, color="#995500", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak LR (during round)")
    ax.set_ylabel("Final eval perplexity")
    ax.set_title("Exploration Landscape: Where Agent Sampled vs Cosine Sweep")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    out = OUT / "04_exploration_landscape.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    rows = load_agent_results()
    decisions = load_agent_decisions()
    cosine = load_cosine_baselines()
    print(f"  agent rounds:        {len(rows)}")
    print(f"  decision trajectories: {len(decisions)}")
    print(f"  cosine baselines:    {len(cosine)}")
    print()
    print("Rendering figures →", OUT)

    figure_cross_round_learning(rows)
    figure_head_to_head(rows, cosine)
    figure_lr_schedule_evolution(rows, decisions)
    figure_exploration_landscape(rows, cosine)

    print("\nDone.")


if __name__ == "__main__":
    main()
