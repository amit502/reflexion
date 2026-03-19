"""
Run this after all 4 experiments complete to generate comparison plots.

Usage:
    python compare_results.py --root_dir root
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import read_jsonl
from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="root")
    parser.add_argument("--language", type=str, default="py")
    return parser.parse_args()


def load_all_metrics(root_dir: str) -> Dict[str, dict]:
    """Load metrics from all strategy runs."""
    metrics = {}
    strategy_dirs = {
        "Simple":              "exp1_simple",
        "CoT+GT":              "exp2_cot_gt",
        "Reflexion":           "exp3_reflexion",
        "Retrieval Reflexion": "exp4_retrieval_reflexion",
    }
    for label, run_name in strategy_dirs.items():
        metrics_path = os.path.join(root_dir, run_name, f"{run_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics[label] = json.load(f)
            print(f"Loaded metrics for {label}")
        else:
            print(f"No metrics found for {label} at {metrics_path}")
    return metrics


def save_comparison_plots(metrics: Dict[str, dict], root_dir: str) -> None:
    if not metrics:
        print("No metrics to plot.")
        return

    strategies = list(metrics.keys())
    colors     = ['steelblue', 'mediumseagreen', 'tomato', 'darkorange']

    pass_at_1   = [metrics[s].get("pass_at_1",       0) for s in strategies]
    avg_iters   = [metrics[s].get("avg_iters",        0) for s in strategies]
    avg_refs    = [metrics[s].get("avg_reflections",  0) for s in strategies]
    solved      = [metrics[s].get("solved",            0) for s in strategies]
    failed      = [metrics[s].get("failed",            0) for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Programming Task Results — Strategy Comparison', fontsize=14)

    # ── 1. Pass@1 ────────────────────────────────────────────────────────────
    bars = axes[0].bar(strategies, pass_at_1, color=colors)
    axes[0].set_title('Pass@1')
    axes[0].set_ylabel('Pass@1 Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=20)
    for bar, v in zip(bars, pass_at_1):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                     f'{v:.2%}', ha='center', fontsize=9)

    # ── 2. Solved vs Failed ──────────────────────────────────────────────────
    x = np.arange(len(strategies))
    width = 0.35
    axes[1].bar(x - width/2, solved, width, label='Solved', color='mediumseagreen')
    axes[1].bar(x + width/2, failed, width, label='Failed', color='tomato')
    axes[1].set_title('Solved vs Failed')
    axes[1].set_ylabel('Number of Problems')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=20)
    axes[1].legend()

    # ── 3. Avg iterations ────────────────────────────────────────────────────
    bars = axes[2].bar(strategies, avg_iters, color=colors)
    axes[2].set_title('Avg Iterations per Problem')
    axes[2].set_ylabel('Avg Iterations')
    axes[2].tick_params(axis='x', rotation=20)
    for bar, v in zip(bars, avg_iters):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.05,
                     f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(root_dir, "strategy_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_path}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Strategy':<25} {'Pass@1':>8} {'Solved':>8} {'Failed':>8} {'AvgIter':>8}")
    print("="*60)
    for s in strategies:
        m = metrics[s]
        print(f"{s:<25} {m.get('pass_at_1',0):>8.2%} "
              f"{m.get('solved',0):>8} "
              f"{m.get('failed',0):>8} "
              f"{m.get('avg_iters',0):>8.1f}")
    print("="*60)


if __name__ == "__main__":
    args = get_args()
    metrics = load_all_metrics(args.root_dir)
    save_comparison_plots(metrics, args.root_dir)