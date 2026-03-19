# import os
# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt

# from immediate_refinement import run_immediate_refinement
# from immediate_reflexion import run_immediate_reflexion
# from simple import run_simple
# from reflexion import run_reflexion
# from cot_gt import run_cot_gt
# from retrieval_reflexion import run_retrieval_reflexion
# # from reflexion_ucs import run_reflexion_ucs
# from test_acc import run_test_acc
# from programming_agents import TrajectoryStore
# from utils import read_jsonl, read_jsonl_gz

# from typing import List


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run_name", type=str, help="The name of the run")
#     parser.add_argument("--root_dir", type=str,
#                         help="The root logging directory", default="root")
#     parser.add_argument("--dataset_path", type=str,
#                         help="The path to the benchmark dataset", default="root")
#     parser.add_argument("--strategy", type=str,
#                         help="Strategy: `simple`, `cot_gt`, `reflexion`, `retrieval_reflexion`")
#     parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
#     parser.add_argument("--model", type=str,
#                         help="OpenAI models only for now. For best results, use GPT-4")
#     parser.add_argument("--pass_at_k", type=int,
#                         help="Pass@k metric", default=1)
#     parser.add_argument("--max_iters", type=int,
#                         help="The maximum number of self-improvement iterations", default=10)
#     parser.add_argument("--expansion_factor", type=int,
#                         help="The expansion factor for the reflexion UCS and A* strategy", default=3)
#     parser.add_argument("--is_leetcode", action='store_true',
#                         help="To run the leetcode benchmark")
#     parser.add_argument("--verbose", action='store_true',
#                         help="To print live logs")
#     args = parser.parse_args()
#     return args


# def strategy_factory(strategy: str):
#     def kwargs_wrapper_gen(func, delete_keys=[]):
#         def kwargs_wrapper(**kwargs):
#             for key in delete_keys:
#                 del kwargs[key]
#             return func(**kwargs)
#         return kwargs_wrapper

#     if strategy == "simple":
#         return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
#     elif strategy == "cot_gt":
#         return kwargs_wrapper_gen(run_cot_gt, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
#     elif strategy == "reflexion":
#         return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor", "trajectory_store"])
#     elif strategy == "retrieval_reflexion":
#         return kwargs_wrapper_gen(run_retrieval_reflexion, delete_keys=["expansion_factor"])
#     elif strategy == "immediate-reflexion":
#         return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor", "trajectory_store"])
#     elif strategy == "immediate-refinement":
#         return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor", "trajectory_store"])
#     # elif strategy == "reflexion-ucs":
#     #     return kwargs_wrapper_gen(run_reflexion_ucs, delete_keys=["trajectory_store"])
#     elif strategy == "test-acc":
#         return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
#     else:
#         raise ValueError(f"Strategy `{strategy}` is not supported")


# # ---------------------------------------------------------------------------
# # Metrics + Plotting
# # ---------------------------------------------------------------------------

# def compute_metrics(log_path: str) -> dict:
#     """Parse the jsonl log to compute metrics."""
#     if not os.path.exists(log_path):
#         return {}

#     items = read_jsonl(log_path)
#     if not items:
#         return {}

#     num_total   = len(items)
#     num_solved  = sum(1 for item in items if item.get("is_solved", False))
#     num_failed  = num_total - num_solved

#     # avg number of implementations tried per problem
#     avg_iters = float(np.mean([
#         len(item.get("implementations", [item.get("solution", "")]))
#         for item in items
#     ]))

#     # avg number of reflections per failed problem
#     failed_items = [item for item in items if not item.get("is_solved", False)]
#     avg_reflections = float(np.mean([
#         len(item.get("reflections", []))
#         for item in failed_items
#     ])) if failed_items else 0.0

#     return {
#         "total":           num_total,
#         "solved":          num_solved,
#         "failed":          num_failed,
#         "pass_at_1":       round(num_solved / num_total, 4) if num_total > 0 else 0.0,
#         "avg_iters":       round(avg_iters, 2),
#         "avg_reflections": round(avg_reflections, 2),
#     }


# def save_metrics_log(metrics: dict, log_dir: str, run_name: str) -> None:
#     metrics_path = os.path.join(log_dir, f"{run_name}_metrics.json")
#     with open(metrics_path, 'w') as f:
#         json.dump(metrics, f, indent=4)
#     print(f"Metrics saved to {metrics_path}")


# def save_comparison_plot(all_metrics: dict, log_dir: str) -> None:
#     """
#     Save comparison plots across strategies.
#     Call this after running all 4 strategies to compare.
#     """
#     if not all_metrics:
#         return

#     strategies  = list(all_metrics.keys())
#     pass_at_1   = [all_metrics[s].get("pass_at_1",       0) for s in strategies]
#     avg_iters   = [all_metrics[s].get("avg_iters",        0) for s in strategies]
#     avg_refs    = [all_metrics[s].get("avg_reflections",  0) for s in strategies]

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # ── 1. Pass@1 ────────────────────────────────────────────────────────────
#     axes[0].bar(strategies, pass_at_1, color=['steelblue', 'mediumseagreen', 'tomato', 'darkorange'])
#     axes[0].set_title('Pass@1 by Strategy')
#     axes[0].set_ylabel('Pass@1')
#     axes[0].set_ylim(0, 1)
#     axes[0].tick_params(axis='x', rotation=15)
#     for j, v in enumerate(pass_at_1):
#         axes[0].text(j, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

#     # ── 2. Avg iterations ────────────────────────────────────────────────────
#     axes[1].bar(strategies, avg_iters, color=['steelblue', 'mediumseagreen', 'tomato', 'darkorange'])
#     axes[1].set_title('Avg Iterations per Problem')
#     axes[1].set_ylabel('Avg Iterations')
#     axes[1].tick_params(axis='x', rotation=15)

#     # ── 3. Avg reflections on failed problems ────────────────────────────────
#     axes[2].bar(strategies, avg_refs, color=['steelblue', 'mediumseagreen', 'tomato', 'darkorange'])
#     axes[2].set_title('Avg Reflections (Failed Problems)')
#     axes[2].set_ylabel('Avg Reflections')
#     axes[2].tick_params(axis='x', rotation=15)

#     plt.tight_layout()
#     plot_path = os.path.join(log_dir, "strategy_comparison.png")
#     plt.savefig(plot_path, dpi=150)
#     plt.close()
#     print(f"Comparison plot saved to {plot_path}")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main(args):
#     if not os.path.exists(args.root_dir):
#         os.makedirs(args.root_dir)

#     dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

#     log_dir = os.path.join(args.root_dir, args.run_name)
#     log_path = os.path.join(
#         log_dir,
#         f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl"
#     )
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     run_strategy = strategy_factory(args.strategy)

#     print(f"""
# -----
# Starting run with the following parameters:
#   Strategy:    {args.strategy}
#   Model:       {args.model}
#   Language:    {args.language}
#   Pass@k:      {args.pass_at_k}
#   Max iters:   {args.max_iters}
#   Dataset:     {args.dataset_path}
#   Logs:        {log_dir}
# -----
# """)

#     # Load dataset
#     print('Loading the dataset...')
#     if args.dataset_path.endswith(".jsonl"):
#         dataset = read_jsonl(args.dataset_path)
#     elif args.dataset_path.endswith(".jsonl.gz"):
#         dataset = read_jsonl_gz(args.dataset_path)
#     else:
#         raise ValueError(f"Dataset path `{args.dataset_path}` is not supported")
#     print(f"Loaded {len(dataset)} examples")

#     # Shared trajectory store for retrieval strategy
#     trajectory_store = TrajectoryStore() \
#         if args.strategy == "retrieval_reflexion" else None

#     # Run strategy
#     try:
#         run_strategy(
#             dataset=dataset,
#             model_name=args.model,
#             language=args.language,
#             max_iters=args.max_iters,
#             pass_at_k=args.pass_at_k,
#             log_path=log_path,
#             verbose=args.verbose,
#             expansion_factor=args.expansion_factor,
#             is_leetcode=args.is_leetcode,
#             trajectory_store=trajectory_store,
#         )
#     except Exception as e:
#         print(f"Run failed with error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # Always compute and save metrics
#         metrics = compute_metrics(log_path)
#         if metrics:
#             print(f"\n--- Results for {args.strategy} ---")
#             print(f"  Pass@1:          {metrics['pass_at_1']:.2%}")
#             print(f"  Solved:          {metrics['solved']}/{metrics['total']}")
#             print(f"  Avg iterations:  {metrics['avg_iters']}")
#             print(f"  Avg reflections: {metrics['avg_reflections']}")
#             save_metrics_log(metrics, log_dir, args.run_name)

#     print(f"Done! Check out the logs in `{log_path}`")


# if __name__ == "__main__":
#     args = get_args()
#     main(args)

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from immediate_refinement import run_immediate_refinement
from immediate_reflexion import run_immediate_reflexion
from simple import run_simple
from reflexion import run_reflexion
from cot_gt import run_cot_gt
from retrieval_reflexion import run_retrieval_reflexion
from test_acc import run_test_acc
from programming_agents import TrajectoryStore
from utils import read_jsonl, read_jsonl_gz

from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `cot_gt`, `reflexion`, `retrieval_reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument("--model", type=str,
                        help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)
    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    # if strategy == "simple":
    #     return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "trajectory_store"])
    # elif strategy == "cot_gt":
    #     return kwargs_wrapper_gen(run_cot_gt, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
    elif strategy == "cot_gt":
        return kwargs_wrapper_gen(run_cot_gt, delete_keys=["expansion_factor", "trajectory_store"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor", "trajectory_store"])
    elif strategy == "retrieval":
        return kwargs_wrapper_gen(run_retrieval_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor", "trajectory_store"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor", "trajectory_store"])
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters", "trajectory_store"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


# ---------------------------------------------------------------------------
# Per-iteration metrics (HotpotQA-style)
# ---------------------------------------------------------------------------

def compute_per_iter_metrics(log_path: str, max_iters: int) -> Dict:
    """
    Parse the jsonl log to compute per-iteration metrics.

    Maps programming iterations to HotpotQA trials:
        iteration 0 = first attempt (no reflection)
        iteration 1 = after 1 reflection
        ...

    For each iteration i, counts how many problems were:
        - solved at or before iteration i  (cumulative success)
        - failed at iteration i (exhausted all iters without solving)
        - halted at iteration i (solved 0 internal tests, gave up early)
    """
    if not os.path.exists(log_path):
        return {}

    items = read_jsonl(log_path)
    if not items:
        return {}

    num_total = len(items)

    # Per-problem: how many implementations were tried and whether solved
    iter_solved   = []   # iteration at which solved (-1 if never)
    iter_counts   = []   # total iterations tried per problem
    halted_flags  = []   # True if problem gave up after 1 attempt (0 reflections)

    for item in items:
        impls       = item.get("implementations", [])
        reflections = item.get("reflections", [])
        is_solved   = item.get("is_solved", False)
        num_impls   = len(impls) if impls else 1

        iter_counts.append(num_impls)

        # Halted = failed AND no reflections generated (gave up immediately)
        halted = not is_solved and len(reflections) == 0
        halted_flags.append(halted)

        if is_solved:
            iter_solved.append(num_impls - 1)  # 0-indexed iteration when solved
        else:
            iter_solved.append(-1)

    # Build per-iteration tracking
    iter_numbers     = list(range(max_iters))
    success_rates    = []
    fail_rates       = []
    halted_rates     = []
    avg_steps        = []

    for it in iter_numbers:
        # Cumulative successes at iteration `it`
        num_solved_at_it = sum(1 for s in iter_solved if 0 <= s <= it)
        # Problems still failing at this iteration
        num_failed_at_it = sum(
            1 for i, s in enumerate(iter_solved)
            if s == -1 and iter_counts[i] > it and not halted_flags[i]
        )
        # Problems that halted (no reflections, gave up)
        num_halted_at_it = sum(1 for h in halted_flags if h)

        success_rates.append(num_solved_at_it / num_total)
        fail_rates.append(num_failed_at_it / num_total)
        halted_rates.append(num_halted_at_it / num_total)

        # Avg implementations tried up to this iteration (active problems)
        active_counts = [min(c, it + 1) for c in iter_counts]
        avg_steps.append(float(np.mean(active_counts)))

    return {
        "iter_numbers":  iter_numbers,
        "success_rates": success_rates,
        "fail_rates":    fail_rates,
        "halted_rates":  halted_rates,
        "avg_steps":     avg_steps,
        # Final summary
        "total":         num_total,
        "solved":        sum(1 for s in iter_solved if s >= 0),
        "failed":        sum(1 for i, s in enumerate(iter_solved) if s == -1 and not halted_flags[i]),
        "halted":        sum(1 for h in halted_flags if h),
        "pass_at_1":     round(sum(1 for s in iter_solved if s >= 0) / num_total, 4),
        "avg_iters":     round(float(np.mean(iter_counts)), 2),
        "avg_reflections": round(float(np.mean([
            len(item.get("reflections", [])) for item in items
            if not item.get("is_solved", False)
        ])) if any(not item.get("is_solved", False) for item in items) else 0.0, 2),
    }


# ---------------------------------------------------------------------------
# Plotting (HotpotQA-style)
# ---------------------------------------------------------------------------

def save_plots(metrics: Dict, log_dir: str, run_name: str, num_problems: int) -> None:
    """Save 3 HotpotQA-style plots."""
    iter_numbers  = metrics.get("iter_numbers",  [])
    success_rates = metrics.get("success_rates", [])
    fail_rates    = metrics.get("fail_rates",    [])
    halted_rates  = metrics.get("halted_rates",  [])
    avg_steps     = metrics.get("avg_steps",     [])

    if not iter_numbers:
        return

    # ── 1. Success rate per iteration ────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, success_rates, marker='o', linewidth=2,
             markersize=6, color='steelblue', label='Success Rate')
    plt.xlabel('Iteration Number')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate per Iteration — {run_name}')
    plt.xticks(iter_numbers)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_success_rate.png'), dpi=150)
    plt.close()

    # ── 2. Failed vs Halted per iteration ────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, fail_rates,   marker='s', linewidth=2,
             markersize=6, color='tomato',     label='Fail Rate')
    plt.plot(iter_numbers, halted_rates, marker='^', linewidth=2,
             markersize=6, color='darkorange', label='Halted Rate')
    plt.xlabel('Iteration Number')
    plt.ylabel('Rate')
    plt.title(f'Failed vs Halted Rate per Iteration — {run_name}')
    plt.xticks(iter_numbers)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_fail_halted.png'), dpi=150)
    plt.close()

    # ── 3. Avg steps (implementations) per iteration ─────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, avg_steps, marker='D', linewidth=2,
             markersize=6, color='mediumseagreen')
    plt.xlabel('Iteration Number')
    plt.ylabel('Avg Implementations Tried')
    plt.title(f'Avg Implementations per Iteration — {run_name}')
    plt.xticks(iter_numbers)
    plt.ylim(0, max(avg_steps) * 1.2 if avg_steps else 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_avg_steps.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {log_dir}")


def save_metrics_csv(metrics: Dict, log_dir: str, run_name: str, num_problems: int) -> None:
    """Save per-iteration metrics as CSV."""
    csv_path = os.path.join(log_dir, f'{num_problems}_problems_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('Iteration,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
        for it, s, fa, h, st in zip(
            metrics["iter_numbers"],
            metrics["success_rates"],
            metrics["fail_rates"],
            metrics["halted_rates"],
            metrics["avg_steps"],
        ):
            f.write(f'{it},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
    print(f"Metrics CSV saved to {csv_path}")


def save_metrics_json(metrics: Dict, log_dir: str, run_name: str) -> None:
    metrics_path = os.path.join(log_dir, f"{run_name}_metrics.json")
    # Save only summary (not per-iter lists) to keep JSON clean
    summary = {k: v for k, v in metrics.items()
               if k not in ("iter_numbers", "success_rates", "fail_rates",
                             "halted_rates", "avg_steps")}
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Metrics JSON saved to {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir,
        f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_strategy = strategy_factory(args.strategy)

    print(f"""
-----
Starting run with the following parameters:
  Strategy:    {args.strategy}
  Model:       {args.model}
  Language:    {args.language}
  Pass@k:      {args.pass_at_k}
  Max iters:   {args.max_iters}
  Dataset:     {args.dataset_path}
  Logs:        {log_dir}
-----
""")

    # Load dataset
    print('Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(f"Dataset path `{args.dataset_path}` is not supported")
    print(f"Loaded {len(dataset)} examples")

    # Shared trajectory store for retrieval strategy
    trajectory_store = TrajectoryStore() \
        if args.strategy == "retrieval_reflexion" else None

    # Run strategy
    try:
        run_strategy(
            dataset=dataset,
            model_name=args.model,
            language=args.language,
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
            expansion_factor=args.expansion_factor,
            is_leetcode=args.is_leetcode,
            trajectory_store=trajectory_store,
        )
    except Exception as e:
        print(f"Run failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always compute and save metrics even if crashed
        metrics = compute_per_iter_metrics(log_path, args.max_iters)
        if metrics:
            num_problems = metrics["total"]
            print(f"\n--- Results for {args.strategy} ---")
            print(f"  Pass@1:          {metrics['pass_at_1']:.2%}")
            print(f"  Solved:          {metrics['solved']}/{metrics['total']}")
            print(f"  Failed:          {metrics['failed']}/{metrics['total']}")
            print(f"  Halted:          {metrics['halted']}/{metrics['total']}")
            print(f"  Avg iterations:  {metrics['avg_iters']}")
            print(f"  Avg reflections: {metrics['avg_reflections']}")

            save_metrics_json(metrics, log_dir, args.run_name)
            save_metrics_csv(metrics, log_dir, args.run_name, num_problems)
            save_plots(metrics, log_dir, args.run_name, num_problems)
        else:
            print("No results to save yet.")

    print(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)