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
from retrieval_reflexion_tapas import run_retrieval_reflexion
from test_acc import run_test_acc
from programming_agents_tapas import TrajectoryStore
from utils import read_jsonl, read_jsonl_gz

import sys
sys.path.append('..')
from expel_store import ExpeL
from expel_programming import run_expel_gather, run_expel_extract_insights, run_expel_eval
from policy_store import PolicyStore   # ← new

from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",         type=str)
    parser.add_argument("--root_dir",         type=str, default="root")
    parser.add_argument("--dataset_path",     type=str, default="root")
    parser.add_argument("--strategy",         type=str)
    parser.add_argument("--language",         type=str)
    parser.add_argument("--model",            type=str)
    parser.add_argument("--pass_at_k",        type=int, default=1)
    parser.add_argument("--max_iters",        type=int, default=10)
    parser.add_argument("--expansion_factor", type=int, default=3)
    parser.add_argument("--is_leetcode",      action='store_true')
    parser.add_argument("--verbose",          action='store_true')
    return parser.parse_args()


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys: del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple,              delete_keys=["expansion_factor", "trajectory_store", "policy_store"])
    elif strategy == "cot_gt":
        return kwargs_wrapper_gen(run_cot_gt,              delete_keys=["expansion_factor", "trajectory_store", "policy_store"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion,           delete_keys=["expansion_factor", "trajectory_store", "policy_store"])
    elif strategy == "retrieval":
        return kwargs_wrapper_gen(run_retrieval_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "tapas":
        return kwargs_wrapper_gen(run_retrieval_reflexion, delete_keys=["expansion_factor"])  # ← same fn, policy_store passed
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion,  delete_keys=["expansion_factor", "trajectory_store", "policy_store"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor", "trajectory_store", "policy_store"])
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc,             delete_keys=["expansion_factor", "max_iters", "trajectory_store", "policy_store"])
    elif strategy == "expel":
        def expel_noop(**kwargs): pass
        return expel_noop
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


# ---------------------------------------------------------------------------
# Metrics / plots / csv — identical to original
# ---------------------------------------------------------------------------

def compute_per_iter_metrics(log_path, max_iters):
    if not os.path.exists(log_path): return {}
    items = read_jsonl(log_path)
    if not items: return {}
    num_total = len(items)
    iter_solved = []; iter_counts = []; halted_flags = []
    for item in items:
        impls = item.get("implementations", [])
        reflections = item.get("reflections", [])
        is_solved = item.get("is_solved", False)
        num_impls = len(impls) if impls else 1
        iter_counts.append(num_impls)
        halted_flags.append(not is_solved and len(reflections) == 0)
        iter_solved.append(num_impls - 1 if is_solved else -1)

    iter_numbers = list(range(max_iters))
    success_rates = []; fail_rates = []; halted_rates = []; avg_steps = []
    for it in iter_numbers:
        num_solved = sum(1 for s in iter_solved if 0 <= s <= it)
        num_halted = sum(1 for h in halted_flags if h)
        num_failed = num_total - num_solved - num_halted
        success_rates.append(num_solved / num_total)
        fail_rates.append(max(num_failed, 0) / num_total)
        halted_rates.append(num_halted / num_total)
        avg_steps.append(float(np.mean([min(c, it + 1) for c in iter_counts])))

    return {
        "iter_numbers": iter_numbers, "success_rates": success_rates,
        "fail_rates": fail_rates, "halted_rates": halted_rates, "avg_steps": avg_steps,
        "total": num_total,
        "solved": sum(1 for s in iter_solved if s >= 0),
        "failed": sum(1 for i, s in enumerate(iter_solved) if s == -1 and not halted_flags[i]),
        "halted": sum(1 for h in halted_flags if h),
        "pass_at_1": round(sum(1 for s in iter_solved if s >= 0) / num_total, 4),
        "avg_iters": round(float(np.mean(iter_counts)), 2),
        "avg_reflections": round(float(np.mean([
            len(item.get("reflections", [])) for item in items
            if not item.get("is_solved", False)
        ])) if any(not item.get("is_solved", False) for item in items) else 0.0, 2),
    }


def save_plots(metrics, log_dir, run_name, num_problems, suffix=''):
    iter_numbers = metrics.get("iter_numbers", [])
    if not iter_numbers: return
    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, metrics["success_rates"], marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.xlabel('Iteration Number'); plt.ylabel('Success Rate')
    plt.title(f'Success Rate — {run_name}{suffix}')
    plt.xticks(iter_numbers); plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_success_rate{suffix}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, metrics["fail_rates"],   marker='s', linewidth=2, markersize=6, color='tomato',     label='Fail Rate')
    plt.plot(iter_numbers, metrics["halted_rates"], marker='^', linewidth=2, markersize=6, color='darkorange', label='Halted Rate')
    plt.xlabel('Iteration Number'); plt.ylabel('Rate')
    plt.title(f'Failed vs Halted — {run_name}{suffix}')
    plt.xticks(iter_numbers); plt.ylim(0, 1); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_fail_halted{suffix}.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(iter_numbers, metrics["avg_steps"], marker='D', linewidth=2, markersize=6, color='mediumseagreen')
    plt.xlabel('Iteration Number'); plt.ylabel('Avg Implementations Tried')
    plt.title(f'Avg Implementations — {run_name}{suffix}')
    plt.xticks(iter_numbers)
    plt.ylim(0, max(metrics["avg_steps"]) * 1.2 if metrics["avg_steps"] else 10)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{num_problems}_problems_avg_steps{suffix}.png'), dpi=150)
    plt.close()


def save_metrics_csv(metrics, log_dir, run_name, num_problems, suffix=''):
    csv_path = os.path.join(log_dir, f'{num_problems}_problems_metrics{suffix}.csv')
    with open(csv_path, 'w') as f:
        f.write('Iteration,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
        for it, s, fa, h, st in zip(metrics["iter_numbers"], metrics["success_rates"],
                                     metrics["fail_rates"], metrics["halted_rates"], metrics["avg_steps"]):
            f.write(f'{it},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
    print(f"Metrics CSV saved to {csv_path}")


def save_metrics_json(metrics, log_dir, run_name):
    path = os.path.join(log_dir, f"{run_name}_metrics.json")
    summary = {k: v for k, v in metrics.items()
               if k not in ("iter_numbers","success_rates","fail_rates","halted_rates","avg_steps")}
    with open(path, 'w') as f: json.dump(summary, f, indent=4)
    print(f"Metrics JSON saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if not os.path.exists(args.root_dir): os.makedirs(args.root_dir)

    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")
    log_dir  = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(log_dir,
        f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}"
        f"_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    print(f"Strategy: {args.strategy} | Model: {args.model} | Max iters: {args.max_iters}")

    print('Loading dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(f"Unsupported: {args.dataset_path}")
    print(f"Loaded {len(dataset)} examples")

    # ── Init stores ──────────────────────────────────────────────────────────
    trajectory_store = TrajectoryStore() if args.strategy in ("retrieval", "tapas") else None
    policy_store     = PolicyStore()     if args.strategy == "tapas" else None   # ← new
    gather_log       = None

    try:
        if args.strategy == "expel":
            expel      = ExpeL(max_insights=10, retrieval_k=3)
            gather_log = log_path.replace('.jsonl', '_gather.jsonl')

            print("\n=== ExpeL Phase 1: Gathering ===")
            run_expel_gather(dataset=dataset, model_name=args.model, language=args.language,
                             max_iters=args.max_iters, pass_at_k=args.pass_at_k,
                             log_path=gather_log, verbose=args.verbose,
                             expel=expel, is_leetcode=args.is_leetcode)

            print("\n=== ExpeL Phase 2: Extracting Insights ===")
            run_expel_extract_insights(expel, args.model)

            print("\n=== ExpeL Phase 3: Evaluation ===")
            if os.path.exists(log_path): os.remove(log_path)
            import copy
            eval_dataset = copy.deepcopy(dataset)
            for item in eval_dataset: item['is_solved'] = False
            run_expel_eval(dataset=eval_dataset, model_name=args.model, language=args.language,
                           pass_at_k=args.pass_at_k, log_path=log_path, verbose=args.verbose,
                           expel=expel, is_leetcode=args.is_leetcode)

        else:
            run_strategy = strategy_factory(args.strategy)
            run_strategy(
                dataset=dataset, model_name=args.model, language=args.language,
                max_iters=args.max_iters, pass_at_k=args.pass_at_k,
                log_path=log_path, verbose=args.verbose,
                expansion_factor=args.expansion_factor,
                is_leetcode=args.is_leetcode,
                trajectory_store=trajectory_store,
                policy_store=policy_store,   # ← new: None for non-TAPAS
            )

    except Exception as e:
        print(f"Run failed: {e}")
        import traceback; traceback.print_exc()

    finally:
        if args.strategy == "expel":
            if gather_log and os.path.exists(gather_log) and os.path.getsize(gather_log) > 0:
                g = compute_per_iter_metrics(gather_log, args.max_iters)
                if g:
                    print(f"\n--- ExpeL Gather: Pass@1={g['pass_at_1']:.2%} ---")
                    save_metrics_json(g, log_dir, args.run_name + '_gather')
                    save_metrics_csv(g, log_dir, args.run_name, g['total'], suffix='_gather_metrics')
                    save_plots(g, log_dir, args.run_name, g['total'], suffix='_gather')
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                e = compute_per_iter_metrics(log_path, args.max_iters)
                if e:
                    print(f"\n--- ExpeL Eval: Pass@1={e['pass_at_1']:.2%} ---")
                    save_metrics_json(e, log_dir, args.run_name + '_eval')
                    save_metrics_csv(e, log_dir, args.run_name, e['total'], suffix='_eval_metrics')
                    save_plots(e, log_dir, args.run_name, e['total'], suffix='_eval')
        else:
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                metrics = compute_per_iter_metrics(log_path, args.max_iters)
            else:
                metrics = {}
            if metrics:
                num_problems = metrics["total"]
                print(f"\n--- {args.strategy} | Pass@1={metrics['pass_at_1']:.2%} | "
                      f"Solved={metrics['solved']}/{metrics['total']} ---")
                save_metrics_json(metrics, log_dir, args.run_name)
                save_metrics_csv(metrics, log_dir, args.run_name, num_problems)
                save_plots(metrics, log_dir, args.run_name, num_problems)

                # Save TAPAS policy snapshot ← new
                if args.strategy == "tapas" and policy_store:
                    policy_path = os.path.join(log_dir, 'tapas_policy_snapshot.txt')
                    with open(policy_path, 'w') as f:
                        for key, pol in policy_store.policies.items():
                            f.write(f"\n{'='*60}\n")
                            f.write(pol.to_prompt_str())
                    print(f'TAPAS policy snapshot saved to {policy_path}')
            else:
                print("No results to save yet.")

    print(f"Done! Logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)