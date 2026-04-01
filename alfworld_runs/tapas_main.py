import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from alfworld_trial import run_trial
from generate_reflections import update_memory
from tapas_alfworld_agents import ReflexionStrategy, TrajectoryStore  # ← tapas agents
from utils import get_chat

import sys
sys.path.append('..')
from expel_store import ExpeL, ExperienceRecord
from expel_alfworld import expel_store_trial_results, build_expel_alfworld_prefix
from policy_store import PolicyStore   # ← new

from typing import Any, List, Dict, Optional


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials",      type=int)
    parser.add_argument("--num_envs",        type=int)
    parser.add_argument("--run_name",        type=str)
    parser.add_argument("--use_memory",      action='store_true')
    parser.add_argument("--is_resume",       action='store_true')
    parser.add_argument("--resume_dir",      type=str, default="")
    parser.add_argument("--start_trial_num", type=int, default=0)
    parser.add_argument("--model",           type=str)
    parser.add_argument(
        "--strategy", type=str, default="base",
        choices=["base", "reflexion", "expert_context",
                 "retrieved_trajectory_reflexion", "expel", "tapas"],  # ← added tapas
    )
    parser.add_argument("--expel_n_gather", type=int, default=3)
    args = parser.parse_args()
    assert args.num_trials > 0
    assert args.num_envs   > 0
    return args


def _parse_step_counts(trial_log_path):
    steps = []
    with open(trial_log_path, 'r') as f:
        for line in f:
            if line.startswith('Steps:'):
                try: steps.append(int(line.strip().split(': ')[1]))
                except: pass
    return steps


def save_plots(base_path, trial_numbers, success_rates, fail_rates,
               halted_rates, avg_steps_list, num_agents, suffix=''):
    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, success_rates, marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.xlabel('Trial Number'); plt.ylabel('Success Rate'); plt.title('Success Rate per Trial')
    plt.xticks(trial_numbers); plt.ylim(0, 1); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_success_rate{suffix}.png'), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, fail_rates,   marker='s', linewidth=2, markersize=6, color='tomato',     label='Fail Rate')
    plt.plot(trial_numbers, halted_rates, marker='^', linewidth=2, markersize=6, color='darkorange',  label='Halted Rate')
    plt.xlabel('Trial Number'); plt.ylabel('Rate'); plt.title('Failed vs Halted Rate per Trial')
    plt.xticks(trial_numbers); plt.ylim(0, 1); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_fail_halted{suffix}.png'), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, avg_steps_list, marker='D', linewidth=2, markersize=6, color='mediumseagreen')
    plt.xlabel('Trial Number'); plt.ylabel('Average Steps'); plt.title('Average Steps per Trial')
    plt.xticks(trial_numbers); plt.ylim(0, max(avg_steps_list) * 1.2 if avg_steps_list else 10)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_avg_steps{suffix}.png'), dpi=150); plt.close()


def save_metrics_log(base_path, trial_numbers, success_rates, fail_rates,
                     halted_rates, avg_steps_list, num_agents, suffix=''):
    log_path = os.path.join(base_path, f'{num_agents}_envs_metrics{suffix}.csv')
    with open(log_path, 'w') as f:
        f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
        for t, s, fa, h, st in zip(trial_numbers, success_rates,
                                    fail_rates, halted_rates, avg_steps_list):
            f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
    print(f'Metrics log saved to {log_path}')


def main(args) -> None:

    strategy_map = {
        'base':                           ReflexionStrategy.NONE,
        'expert_context':                 ReflexionStrategy.EXPERT_CONTEXT,
        'reflexion':                      ReflexionStrategy.REFLEXION,
        'retrieved_trajectory_reflexion': ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
        'expel':                          ReflexionStrategy.REFLEXION,
        'tapas':                          ReflexionStrategy.TAPAS,   # ← new
    }
    strategy: ReflexionStrategy = strategy_map[args.strategy]

    if strategy in (ReflexionStrategy.REFLEXION,
                    ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
                    ReflexionStrategy.TAPAS):
        use_memory = True
    else:
        use_memory = args.use_memory

    trajectory_store: Optional[TrajectoryStore] = TrajectoryStore() \
        if strategy in (ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
                        ReflexionStrategy.TAPAS) else None

    # ── ExpeL init ───────────────────────────────────────────────────────────
    is_expel    = args.strategy == 'expel'
    expel       = ExpeL(max_insights=10, retrieval_k=3) if is_expel else None
    n_gather    = args.expel_n_gather if is_expel else 0
    expel_ready = False
    llm_fn      = lambda p: get_chat(p, model=args.model, stop_strs=[])

    # ── TAPAS init ───────────────────────────────────────────────────────────
    is_tapas      = args.strategy == 'tapas'
    policy_store  = PolicyStore() if is_tapas else None   # ← new

    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume dir `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir
        env_config_path = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
        if not os.path.exists(env_config_path):
            raise ValueError(f"Env config not found: {env_config_path}")
        with open(env_config_path, 'r') as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name
        env_configs: List[Dict[str, Any]] = [
            {'name': f'env_{i}', 'memory': [], 'is_success': False,
             'skip': False, 'steps': 0, 'error_class': 'UNKNOWN'}
            for i in range(args.num_envs)
        ]

    world_log_path = os.path.join(logging_dir, 'world.log')

    print(f"""
-----
  Strategy:  {args.strategy}  Trials: {args.num_trials}
  Envs:      {args.num_envs}  Model:  {args.model}
  ExpeL:     {is_expel} (gather={n_gather})  TAPAS: {is_tapas}
-----
""")

    trial_numbers:  List[int]   = []
    success_rates:  List[float] = []
    fail_rates:     List[float] = []
    halted_rates:   List[float] = []
    avg_steps_list: List[float] = []

    trial_idx = args.start_trial_num
    try:
        while trial_idx < args.num_trials:
            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

            trial_log_path             = os.path.join(args.run_name, f'trial_{trial_idx}.log')
            trial_env_configs_log_path = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')

            if os.path.exists(trial_log_path):            open(trial_log_path, 'w').close()
            if os.path.exists(trial_env_configs_log_path): open(trial_env_configs_log_path, 'w').close()

            # ── ExpeL insight extraction ─────────────────────────────────────
            if is_expel and trial_idx == n_gather and not expel_ready:
                print(f"\n=== ExpeL: Extracting insights ===")
                expel.extract_insights(llm_fn)
                expel_ready = True
                for i, ins in enumerate(expel.insights, 1): print(f"  {i}. {ins}")
                env_configs = [
                    {'name': f'env_{i}', 'memory': [], 'is_success': False,
                     'skip': False, 'steps': 0, 'error_class': 'UNKNOWN'}
                    for i in range(args.num_envs)
                ]

            # ── Strategy routing ─────────────────────────────────────────────
            if is_expel and trial_idx < n_gather:
                run_strategy = ReflexionStrategy.REFLEXION
                run_memory   = True
                run_expel    = None
            elif is_expel and trial_idx >= n_gather:
                run_strategy = ReflexionStrategy.NONE
                run_memory   = False
                run_expel    = expel
            else:
                run_strategy = strategy
                run_memory   = use_memory
                run_expel    = None

            try:
                env_configs = run_trial(
                    trial_log_path=trial_log_path,
                    world_log_path=world_log_path,
                    trial_idx=trial_idx,
                    env_configs=env_configs,
                    use_memory=run_memory,
                    model=args.model,
                    strategy=run_strategy,
                    trajectory_store=trajectory_store,
                    expel=run_expel,
                    policy_store=policy_store,   # ← new: None for non-TAPAS
                )
            except Exception as e:
                print(f'Trial {trial_idx} failed: {e}')
                import traceback; traceback.print_exc()

            if is_expel and trial_idx < n_gather:
                expel_store_trial_results(env_configs, trial_log_path, expel)
                env_configs = update_memory(trial_log_path, env_configs)
            elif strategy == ReflexionStrategy.REFLEXION:
                env_configs = update_memory(trial_log_path, env_configs)
            elif is_tapas:
                env_configs = update_memory(trial_log_path, env_configs)

            with open(trial_env_configs_log_path, 'w') as wf:
                json.dump(env_configs, wf, indent=4)

            num_envs     = len(env_configs)
            num_success  = sum(1 for e in env_configs if e['is_success'])
            active_steps = [e['steps'] for e in env_configs if e.get('steps', 0) > 0]
            avg_steps    = float(np.mean(active_steps)) if active_steps else 0.0
            step_counts  = _parse_step_counts(trial_log_path)
            num_halted   = sum(1 for s in step_counts if 0 < s <= 3)
            num_failed   = max((num_envs - num_success) - num_halted, 0)

            success_rate = num_success / num_envs
            fail_rate    = num_failed  / num_envs
            halted_rate  = num_halted  / num_envs

            trial_numbers.append(trial_idx)
            success_rates.append(success_rate)
            fail_rates.append(fail_rate)
            halted_rates.append(halted_rate)
            avg_steps_list.append(avg_steps)

            print(f'Trial {trial_idx} | Success: {success_rate:.1%} | '
                  f'Fail: {fail_rate:.1%} | Halted: {halted_rate:.1%} | '
                  f'Avg Steps: {avg_steps:.1f}')

            # Print policy state after each TAPAS trial
            if is_tapas and policy_store:
                print(f'  TAPAS policies: { {k: v.version for k, v in policy_store.policies.items()} }')

            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

            trial_idx += 1

    finally:
        if trial_numbers:
            if is_expel:
                g_idx = [i for i, t in enumerate(trial_numbers) if t < n_gather]
                e_idx = [i for i, t in enumerate(trial_numbers) if t >= n_gather]
                def _s(lst, idx): return [lst[i] for i in idx]
                if g_idx:
                    save_metrics_log(logging_dir, _s(trial_numbers,g_idx), _s(success_rates,g_idx),
                                     _s(fail_rates,g_idx), _s(halted_rates,g_idx),
                                     _s(avg_steps_list,g_idx), args.num_envs, suffix='_gather_metrics')
                    save_plots(logging_dir, _s(trial_numbers,g_idx), _s(success_rates,g_idx),
                               _s(fail_rates,g_idx), _s(halted_rates,g_idx),
                               _s(avg_steps_list,g_idx), args.num_envs, suffix='_gather')
                if e_idx:
                    save_metrics_log(logging_dir, _s(trial_numbers,e_idx), _s(success_rates,e_idx),
                                     _s(fail_rates,e_idx), _s(halted_rates,e_idx),
                                     _s(avg_steps_list,e_idx), args.num_envs, suffix='_eval_metrics')
                    save_plots(logging_dir, _s(trial_numbers,e_idx), _s(success_rates,e_idx),
                               _s(fail_rates,e_idx), _s(halted_rates,e_idx),
                               _s(avg_steps_list,e_idx), args.num_envs, suffix='_eval')
            else:
                save_metrics_log(logging_dir, trial_numbers, success_rates,
                                 fail_rates, halted_rates, avg_steps_list, args.num_envs)
                save_plots(logging_dir, trial_numbers, success_rates,
                           fail_rates, halted_rates, avg_steps_list, args.num_envs)

            # Save TAPAS policy snapshot
            if is_tapas and policy_store:
                policy_path = os.path.join(logging_dir, 'tapas_policy_snapshot.txt')
                with open(policy_path, 'w') as f:
                    for key, pol in policy_store.policies.items():
                        f.write(f"\n{'='*60}\n")
                        f.write(pol.to_prompt_str())
                print(f'TAPAS policy snapshot saved to {policy_path}')

            print("Done. All metrics and plots saved.")
        else:
            print("No trials completed, nothing to save.")


if __name__ == '__main__':
    args = get_args()
    main(args)