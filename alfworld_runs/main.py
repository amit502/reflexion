# import os
# import json
# import argparse

# from alfworld_trial import run_trial
# from generate_reflections import update_memory

# from typing import Any, List, Dict

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_trials", type=int, help="The number of trials to run")
#     parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
#     parser.add_argument("--run_name", type=str, help="The name of the run")
#     parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
#     parser.add_argument("--is_resume", action='store_true', help="To resume run")
#     parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
#     parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
#     parser.add_argument("--model", type=str, help="The model to use. One of `gpt-4`, `gpt-3.5-turbo`, or `text-davinci-003")

#     args = parser.parse_args()

#     assert args.num_trials > 0, "Number of trials should be positive"
#     assert args.num_envs > 0, "Number of environments should be positive"

#     return args

# def main(args) -> None:
#     if args.is_resume:
#         if not os.path.exists(args.resume_dir):
#             raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
#         logging_dir = args.resume_dir

#         # load environment configs
#         env_config_path: str = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
#         if not os.path.exists(env_config_path):
#             raise ValueError(f"Environment config file `{env_config_path}` does not exist")
#         with open(env_config_path, 'r') as rf:
#             env_configs: List[Dict[str, Any]] = json.load(rf)
#     else:
#         # Create the run directory
#         if not os.path.exists(args.run_name):
#             os.makedirs(args.run_name)
#         logging_dir = args.run_name

#         # initialize environment configs
#         env_configs: List[Dict[str, Any]] = []
#         for i in range(args.num_envs):
#             env_configs += [{
#                 'name': f'env_{i}',
#                 'memory': [],
#                 'is_success': False,
#                 'skip': False
#             }]
    
#     world_log_path: str = os.path.join(logging_dir, 'world.log')

#     # print start status to user
#     if args.is_resume:
#         print(f"""
#     -----
#     Resuming run with the following parameters:
#     Run name: {logging_dir}
#     Number of trials: {args.num_trials}
#     Number of environments: {args.num_envs}
#     Use memory: {args.use_memory}
#     Resume trial number: {args.start_trial_num}

#     Sending all logs to `{args.run_name}`
#     -----
#     """)
#     else:
#         print(f"""
#     -----
#     Starting run with the following parameters:
#     Run name: {logging_dir}
#     Number of trials: {args.num_trials}
#     Number of environments: {args.num_envs}
#     Use memory: {args.use_memory}

#     Sending all logs to `{args.run_name}`
#     -----
#     """)

#     # run trials
#     trial_idx = args.start_trial_num
#     while trial_idx < args.num_trials:
#         with open(world_log_path, 'a') as wf:
#             wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

#         # set paths to log files
#         trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx}.log')
#         trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
#         if os.path.exists(trial_log_path):
#             open(trial_log_path, 'w').close()
#         if os.path.exists(trial_env_configs_log_path):
#             open(trial_env_configs_log_path, 'w').close()

#         # run trial
#         run_trial(trial_log_path, world_log_path, trial_idx, env_configs, args.use_memory, args.model)

#         # update memory if needed
#         if args.use_memory:
#             env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs)

#         # log env configs for trial
#         with open(trial_env_configs_log_path, 'w') as wf:
#             json.dump(env_configs, wf, indent=4)

#         # log world for trial
#         with open(world_log_path, 'a') as wf:
#             wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

#         trial_idx += 1


# if __name__ == '__main__':
#     args = get_args()
#     main(args)





# import os
# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt

# from alfworld_trial import run_trial
# from generate_reflections import update_memory
# from alfword_agents import ReflexionStrategy, TrajectoryStore

# from typing import Any, List, Dict


# # ---------------------------------------------------------------------------
# # Argument parsing
# # ---------------------------------------------------------------------------

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_trials",       type=int,  help="The number of trials to run")
#     parser.add_argument("--num_envs",         type=int,  help="The number of environments per trial")
#     parser.add_argument("--run_name",         type=str,  help="The name of the run")
#     parser.add_argument("--use_memory",       action='store_true', help="Allow the Agent to use memory (Reflexion)")
#     parser.add_argument("--is_resume",        action='store_true', help="To resume run")
#     parser.add_argument("--resume_dir",       type=str,  help="If resume, the logging directory", default="")
#     parser.add_argument("--start_trial_num",  type=int,  help="If resume, the start trial num",   default=0)
#     parser.add_argument("--model",            type=str,  help="The model to use. One of `gpt-4`, `gpt-3.5-turbo`, or `text-davinci-003`")

#     # Strategy selection
#     parser.add_argument(
#         "--strategy",
#         type=str,
#         default="base",
#         choices=["base", "reflexion", "expert_context", "retrieved_trajectory_reflexion"],
#         help=(
#             "Experiment strategy:\n"
#             "  base                          -> Experiment 2: ReAct only\n"
#             "  expert_context                -> Experiment 1: CoT + Expert Context\n"
#             "  reflexion                     -> Experiment 3: ReAct + Reflexion\n"
#             "  retrieved_trajectory_reflexion-> Experiment 4: Retrieval-augmented Reflexion\n"
#         ),
#     )

#     args = parser.parse_args()
#     assert args.num_trials > 0, "Number of trials should be positive"
#     assert args.num_envs   > 0, "Number of environments should be positive"
#     return args


# # ---------------------------------------------------------------------------
# # Metrics helpers
# # ---------------------------------------------------------------------------

# def _parse_trial_metrics(world_log_path: str, trial_idx: int, num_envs: int):
#     """
#     Parse world.log to extract per-trial success / fail / halted counts.
#     Halted = agent exhausted (check_is_exhausted) but not marked SUCCESS.
#     We detect halted from trial log instead of world log.
#     """
#     successes = 0
#     fails     = 0
#     with open(world_log_path, 'r') as f:
#         for line in f:
#             if f'Trial #{trial_idx}: SUCCESS' in line:
#                 successes += 1
#             elif f'Trial #{trial_idx}: FAIL' in line:
#                 fails += 1
#     # halted = environments that were active but neither succeeded nor failed cleanly
#     # (env_history.check_is_exhausted() == True cases end up as FAIL in world.log,
#     #  so we separate them via the trial log in _parse_step_counts)
#     return successes, fails


# def _parse_step_counts(trial_log_path: str) -> List[int]:
#     """Extract per-environment step counts from the trial log."""
#     steps = []
#     with open(trial_log_path, 'r') as f:
#         for line in f:
#             if line.startswith('Steps:'):
#                 try:
#                     steps.append(int((line or '').strip().split(': ')[1]))
#                 except (IndexError, ValueError):
#                     pass
#     return steps


# def _parse_halted_count(trial_log_path: str) -> int:
#     """
#     Count environments where the agent was exhausted (repeated action loop)
#     vs those that simply ran out of steps (inefficient planning).
#     Both show as FAIL in world.log, but exhausted ones hit check_is_exhausted().
#     We detect this by looking for environments with very low step counts that failed —
#     a heuristic: steps < 5 AND FAIL = likely halted/exhausted.
#     This is an approximation; for exact counts, EnvironmentHistory.check_is_exhausted()
#     would need to be logged explicitly.
#     """
#     # For now return 0; can be refined with explicit logging
#     return 0


# # ---------------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------------

# def save_plots(base_path: str,
#                trial_numbers: List[int],
#                success_rates:   List[float],
#                fail_rates:      List[float],
#                halted_rates:    List[float],
#                avg_steps_list:  List[float],
#                num_agents:      int):

#     # ── 1. Success rate per trial ────────────────────────────────────────────
#     plt.figure(figsize=(8, 5))
#     plt.plot(trial_numbers, success_rates, marker='o', linewidth=2,
#              markersize=6, color='steelblue', label='Success Rate')
#     plt.xlabel('Trial Number')
#     plt.ylabel('Success Rate')
#     plt.title('Success Rate per Trial')
#     plt.xticks(trial_numbers)
#     plt.ylim(0, 1)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(base_path, f'{num_agents}_envs_success_rate.png'), dpi=150)
#     plt.close()

#     # ── 2. Failed vs Halted per trial ────────────────────────────────────────
#     plt.figure(figsize=(8, 5))
#     plt.plot(trial_numbers, fail_rates,   marker='s', linewidth=2,
#              markersize=6, color='tomato',    label='Fail Rate')
#     plt.plot(trial_numbers, halted_rates, marker='^', linewidth=2,
#              markersize=6, color='darkorange', label='Halted Rate')
#     plt.xlabel('Trial Number')
#     plt.ylabel('Rate')
#     plt.title('Failed vs Halted Rate per Trial')
#     plt.xticks(trial_numbers)
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(base_path, f'{num_agents}_envs_fail_halted.png'), dpi=150)
#     plt.close()

#     # ── 3. Avg steps per trial ───────────────────────────────────────────────
#     plt.figure(figsize=(8, 5))
#     plt.plot(trial_numbers, avg_steps_list, marker='D', linewidth=2,
#              markersize=6, color='mediumseagreen')
#     plt.xlabel('Trial Number')
#     plt.ylabel('Average Steps')
#     plt.title('Average Steps per Trial (Active Environments Only)')
#     plt.xticks(trial_numbers)
#     plt.ylim(0, max(avg_steps_list) * 1.2 if avg_steps_list else 10)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(base_path, f'{num_agents}_envs_avg_steps.png'), dpi=150)
#     plt.close()


# def save_metrics_log(base_path: str,
#                      trial_numbers:  List[int],
#                      success_rates:  List[float],
#                      fail_rates:     List[float],
#                      halted_rates:   List[float],
#                      avg_steps_list: List[float],
#                      num_agents:     int):
#     log_path = os.path.join(base_path, f'{num_agents}_envs_metrics.csv')
#     with open(log_path, 'w') as f:
#         f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
#         for t, s, fa, h, st in zip(trial_numbers, success_rates,
#                                     fail_rates, halted_rates, avg_steps_list):
#             f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
#     print(f'Metrics log saved to {log_path}')


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main(args) -> None:

#     # ── Strategy resolution ──────────────────────────────────────────────────
#     strategy_map = {
#         'base':                           ReflexionStrategy.NONE,
#         'expert_context':                 ReflexionStrategy.EXPERT_CONTEXT,
#         'reflexion':                      ReflexionStrategy.REFLEXION,
#         'retrieved_trajectory_reflexion': ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
#     }
#     strategy: ReflexionStrategy = strategy_map[args.strategy]

#     # For strategies that need memory, force use_memory on
#     if strategy in (ReflexionStrategy.REFLEXION,
#                     ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION):
#         use_memory = True
#     else:
#         use_memory = args.use_memory

#     # Shared trajectory store for retrieval strategy (persists across trials)
#     trajectory_store: TrajectoryStore = TrajectoryStore() \
#         if strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION else None

#     # ── Logging dir ──────────────────────────────────────────────────────────
#     if args.is_resume:
#         if not os.path.exists(args.resume_dir):
#             raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
#         logging_dir = args.resume_dir
#         env_config_path = os.path.join(
#             args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json'
#         )
#         if not os.path.exists(env_config_path):
#             raise ValueError(f"Environment config file `{env_config_path}` does not exist")
#         with open(env_config_path, 'r') as rf:
#             env_configs: List[Dict[str, Any]] = json.load(rf)
#     else:
#         if not os.path.exists(args.run_name):
#             os.makedirs(args.run_name)
#         logging_dir = args.run_name
#         env_configs: List[Dict[str, Any]] = [
#             {
#                 'name':       f'env_{i}',
#                 'memory':     [],
#                 'is_success': False,
#                 'skip':       False,
#                 'steps':      0,
#                 'error_class': 'UNKNOWN',
#             }
#             for i in range(args.num_envs)
#         ]

#     world_log_path: str = os.path.join(logging_dir, 'world.log')

#     print(f"""
# -----
# Starting run with the following parameters:
#   Run name:    {logging_dir}
#   Strategy:    {strategy.value}
#   Trials:      {args.num_trials}
#   Envs:        {args.num_envs}
#   Use memory:  {use_memory}
#   Model:       {args.model}
# -----
# """)

#     # ── Metrics tracking ─────────────────────────────────────────────────────
#     trial_numbers:   List[int]   = []
#     success_rates:   List[float] = []
#     fail_rates:      List[float] = []
#     halted_rates:    List[float] = []
#     avg_steps_list:  List[float] = []

#     # ── Trial loop ───────────────────────────────────────────────────────────
#     trial_idx = args.start_trial_num
#     try:
#         while trial_idx < args.num_trials:

#             with open(world_log_path, 'a') as wf:
#                 wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

#             trial_log_path            = os.path.join(args.run_name, f'trial_{trial_idx}.log')
#             trial_env_configs_log_path = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')

#             if os.path.exists(trial_log_path):
#                 open(trial_log_path, 'w').close()
#             if os.path.exists(trial_env_configs_log_path):
#                 open(trial_env_configs_log_path, 'w').close()

#             # ── Run trial ────────────────────────────────────────────────────────
#             try:
#                 env_configs = run_trial(
#                     trial_log_path=trial_log_path,
#                     world_log_path=world_log_path,
#                     trial_idx=trial_idx,
#                     env_configs=env_configs,
#                     use_memory=use_memory,
#                     model=args.model,
#                     strategy=strategy,
#                     trajectory_store=trajectory_store,
#                 )
#             except Exception as e:
#                     print(f'Trial {trial_idx} failed with error: {e}')

#             # ── Update memory (Reflexion standard) ───────────────────────────────
#             if strategy == ReflexionStrategy.REFLEXION:
#                 env_configs = update_memory(trial_log_path, env_configs)

#             # ── Save env configs ─────────────────────────────────────────────────
#             with open(trial_env_configs_log_path, 'w') as wf:
#                 json.dump(env_configs, wf, indent=4)

#             # ── Compute metrics ──────────────────────────────────────────────────
#             num_envs     = len(env_configs)
#             num_success  = sum(1 for e in env_configs if e['is_success'])
#             # steps from env_configs (updated by run_trial)
#             active_steps = [e['steps'] for e in env_configs if e.get('steps', 0) > 0]
#             avg_steps    = float(np.mean(active_steps)) if active_steps else 0.0

#             # Parse halted vs failed from trial log
#             step_counts   = _parse_step_counts(trial_log_path)
#             # Halted: exhausted (repeated action, very few steps) — approximate
#             num_halted    = sum(1 for s in step_counts if 0 < s <= 3)
#             num_failed    = (num_envs - num_success) - num_halted

#             success_rate  = num_success / num_envs
#             fail_rate     = max(num_failed,  0) / num_envs
#             halted_rate   = max(num_halted,  0) / num_envs

#             trial_numbers.append(trial_idx)
#             success_rates.append(success_rate)
#             fail_rates.append(fail_rate)
#             halted_rates.append(halted_rate)
#             avg_steps_list.append(avg_steps)

#             print(
#                 f'Trial {trial_idx} | '
#                 f'Success: {success_rate:.1%} | '
#                 f'Fail: {fail_rate:.1%} | '
#                 f'Halted: {halted_rate:.1%} | '
#                 f'Avg Steps: {avg_steps:.1f}'
#             )

#             with open(world_log_path, 'a') as wf:
#                 wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

#             trial_idx += 1
#     finally:
#          # Always save whatever data we have, even if crashed mid-run
#         if trial_numbers:
#             # ── Save metrics + plots ─────────────────────────────────────────────────
#             save_metrics_log(
#                 base_path=logging_dir,
#                 trial_numbers=trial_numbers,
#                 success_rates=success_rates,
#                 fail_rates=fail_rates,
#                 halted_rates=halted_rates,
#                 avg_steps_list=avg_steps_list,
#                 num_agents=args.num_envs,
#             )
#             save_plots(
#                 base_path=logging_dir,
#                 trial_numbers=trial_numbers,
#                 success_rates=success_rates,
#                 fail_rates=fail_rates,
#                 halted_rates=halted_rates,
#                 avg_steps_list=avg_steps_list,
#                 num_agents=args.num_envs,
#             )
#             print("Done. All metrics and plots saved.")
#         else:
#             print("No trials completed, nothing to save.")


# if __name__ == '__main__':
#     args = get_args()
#     main(args)



import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from alfworld_trial import run_trial
from generate_reflections import update_memory
from alfword_agents import ReflexionStrategy, TrajectoryStore
from utils import get_chat  # ← for llm_fn in insight extraction

import sys
sys.path.append('..')
from expel_store import ExpeL, ExperienceRecord
from expel_alfworld import expel_store_trial_results, build_expel_alfworld_prefix

from typing import Any, List, Dict, Optional


# ---------------------------------------------------------------------------
# Argument parsing  (only change: add 'expel' to choices)
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials",       type=int)
    parser.add_argument("--num_envs",         type=int)
    parser.add_argument("--run_name",         type=str)
    parser.add_argument("--use_memory",       action='store_true')
    parser.add_argument("--is_resume",        action='store_true')
    parser.add_argument("--resume_dir",       type=str, default="")
    parser.add_argument("--start_trial_num",  type=int, default=0)
    parser.add_argument("--model",            type=str)
    parser.add_argument(
        "--strategy",
        type=str,
        default="base",
        choices=["base", "reflexion", "expert_context",
                 "retrieved_trajectory_reflexion", "expel"],  # ← added expel
    )
    parser.add_argument(
        "--expel_n_gather", type=int, default=3,
        help="Number of Reflexion gathering trials before ExpeL eval (default 3)"
    )
    args = parser.parse_args()
    assert args.num_trials > 0
    assert args.num_envs   > 0
    return args


# ---------------------------------------------------------------------------
# Metrics helpers (unchanged)
# ---------------------------------------------------------------------------

def _parse_step_counts(trial_log_path: str) -> List[int]:
    steps = []
    with open(trial_log_path, 'r') as f:
        for line in f:
            if line.startswith('Steps:'):
                try:
                    steps.append(int(line.strip().split(': ')[1]))
                except (IndexError, ValueError):
                    pass
    return steps


def save_plots(base_path, trial_numbers, success_rates, fail_rates,
               halted_rates, avg_steps_list, num_agents):
    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, success_rates, marker='o', linewidth=2,
             markersize=6, color='steelblue')
    plt.xlabel('Trial Number'); plt.ylabel('Success Rate')
    plt.title('Success Rate per Trial')
    plt.xticks(trial_numbers); plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_success_rate.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, fail_rates,   marker='s', linewidth=2,
             markersize=6, color='tomato',     label='Fail Rate')
    plt.plot(trial_numbers, halted_rates, marker='^', linewidth=2,
             markersize=6, color='darkorange',  label='Halted Rate')
    plt.xlabel('Trial Number'); plt.ylabel('Rate')
    plt.title('Failed vs Halted Rate per Trial')
    plt.xticks(trial_numbers); plt.ylim(0, 1); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_fail_halted.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, avg_steps_list, marker='D', linewidth=2,
             markersize=6, color='mediumseagreen')
    plt.xlabel('Trial Number'); plt.ylabel('Average Steps')
    plt.title('Average Steps per Trial')
    plt.xticks(trial_numbers)
    plt.ylim(0, max(avg_steps_list) * 1.2 if avg_steps_list else 10)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(base_path, f'{num_agents}_envs_avg_steps.png'), dpi=150)
    plt.close()


def save_metrics_log(base_path, trial_numbers, success_rates, fail_rates,
                     halted_rates, avg_steps_list, num_agents):
    log_path = os.path.join(base_path, f'{num_agents}_envs_metrics.csv')
    with open(log_path, 'w') as f:
        f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
        for t, s, fa, h, st in zip(trial_numbers, success_rates,
                                    fail_rates, halted_rates, avg_steps_list):
            f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
    print(f'Metrics log saved to {log_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:

    # ── Strategy resolution ──────────────────────────────────────────────────
    strategy_map = {
        'base':                           ReflexionStrategy.NONE,
        'expert_context':                 ReflexionStrategy.EXPERT_CONTEXT,
        'reflexion':                      ReflexionStrategy.REFLEXION,
        'retrieved_trajectory_reflexion': ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
        'expel':                          ReflexionStrategy.REFLEXION,  # gathering uses Reflexion
    }
    strategy: ReflexionStrategy = strategy_map[args.strategy]

    if strategy in (ReflexionStrategy.REFLEXION,
                    ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION):
        use_memory = True
    else:
        use_memory = args.use_memory

    trajectory_store: Optional[TrajectoryStore] = TrajectoryStore() \
        if strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION else None

    # ── ExpeL init ───────────────────────────────────────────────────────────
    is_expel      = args.strategy == 'expel'
    expel         = ExpeL(max_insights=10, retrieval_k=3) if is_expel else None
    n_gather      = args.expel_n_gather if is_expel else 0
    expel_ready   = False   # becomes True after insight extraction
    llm_fn        = lambda p: get_chat(p, model=args.model, stop_strs=[])

    # ── Logging dir ──────────────────────────────────────────────────────────
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume dir `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir
        env_config_path = os.path.join(
            args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
        if not os.path.exists(env_config_path):
            raise ValueError(f"Env config `{env_config_path}` does not exist")
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

    world_log_path: str = os.path.join(logging_dir, 'world.log')

    print(f"""
-----
Starting run:
  Strategy:  {args.strategy}
  Trials:    {args.num_trials}
  Envs:      {args.num_envs}
  Model:     {args.model}
  ExpeL:     {is_expel} (gather={n_gather} trials)
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

            if os.path.exists(trial_log_path):
                open(trial_log_path, 'w').close()
            if os.path.exists(trial_env_configs_log_path):
                open(trial_env_configs_log_path, 'w').close()

            # ── ExpeL: extract insights after last gathering trial ────────────
            if is_expel and trial_idx == n_gather and not expel_ready:
                print(f"\n=== ExpeL: Extracting insights after {n_gather} gathering trials ===")
                expel.extract_insights(llm_fn)
                expel_ready = True
                for i, ins in enumerate(expel.insights, 1):
                    print(f"  {i}. {ins}")
                # Reset env_configs so eval starts fresh
                env_configs = [
                    {'name': f'env_{i}', 'memory': [], 'is_success': False,
                     'skip': False, 'steps': 0, 'error_class': 'UNKNOWN'}
                    for i in range(args.num_envs)
                ]
                print("=== ExpeL: Starting eval phase ===\n")

            # ── Determine run strategy for this trial ────────────────────────
            if is_expel and trial_idx < n_gather:
                # Gathering phase — run Reflexion
                run_strategy  = ReflexionStrategy.REFLEXION
                run_memory    = True
                run_expel     = None
            elif is_expel and trial_idx >= n_gather:
                # Eval phase — run base ReAct with ExpeL context injected
                run_strategy  = ReflexionStrategy.NONE
                run_memory    = False
                run_expel     = expel
            else:
                run_strategy  = strategy
                run_memory    = use_memory
                run_expel     = None

            # ── Run trial ────────────────────────────────────────────────────
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
                    expel=run_expel,           # ← new param
                )
            except Exception as e:
                print(f'Trial {trial_idx} failed: {e}')
                import traceback; traceback.print_exc()

            # ── ExpeL: store gathering trajectories ──────────────────────────
            if is_expel and trial_idx < n_gather:
                expel_store_trial_results(env_configs, trial_log_path, expel)
                env_configs = update_memory(trial_log_path, env_configs)

            # ── Standard Reflexion memory update ─────────────────────────────
            elif strategy == ReflexionStrategy.REFLEXION:
                env_configs = update_memory(trial_log_path, env_configs)

            with open(trial_env_configs_log_path, 'w') as wf:
                json.dump(env_configs, wf, indent=4)

            # ── Metrics ──────────────────────────────────────────────────────
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

            with open(world_log_path, 'a') as wf:
                wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

            trial_idx += 1

    finally:
        if trial_numbers:
            save_metrics_log(logging_dir, trial_numbers, success_rates,
                             fail_rates, halted_rates, avg_steps_list, args.num_envs)
            save_plots(logging_dir, trial_numbers, success_rates,
                       fail_rates, halted_rates, avg_steps_list, args.num_envs)
            print("Done. All metrics and plots saved.")
        else:
            print("No trials completed, nothing to save.")


if __name__ == '__main__':
    args = get_args()
    main(args)