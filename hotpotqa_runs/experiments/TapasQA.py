import sys, os
import joblib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
root = '../root/'

from util import summarize_react_trial, log_react_trial
from tapas_retrieval_agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

sys.path.append('../..')
from policy_store import PolicyStore


# ---------------------------------------------------------------------------
# Data + shared stores
# ---------------------------------------------------------------------------

hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)

strategy         = ReflexionStrategy.TAPAS
policy_store     = PolicyStore()                    # ← shared across all agents + trials

# All agents share the same policy_store and trajectory_store
# trajectory_store is created inside ReactReflectAgent by default
# but we pass policy_store explicitly so policy accumulates across questions
agents = [
    ReactReflectAgent(
        row['question'], row['answer'],
        policy_store=policy_store,               # ← shared policy
    )
    for _, row in hotpot.iterrows()
]

n     = 5
trial = 0
log   = ''

trial_numbers   = []
success_rates   = []
halt_rates      = []
incorrect_rates = []
avg_steps       = []

for i in range(n):
    active_agents = [a for a in agents if not a.is_correct()]

    for agent in active_agents:
        agent.run(reflect_strategy=strategy)
        print(f'Answer: {agent.key}')

    trial += 1
    log   += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Trial {trial} | Correct: {len(correct)} | '
          f'Incorrect: {len(incorrect)} | Halted: {len(halted)}')

    total        = len(agents)
    success_rate = len(correct)   / total if total > 0 else 0
    halt_rate    = len(halted)    / total if total > 0 else 0
    inc_rate     = len(incorrect) / total if total > 0 else 0
    steps        = float(np.mean([a.step_n for a in active_agents])) if active_agents else 0.0

    trial_numbers.append(trial)
    success_rates.append(success_rate)
    halt_rates.append(halt_rate)
    incorrect_rates.append(inc_rate)
    avg_steps.append(steps)

    print(f'  Success: {success_rate:.1%} | Incorrect: {inc_rate:.1%} | '
          f'Halted: {halt_rate:.1%} | Avg Steps: {steps:.1f}')

    # Print policy state after each trial
    print(f'  PolicyStore: {len(policy_store.policies)} policies, '
          f'versions: { {k: v.version for k, v in policy_store.policies.items()} }')

# ---------------------------------------------------------------------------
# Save logs + CSV
# ---------------------------------------------------------------------------

base_path = os.path.join(root, 'TAPAS', 'hotpot')
os.makedirs(base_path, exist_ok=True)

# Log
log_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials.txt')
with open(log_path, 'w') as f:
    f.write(log)

# Policy snapshot
policy_path = os.path.join(base_path, 'policy_snapshot.txt')
with open(policy_path, 'w') as f:
    for key, pol in policy_store.policies.items():
        f.write(f"\n{'='*60}\n")
        f.write(pol.to_prompt_str())
print(f'Policy snapshot saved to {policy_path}')

# Plots
plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, success_rates, marker='o', linewidth=2, markersize=6, color='steelblue')
plt.xlabel('Trial Number'); plt.ylabel('Success Rate')
plt.title('TAPAS — Success Rate vs Trials (HotPotQA)')
plt.xticks(trial_numbers); plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_success_rate.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, halt_rates,      marker='s', linewidth=2, markersize=6,
         color='tomato',     label='Halt Rate')
plt.plot(trial_numbers, incorrect_rates, marker='^', linewidth=2, markersize=6,
         color='darkorange', label='Incorrect Rate')
plt.xlabel('Trial Number'); plt.ylabel('Rate')
plt.title('TAPAS — Halt vs Incorrect Rate (HotPotQA)')
plt.xticks(trial_numbers); plt.ylim(0, 1); plt.legend()
plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_halt_incorrect.png'), dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, avg_steps, marker='D', linewidth=2, markersize=6, color='mediumseagreen')
plt.xlabel('Trial Number'); plt.ylabel('Average Steps')
plt.title('TAPAS — Avg Steps per Trial (HotPotQA)')
plt.xticks(trial_numbers)
plt.ylim(0, max(avg_steps) * 1.2 if avg_steps else 10)
plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_avg_steps.png'), dpi=150)
plt.close()

# CSV
csv_path = os.path.join(base_path, f'{len(agents)}_questions_metrics.csv')
with open(csv_path, 'w') as f:
    f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
    for t, s, fa, h, st in zip(trial_numbers, success_rates,
                                incorrect_rates, halt_rates, avg_steps):
        f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
print(f'Metrics CSV saved to {csv_path}')