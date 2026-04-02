import sys, os
import joblib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
root = '../root/'

from util import summarize_react_trial, log_react_trial
from star_agents import (
    STARReactAgent,
    StepKnowledgeStore,
    ReflexionStrategy,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

hotpot   = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)
strategy = ReflexionStrategy.STAR

# Shared knowledge store — persists across all agents and all trials
# so knowledge accumulates across 100 questions × 5 trials
knowledge_store = StepKnowledgeStore()

agents = [
    STARReactAgent(
        question        = row['question'],
        key             = row['answer'],
        knowledge_store = knowledge_store,
        retrieval_k     = 3,
        use_reflection  = True,
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

# ---------------------------------------------------------------------------
# Trial loop — identical structure to RetrievalQA.py
# ---------------------------------------------------------------------------

for i in range(n):
    active_agents = [a for a in agents if not a.is_correct()]

    for agent in active_agents:
        agent.run(reflect_strategy=strategy)
        print(f'Answer: {agent.key}')

    trial += 1
    log   += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial} | Correct: {len(correct)} | '
          f'Incorrect: {len(incorrect)} | Halted: {len(halted)}')
    print(f'  Knowledge store: {len(knowledge_store.knowledge)} rules accumulated')

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

    print(
        f'  Success: {success_rate:.1%} | '
        f'Incorrect: {inc_rate:.1%} | '
        f'Halted: {halt_rate:.1%} | '
        f'Avg Steps: {steps:.1f}'
    )

# ---------------------------------------------------------------------------
# Save — same format as RetrievalQA.py
# ---------------------------------------------------------------------------

base_path = os.path.join(root, 'ReAct', strategy.value)
os.makedirs(base_path, exist_ok=True)

# Full log
log_path = os.path.join(base_path,
    f'{len(agents)}_questions_{trial}_trials.txt')
with open(log_path, 'w') as f:
    f.write(log)

# Knowledge snapshot (extra artifact — not in RAR)
knowledge_path = os.path.join(base_path, 'knowledge_snapshot.txt')
with open(knowledge_path, 'w') as f:
    f.write(f"Total rules accumulated: {len(knowledge_store.knowledge)}\n\n")
    positives   = [sk for sk in knowledge_store.knowledge if sk.positive]
    corrections = [sk for sk in knowledge_store.knowledge if not sk.positive]
    f.write(f"Confirmed rules: {len(positives)}\n")
    f.write(f"Corrections:     {len(corrections)}\n\n")
    for i, sk in enumerate(knowledge_store.knowledge, 1):
        icon = "✓" if sk.positive else "✗"
        f.write(f"{i}. {icon} Intent: {sk.action_intent}\n")
        f.write(f"   Rule:   {sk.rule}\n\n")
print(f'Knowledge snapshot saved to {knowledge_path}')

# ── Success rate plot ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, success_rates, marker='o', linewidth=2,
         markersize=6, color='steelblue')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Number of Trials')
plt.xticks(trial_numbers)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_success_rate.png'), dpi=150)
plt.close()

# ── Halt vs Incorrect plot ───────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, halt_rates,      marker='s', linewidth=2,
         markersize=6, color='tomato',     label='Halt Rate')
plt.plot(trial_numbers, incorrect_rates, marker='^', linewidth=2,
         markersize=6, color='darkorange', label='Incorrect Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Rate')
plt.title('Halt Rate vs Incorrect Rate per Trial')
plt.xticks(trial_numbers)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_halt_incorrect.png'), dpi=150)
plt.close()

# ── Avg steps plot ───────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, avg_steps, marker='D', linewidth=2,
         markersize=6, color='mediumseagreen')
plt.xlabel('Number of Trials')
plt.ylabel('Average Steps')
plt.title('Average Steps per Trial (Active Agents Only)')
plt.xticks(trial_numbers)
plt.ylim(0, max(avg_steps) * 1.2 if avg_steps else 10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'{len(agents)}_questions_avg_steps.png'), dpi=150)
plt.close()

# ── CSV — same format as RetrievalQA.py ─────────────────────────────────────
csv_path = os.path.join(base_path, f'{len(agents)}_questions_metrics.csv')
with open(csv_path, 'w') as f:
    f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
    for t, s, fa, h, st in zip(trial_numbers, success_rates,
                                incorrect_rates, halt_rates, avg_steps):
        f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
print(f'Metrics CSV saved to {csv_path}')