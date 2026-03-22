# import sys, os
# import joblib
# from util import summarize_react_trial, log_react_trial, save_agents
# from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

# sys.path.append('..')
# root  = '../root/'
# hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
# strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
# agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
# agents = [agent_cls(row['question'], row['answer']) for i, row in hotpot.iterrows() if i<1]

# n = 1
# trial = 0
# log = ''

# for i in range(n):
#     for agent in [a for a in agents if not a.is_correct()]:
#         if strategy != ReflexionStrategy.NONE:
#             agent.run(reflect_strategy = strategy)
#         else:
#             agent.run()
#         print(f'Answer: {agent.key}')
#     trial += 1
#     log += log_react_trial(agents, trial)
#     correct, incorrect, halted = summarize_react_trial(agents)
#     print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

# file_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt')
# os.makedirs(os.path.dirname(file_path), exist_ok=True)
# with open(file_path, 'w') as f:
#     f.write(log)




# import sys, os
# import joblib
# import matplotlib.pyplot as plt

# sys.path.append('..')
# root  = '../root/'

# from util import summarize_react_trial, log_react_trial, save_agents
# from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy


# hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
# strategy: ReflexionStrategy = ReflexionStrategy.NONE
# agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
# agents = [agent_cls(row['question'], row['answer']) for i, row in hotpot.iterrows() if i < 1]

# n = 2
# trial = 0
# log = ''

# # --- Tracking ---
# trial_numbers = []
# success_rates = []

# for i in range(n):
#     for agent in [a for a in agents if not a.is_correct()]:
#         if strategy != ReflexionStrategy.NONE:
#             agent.run(reflect_strategy=strategy)
#         else:
#             agent.run()
#         print(f'Answer: {agent.key}')
#     trial += 1
#     log += log_react_trial(agents, trial)
#     correct, incorrect, halted = summarize_react_trial(agents)
#     print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

#     # --- Record success rate for this trial ---
#     total = len(agents)
#     success_rate = len(correct) / total if total > 0 else 0
#     trial_numbers.append(trial)
#     success_rates.append(success_rate)

# # --- Save log ---
# file_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt')
# os.makedirs(os.path.dirname(file_path), exist_ok=True)
# with open(file_path, 'w') as f:
#     f.write(log)

# # --- Save success rate log ---
# stats_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials_success_rate.txt')
# with open(stats_path, 'w') as f:
#     f.write('Trial Number, Success Rate\n')
#     for t, s in zip(trial_numbers, success_rates):
#         f.write(f'{t}, {s:.4f}\n')
# print(f'Success rate log saved to {stats_path}')

# # --- Plot success rate vs number of trials ---
# plt.figure(figsize=(8, 5))
# plt.plot(trial_numbers, success_rates, marker='o', linewidth=2, markersize=6, color='steelblue')
# plt.xlabel('Number of Trials')
# plt.ylabel('Success Rate')
# plt.title('Success Rate vs Number of Trials')
# plt.xticks(trial_numbers)
# plt.ylim(0, 1)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()

# plot_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_success_rate.png')
# plt.savefig(plot_path, dpi=150)
# #plt.show()
# print(f'Plot saved to {plot_path}')





import sys, os
import joblib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
root  = '../root/'

from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
strategy: ReflexionStrategy = ReflexionStrategy.NONE
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for i, row in hotpot.iterrows()]

n = 5
trial = 0
log = ''

# --- Tracking ---
trial_numbers   = []
success_rates   = []
halt_rates      = []
incorrect_rates = []
avg_steps       = []

for i in range(n):
    active_agents = [a for a in agents if not a.is_correct()]

    for agent in active_agents:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy=strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')

    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

    total = len(agents)

    # --- Success rate ---
    success_rate = len(correct)   / total if total > 0 else 0
    halt_rate    = len(halted)    / total if total > 0 else 0
    inc_rate     = len(incorrect) / total if total > 0 else 0

    # --- Avg steps — only over agents that were active this trial ---
    steps = np.mean([a.step_n for a in active_agents]) if active_agents else 0.0

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

# ── Save log ────────────────────────────────────────────────────────────────
file_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as f:
    f.write(log)

base_path = os.path.join(root, 'ReAct', strategy.value)

# ── 1. Success rate log + plot (existing) ───────────────────────────────────
# stats_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials_success_rate.txt')
# with open(stats_path, 'w') as f:
#     f.write('Trial Number,Success Rate\n')
#     for t, s in zip(trial_numbers, success_rates):
#         f.write(f'{t},{s:.4f}\n')
# print(f'Success rate log saved to {stats_path}')

plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, success_rates, marker='o', linewidth=2, markersize=6, color='steelblue')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Number of Trials')
plt.xticks(trial_numbers)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plot_path = os.path.join(base_path, f'{len(agents)}_questions_success_rate.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f'Success rate plot saved to {plot_path}')

# ── 2. Halt rate vs Incorrect rate log + plot ───────────────────────────────
# halt_stats_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials_halt_incorrect.txt')
# with open(halt_stats_path, 'w') as f:
#     f.write('Trial Number,Halt Rate,Incorrect Rate\n')
#     for t, h, inc in zip(trial_numbers, halt_rates, incorrect_rates):
#         f.write(f'{t},{h:.4f},{inc:.4f}\n')
# print(f'Halt/Incorrect log saved to {halt_stats_path}')

plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, halt_rates,      marker='s', linewidth=2, markersize=6,
         color='tomato',      label='Halt Rate')
plt.plot(trial_numbers, incorrect_rates, marker='^', linewidth=2, markersize=6,
         color='darkorange',  label='Incorrect Rate')
plt.xlabel('Number of Trials')
plt.ylabel('Rate')
plt.title('Halt Rate vs Incorrect Rate per Trial')
plt.xticks(trial_numbers)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
halt_plot_path = os.path.join(base_path, f'{len(agents)}_questions_halt_incorrect.png')
plt.savefig(halt_plot_path, dpi=150)
plt.close()
print(f'Halt/Incorrect plot saved to {halt_plot_path}')

# ── 3. Avg steps per trial log + plot ───────────────────────────────────────
# steps_stats_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials_avg_steps.txt')
# with open(steps_stats_path, 'w') as f:
#     f.write('Trial Number,Avg Steps\n')
#     for t, s in zip(trial_numbers, avg_steps):
#         f.write(f'{t},{s:.4f}\n')
# print(f'Avg steps log saved to {steps_stats_path}')

plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, avg_steps, marker='D', linewidth=2, markersize=6, color='mediumseagreen')
plt.xlabel('Number of Trials')
plt.ylabel('Average Steps')
plt.title('Average Steps per Trial (Active Agents Only)')
plt.xticks(trial_numbers)
plt.ylim(0, max(avg_steps) * 1.2 if avg_steps else 10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
steps_plot_path = os.path.join(base_path, f'{len(agents)}_questions_avg_steps.png')
plt.savefig(steps_plot_path, dpi=150)
plt.close()
print(f'Avg steps plot saved to {steps_plot_path}')

# ── Single unified CSV (matches ALFWorld format) ─────────────────────────────
csv_path = os.path.join(base_path, f'{len(agents)}_questions_metrics.csv')
with open(csv_path, 'w') as f:
    f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
    for t, s, fa, h, st in zip(trial_numbers, success_rates,
                                incorrect_rates, halt_rates, avg_steps):
        f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
print(f'Metrics CSV saved to {csv_path}')