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

import sys, os
import joblib
import matplotlib.pyplot as plt

sys.path.append('..')
root  = '../root/'

from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION_WITH_CRITIQUE
agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for i, row in hotpot.iterrows() if i<1]

n = 3
trial = 0
log = ''

# --- Tracking ---
trial_numbers = []
success_rates = []

for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy=strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')

    # --- Record success rate for this trial ---
    total = len(agents)
    success_rate = len(correct) / total if total > 0 else 0
    trial_numbers.append(trial)
    success_rates.append(success_rate)

# --- Save log ---
file_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as f:
    f.write(log)

# --- Save success rate log ---
stats_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials_success_rate.txt')
with open(stats_path, 'w') as f:
    f.write('Trial Number, Success Rate\n')
    for t, s in zip(trial_numbers, success_rates):
        f.write(f'{t}, {s:.4f}\n')
print(f'Success rate log saved to {stats_path}')

# --- Plot success rate vs number of trials ---
plt.figure(figsize=(8, 5))
plt.plot(trial_numbers, success_rates, marker='o', linewidth=2, markersize=6, color='steelblue')
plt.xlabel('Number of Trials')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Number of Trials')
plt.xticks(trial_numbers)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_success_rate.png')
plt.savefig(plot_path, dpi=150)
#plt.show()
print(f'Plot saved to {plot_path}')