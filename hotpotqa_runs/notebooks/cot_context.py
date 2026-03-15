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
root = '../root/'

from agents import CoTAgent, ReflexionStrategy
from util import summarize_trial, log_trial, save_agents


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    if ind>0: continue
    supporting_articles = row['supporting_facts']['title']
    articles = row['context']['title']
    sentences = row['context']['sentences'] 
    supporting_paragraphs = []
    for article in supporting_articles:
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs

strategy: ReflexionStrategy = ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION

from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshots import COT, COT_REFLECT
agents = [CoTAgent(row['question'],
                   row['supporting_paragraphs'],
                   row['answer'],
                   agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                   cot_examples=COT,
                   reflect_prompt=cot_reflect_prompt,
                   reflect_examples=COT_REFLECT,
                    ) for _, row in hotpot.iterrows()]

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
            agent.run(reflexion_strategy=strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')

    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

    total = len(agents)

    # --- Success rate ---
    success_rate = len(correct)   / total if total > 0 else 0
    inc_rate     = len(incorrect) / total if total > 0 else 0

    # --- Avg steps — only over agents that were active this trial ---
    steps = np.mean([a.step_n for a in active_agents]) if active_agents else 0.0

    trial_numbers.append(trial)
    success_rates.append(success_rate)
    incorrect_rates.append(inc_rate)
    avg_steps.append(steps)

    print(
        f'  Success: {success_rate:.1%} | '
        f'Incorrect: {inc_rate:.1%} | '
        f'Avg Steps: {steps:.1f}'
    )

# ── Save log ────────────────────────────────────────────────────────────────
file_path = os.path.join(root, 'CoT','context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as f:
    f.write(log)

base_path = os.path.join(root, 'CoT','context', strategy.value)

# ── 1. Success rate log + plot (existing) ───────────────────────────────────
stats_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials_success_rate.txt')
with open(stats_path, 'w') as f:
    f.write('Trial Number,Success Rate\n')
    for t, s in zip(trial_numbers, success_rates):
        f.write(f'{t},{s:.4f}\n')
print(f'Success rate log saved to {stats_path}')

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

# ── 3. Avg steps per trial log + plot ───────────────────────────────────────
steps_stats_path = os.path.join(base_path, f'{len(agents)}_questions_{trial}_trials_avg_steps.txt')
with open(steps_stats_path, 'w') as f:
    f.write('Trial Number,Avg Steps\n')
    for t, s in zip(trial_numbers, avg_steps):
        f.write(f'{t},{s:.4f}\n')
print(f'Avg steps log saved to {steps_stats_path}')

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