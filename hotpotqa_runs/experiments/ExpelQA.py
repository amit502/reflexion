import sys, os
import joblib
import numpy as np

sys.path.append('..')
root = '../root/'

from util import summarize_react_trial, log_react_trial
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy
#from retrieval_agents import ReactReflectAgent as ExpeL_ReactReflectAgent
from prompts import REFLECTION_SYSTEM_PROMPT
from llm import AnyOpenAILLM

sys.path.append('../..')
from expel_store import ExpeL, ExperienceRecord


# ---------------------------------------------------------------------------
# ExpeL-augmented ReactAgent — injects context into agent prompt
# ---------------------------------------------------------------------------

class ExpeL_ReactAgent(ReactAgent):
    """ReactAgent that prepends ExpeL context (insights + examples) to prompt."""

    def set_expel_context(self, context: str) -> None:
        self._expel_context = context

    def _build_agent_prompt(self) -> str:
        base = super()._build_agent_prompt()
        ctx  = getattr(self, '_expel_context', '')
        if ctx:
            return ctx + "\n\n" + base
        return base


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)
llm    = AnyOpenAILLM()
llm_fn = lambda p: llm(p, REFLECTION_SYSTEM_PROMPT)

N_GATHER = 5   # gathering trials (Reflexion)
N_TOTAL  = len(hotpot)

# ---------------------------------------------------------------------------
# Phase 1 — Experience Gathering (Reflexion for N_GATHER trials)
# ---------------------------------------------------------------------------

print(f"\n=== ExpeL Phase 1: Gathering ({N_GATHER} trials, {N_TOTAL} questions) ===")

expel = ExpeL(max_insights=10, retrieval_k=3)

gather_agents = [
    ReactReflectAgent(row['question'], row['answer'])
    for _, row in hotpot.iterrows()
]

gather_trial_numbers   = []
gather_success_rates   = []
gather_halt_rates      = []
gather_incorrect_rates = []
gather_avg_steps       = []
gather_log = ''
gather_trial = 0

for i in range(N_GATHER):
    active_agents = [a for a in gather_agents if not a.is_correct()]

    for agent in active_agents:
        agent.run(reflect_strategy=ReflexionStrategy.REFLEXION)
        print(f'Answer: {agent.key}')

    gather_trial += 1
    gather_log += log_react_trial(gather_agents, gather_trial)
    correct, incorrect, halted = summarize_react_trial(gather_agents)
    print(f'Gathering Trial {gather_trial} | Correct: {len(correct)} | '
          f'Incorrect: {len(incorrect)} | Halted: {len(halted)}')

    total        = len(gather_agents)
    success_rate = len(correct)   / total if total > 0 else 0
    halt_rate    = len(halted)    / total if total > 0 else 0
    inc_rate     = len(incorrect) / total if total > 0 else 0
    steps        = float(np.mean([a.step_n for a in active_agents])) if active_agents else 0.0

    gather_trial_numbers.append(gather_trial)
    gather_success_rates.append(success_rate)
    gather_halt_rates.append(halt_rate)
    gather_incorrect_rates.append(inc_rate)
    gather_avg_steps.append(steps)

# Store all gathered trajectories in ExpeL pool
for agent in gather_agents:
    expel.add(ExperienceRecord(
        task_id   = agent.question[:50],
        task_desc = agent.question,
        trajectory= agent.scratchpad,
        success   = agent.is_correct(),
        answer    = agent.key,
    ))

print(f"\nExpeL pool: "
      f"{len([r for r in expel.pool if r.success])} successes, "
      f"{len([r for r in expel.pool if not r.success])} failures")

# Extract insights
print("\n=== ExpeL: Extracting insights ===")
expel.extract_insights(llm_fn)
for i, ins in enumerate(expel.insights, 1):
    print(f"  {i}. {ins}")

# ---------------------------------------------------------------------------
# Phase 2 — Evaluation (single attempt with ExpeL context)
# ---------------------------------------------------------------------------

print(f"\n=== ExpeL Phase 2: Evaluation (single attempt, {N_TOTAL} questions) ===")

eval_agents = [
    ExpeL_ReactAgent(row['question'], row['answer'])
    for _, row in hotpot.iterrows()
]

# Inject ExpeL context into each agent
for agent in eval_agents:
    context = expel.format_inference_context(agent.question)
    agent.set_expel_context(context)

# Single attempt — no reflection
eval_trial_numbers   = [1]
eval_success_rates   = []
eval_halt_rates      = []
eval_incorrect_rates = []
eval_avg_steps       = []
eval_log = ''

for agent in eval_agents:
    agent.run()
    print(f'Answer: {agent.key} | Correct: {agent.is_correct()}')

eval_log += log_react_trial(eval_agents, 1)
correct, incorrect, halted = summarize_react_trial(eval_agents)
print(f'\nEval Results | Correct: {len(correct)} | '
      f'Incorrect: {len(incorrect)} | Halted: {len(halted)}')

total        = len(eval_agents)
success_rate = len(correct)   / total if total > 0 else 0
halt_rate    = len(halted)    / total if total > 0 else 0
inc_rate     = len(incorrect) / total if total > 0 else 0
steps        = float(np.mean([a.step_n for a in eval_agents]))

eval_success_rates.append(success_rate)
eval_halt_rates.append(halt_rate)
eval_incorrect_rates.append(inc_rate)
eval_avg_steps.append(steps)

print(f'  Success: {success_rate:.1%} | Incorrect: {inc_rate:.1%} | '
      f'Halted: {halt_rate:.1%} | Avg Steps: {steps:.1f}')

# ---------------------------------------------------------------------------
# Save logs + CSV  (same format as RetrievalQA.py)
# ---------------------------------------------------------------------------

base_path = os.path.join(root, 'ExpeL', 'hotpot')
os.makedirs(base_path, exist_ok=True)

# Gather log
gather_log_path = os.path.join(base_path, f'{N_TOTAL}_questions_{N_GATHER}_gather_trials.txt')
with open(gather_log_path, 'w') as f:
    f.write(gather_log)

# Eval log
eval_log_path = os.path.join(base_path, f'{N_TOTAL}_questions_eval.txt')
with open(eval_log_path, 'w') as f:
    f.write(eval_log)

# Insights log
insights_path = os.path.join(base_path, 'insights.txt')
with open(insights_path, 'w') as f:
    f.write('\n'.join(expel.insights))
print(f'Insights saved to {insights_path}')

# ── Gather CSV ───────────────────────────────────────────────────────────────
gather_csv_path = os.path.join(base_path, f'{N_TOTAL}_questions_gather_metrics.csv')
with open(gather_csv_path, 'w') as f:
    f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
    for t, s, fa, h, st in zip(gather_trial_numbers, gather_success_rates,
                                gather_incorrect_rates, gather_halt_rates,
                                gather_avg_steps):
        f.write(f'{t},{s:.4f},{fa:.4f},{h:.4f},{st:.4f}\n')
print(f'Gather metrics CSV saved to {gather_csv_path}')

# ── Single unified CSV (matches ALFWorld format) ─────────────────────────────
csv_path = os.path.join(base_path, f'{N_TOTAL}_questions_eval_metrics.csv')
with open(csv_path, 'w') as f:
    f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
    # Trial 1 = eval result (single attempt)
    f.write(f'1,{eval_success_rates[0]:.4f},'
            f'{eval_incorrect_rates[0]:.4f},'
            f'{eval_halt_rates[0]:.4f},'
            f'{eval_avg_steps[0]:.4f}\n')
print(f'Eval Metrics CSV saved to {csv_path}')