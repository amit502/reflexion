"""
ExpeL baseline for HotPotQA.

Two phases:
    Phase 1 — Experience Gathering (run_expel_gather):
        Runs Reflexion for n_gather trials on all questions.
        Stores successful + failed trajectories in ExpeL pool.
        Extracts insights after gathering.

    Phase 2 — Evaluation (run_expel_eval):
        Single attempt per question.
        Injects insights + retrieved successes into ReAct prompt.
        No reflection at eval time.

Usage:
    # In your run script:
    from expel_hotpot import run_expel_gather, run_expel_eval
    from expel_store import ExpeL

    expel = ExpeL(max_insights=10, retrieval_k=3)

    # Phase 1
    run_expel_gather(agents, expel, llm_fn, n_gather=3)

    # Phase 2
    run_expel_eval(agents_eval, expel)
"""

import sys, os
import joblib
import numpy as np

sys.path.append('..')
root = '../root/'

from util import summarize_react_trial, log_react_trial
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy
from expel_store import ExpeL, ExperienceRecord
from llm import AnyOpenAILLM

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Phase 1 — Experience Gathering
# ---------------------------------------------------------------------------

def run_expel_gather(
        agents: List,
        expel: ExpeL,
        llm_fn,
        n_gather: int = 3,
) -> None:
    """
    Run Reflexion for n_gather trials to populate the ExpeL pool.
    After gathering, extract insights.

    Args:
        agents:   list of ReactReflectAgent (initialised with questions)
        expel:    ExpeL pool to populate
        llm_fn:   callable(prompt) -> str for insight extraction
        n_gather: number of gathering trials (original paper uses 3-5)
    """
    print(f"\n=== ExpeL Phase 1: Experience Gathering ({n_gather} trials) ===")

    for trial in range(n_gather):
        active = [a for a in agents if not a.is_correct()]
        print(f"Gathering trial {trial+1}/{n_gather} — {len(active)} active agents")

        for agent in active:
            agent.run(reflect_strategy=ReflexionStrategy.REFLEXION)

        correct, incorrect, halted = summarize_react_trial(agents)
        print(f"  Correct: {len(correct)} | Incorrect: {len(incorrect)} | Halted: {len(halted)}")

    # Store all trajectories in ExpeL pool
    for agent in agents:
        expel.add(ExperienceRecord(
            task_id=agent.question[:50],
            task_desc=agent.question,
            trajectory=agent.scratchpad,
            success=agent.is_correct(),
            answer=agent.key,
        ))

    print(f"\nExpeL pool: {len([r for r in expel.pool if r.success])} successes, "
          f"{len([r for r in expel.pool if not r.success])} failures")

    # Extract insights
    print("\n=== ExpeL: Extracting insights ===")
    expel.extract_insights(llm_fn)
    for i, ins in enumerate(expel.insights, 1):
        print(f"  {i}. {ins}")


# ---------------------------------------------------------------------------
# Phase 2 — Evaluation
# ---------------------------------------------------------------------------

EXPEL_REACT_SYSTEM_PROMPT = """You are an expert question-answering agent.
You will be given:
1. Generalised insights from past experience
2. Similar successful past trajectories
3. The current question to answer

Use the insights and past trajectories to guide your reasoning.
Interleave Thought, Action, Observation steps.
Use Search[entity] to search Wikipedia, Lookup[keyword] to look up a keyword,
and Finish[answer] when you have the final answer.
"""


def _build_expel_prompt(agent, expel_context: str) -> str:
    """Prepend ExpeL context to the standard ReAct prompt."""
    base_prompt = agent._build_agent_prompt()
    if expel_context:
        return expel_context + "\n\n" + base_prompt
    return base_prompt


def run_expel_eval(
        questions: List[dict],
        expel: ExpeL,
        root: str = '../root/',
) -> Tuple[List, List[float], List[float], List[float], List[float]]:
    """
    Single-attempt evaluation with ExpeL context injection.

    Args:
        questions: list of hotpot dicts with 'question' and 'answer'
        expel:     populated ExpeL pool with insights extracted
        root:      logging root directory

    Returns:
        agents, success_rates, fail_rates, halt_rates, avg_steps
    """
    print("\n=== ExpeL Phase 2: Evaluation (single attempt) ===")

    # Build fresh agents — no reflection history
    agents = [ReactAgent(row['question'], row['answer'])
              for _, row in questions.iterrows()]

    # Inject ExpeL context into each agent's first prompt
    # We do this by patching the agent's scratchpad with the context block
    # then running normally — the context is prepended as a system observation

    for agent in agents:
        context = expel.format_inference_context(agent.question)
        if context:
            # Inject as initial context in scratchpad
            agent.scratchpad = f"[ExpeL Context]\n{context}\n[End ExpeL Context]\n"

    # Single run — no reflection
    for agent in agents:
        agent.run()
        print(f'Answer: {agent.key} | Correct: {agent.is_correct()}')

    correct  = [a for a in agents if a.is_correct()]
    incorrect = [a for a in agents if not a.is_correct() and not a.is_halted()]
    halted   = [a for a in agents if a.is_halted()]
    total    = len(agents)

    success_rate = len(correct)  / total
    fail_rate    = len(incorrect) / total
    halt_rate    = len(halted)   / total
    avg_steps    = float(np.mean([a.step_n for a in agents]))

    print(f"\nExpeL Eval Results:")
    print(f"  Success: {success_rate:.1%} | Fail: {fail_rate:.1%} | "
          f"Halted: {halt_rate:.1%} | Avg Steps: {avg_steps:.1f}")

    return agents, [success_rate], [fail_rate], [halt_rate], [avg_steps]


# ---------------------------------------------------------------------------
# Full ExpeL run script (HotPotQA)
# ---------------------------------------------------------------------------

def main():
    hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)

    # ── Phase 1: Gathering ───────────────────────────────────────────────────
    gather_agents = [
        ReactReflectAgent(row['question'], row['answer'])
        for _, row in hotpot.iterrows()
    ]

    expel   = ExpeL(max_insights=10, retrieval_k=3)
    llm     = AnyOpenAILLM()
    llm_fn  = lambda p: llm(p, "You are a helpful assistant.")

    run_expel_gather(gather_agents, expel, llm_fn, n_gather=3)

    # ── Phase 2: Evaluation ──────────────────────────────────────────────────
    agents, success_rates, fail_rates, halt_rates, avg_steps = \
        run_expel_eval(hotpot, expel, root=root)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    base_path = os.path.join(root, 'ExpeL', 'hotpot')
    os.makedirs(base_path, exist_ok=True)

    csv_path = os.path.join(base_path, f'{len(agents)}_questions_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('Trial,SuccessRate,FailRate,HaltedRate,AvgSteps\n')
        f.write(f'1,{success_rates[0]:.4f},{fail_rates[0]:.4f},'
                f'{halt_rates[0]:.4f},{avg_steps[0]:.4f}\n')
    print(f'Metrics saved to {csv_path}')


if __name__ == '__main__':
    main()