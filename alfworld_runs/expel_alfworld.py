"""
ExpeL baseline for ALFWorld.

Plugs into existing main.py and alfworld_trial.py infrastructure.
Add 'expel' to strategy choices in main.py and call run_expel_alfworld_gather
during gathering trials, then run_trial with expel context at eval.

Two phases:
    Phase 1 — Gathering: run standard Reflexion trials, store trajectories
    Phase 2 — Eval: single attempt with insights + retrieved successes injected
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

from expel_store import ExpeL, ExperienceRecord


# ---------------------------------------------------------------------------
# Phase 1 — Gathering (called from main.py after each gathering trial)
# ---------------------------------------------------------------------------

def expel_store_trial_results(
        env_configs: List[Dict[str, Any]],
        trial_log_path: str,
        expel: ExpeL,
) -> None:
    """
    After each gathering trial, read the trial log and store
    trajectories in the ExpeL pool.

    Call this after each gathering trial in main.py:
        expel_store_trial_results(env_configs, trial_log_path, expel)
    """
    # Parse trajectories from trial log
    if not os.path.exists(trial_log_path):
        return

    with open(trial_log_path, 'r') as f:
        content = f.read()

    # Split by environment blocks
    blocks = content.split('#####')
    for block in blocks:
        if not block.strip():
            continue
        is_success = 'STATUS: OK' in block

        # Extract task description from block
        task_desc = ''
        for line in block.split('\n'):
            if line.strip().startswith('Task:'):
                task_desc = line.strip()[5:].strip()
                break

        if not task_desc:
            continue

        # Use block as trajectory
        trajectory = block.strip()[:1500]

        expel.add(ExperienceRecord(
            task_id=task_desc[:50],
            task_desc=task_desc,
            trajectory=trajectory,
            success=is_success,
        ))

    print(f"ExpeL pool: {len([r for r in expel.pool if r.success])} successes, "
          f"{len([r for r in expel.pool if not r.success])} failures")


# ---------------------------------------------------------------------------
# Phase 2 — Build ExpeL prompt prefix for ALFWorld
# ---------------------------------------------------------------------------

def build_expel_alfworld_prefix(task_desc: str, expel: ExpeL) -> str:
    """
    Build the ExpeL context block to prepend to the ALFWorld base prompt.
    Injects insights + retrieved successful trajectories.
    """
    context = expel.format_inference_context(task_desc)
    if not context:
        return ''
    return (
        "=== EXPERIENCE FROM PAST TASKS ===\n"
        f"{context}\n"
        "=== END OF PAST EXPERIENCE ===\n\n"
    )


# ---------------------------------------------------------------------------
# Integration instructions for main.py
# ---------------------------------------------------------------------------
#
# In main.py, add 'expel' to strategy choices and handle as follows:
#
# GATHERING PHASE (first N trials):
#   expel = ExpeL(max_insights=10, retrieval_k=3)
#   for trial_idx in range(n_gather):
#       env_configs = run_trial(..., strategy=ReflexionStrategy.REFLEXION, ...)
#       expel_store_trial_results(env_configs, trial_log_path, expel)
#   expel.extract_insights(llm_fn)
#
# EVAL PHASE (single trial):
#   for z, env_config in enumerate(env_configs):
#       task_desc = ...
#       expel_prefix = build_expel_alfworld_prefix(task_desc, expel)
#       base_prompt  = expel_prefix + standard_base_prompt
#       # run with base_prompt, no memory, no reflection
#