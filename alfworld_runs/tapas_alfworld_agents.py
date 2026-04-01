"""
ALFWorld Agents — Attention-Weighted Retrieval + TAPAS policy learning.

Strategies:
    NONE                           -> ReAct only
    REFLEXION                      -> ReAct + Reflexion
    EXPERT_CONTEXT                 -> CoT + Expert context
    RETRIEVED_TRAJECTORY_REFLEXION -> Attention-weighted retrieval
    TAPAS                          -> RAR + per-task-type policy learning
"""

import re
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

import sys
sys.path.append('..')
from policy_store import PolicyStore


class ReflexionStrategy(Enum):
    NONE                           = 'base'
    REFLEXION                      = 'reflexion'
    EXPERT_CONTEXT                 = 'expert_context'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'
    TAPAS                          = 'tapas'   # ← new


ALFWORLD_ERROR_TAXONOMY = [
    "WRONG_LOCATION", "WRONG_OBJECT", "LOOP_DETECTED",
    "INEFFICIENT_PLAN", "MISSING_STEP", "OBJECT_NOT_IN_HAND",
    "RECEPTACLE_NOT_OPEN", "UNKNOWN",
]

ALFWORLD_TASK_TYPES = {
    'pick_and_place':        'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place':  'heat',
    'pick_cool_then_place':  'cool',
    'look_at_obj':           'examine',
    'pick_two_obj':          'puttwo',
}

_STRUCTURALLY_SIMILAR = {
    'pick_clean_then_place': ['pick_heat_then_place', 'pick_cool_then_place'],
    'pick_heat_then_place':  ['pick_clean_then_place', 'pick_cool_then_place'],
    'pick_cool_then_place':  ['pick_heat_then_place', 'pick_clean_then_place'],
    'pick_and_place':        ['pick_two_obj'],
    'pick_two_obj':          ['pick_and_place'],
    'look_at_obj':           [],
}


class TrajectoryRecord:
    def __init__(self, task_type, task_desc, history_str, reflection,
                 success, error_class="UNKNOWN"):
        self.task_type   = task_type
        self.task_desc   = task_desc
        self.history_str = history_str
        self.reflection  = reflection
        self.success     = success
        self.error_class = error_class
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.task_desc)
        return self._embedding


class TrajectoryStore:
    _st_model = None

    def __init__(self, embed_fn=None, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
        self.records      = []
        self.embed_fn     = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.tau          = tau
        self.adaptive_tau = adaptive_tau
        self.mmr_lambda   = mmr_lambda

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self, task_type, task_desc, error_class,
                 k=5, max_failures=3, max_successes=2):
        if not self.records:
            return []

        q_emb = self.embed_fn(task_desc)
        d     = q_emb.shape[0]
        tau   = (0.05 + 0.25 * min(len(self.records) / 100.0, 1.0)
                 if self.adaptive_tau else self.tau)

        logits = []
        for rec in self.records:
            dot = float(np.dot(q_emb, rec.embedding(self.embed_fn))) / np.sqrt(d)
            if rec.task_type == task_type:
                type_bonus = 1.0
            elif rec.task_type in _STRUCTURALLY_SIMILAR.get(task_type, []):
                type_bonus = 0.5
            else:
                type_bonus = 0.0
            logits.append((dot + type_bonus) / tau)

        logits_arr  = np.array(logits)
        logits_arr -= logits_arr.max()
        alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()

        scored    = sorted(zip(alphas.tolist(), self.records), key=lambda x: x[0], reverse=True)
        failures  = [(a, r) for a, r in scored if not r.success]
        successes = [(a, r) for a, r in scored if r.success]
        return self._mmr_select(failures, max_failures) + self._mmr_select(successes, max_successes)

    def _mmr_select(self, scored, k):
        if not scored: return []
        selected, candidates = [], list(scored)
        while len(selected) < k and candidates:
            if not selected:
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score, best = -1e9, None
                sel_embs = [r.embedding(self.embed_fn) for r in selected]
                for attn_score, rec in candidates:
                    max_sim = max(float(np.dot(rec.embedding(self.embed_fn), se)) for se in sel_embs)
                    mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
                    if mmr > best_score:
                        best_score, best = mmr, rec
            selected.append(best)
            candidates = [(a, r) for a, r in candidates if r is not best]
        return selected

    @staticmethod
    def _get_st_model():
        if TrajectoryStore._st_model is None:
            from sentence_transformers import SentenceTransformer
            TrajectoryStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return TrajectoryStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        return TrajectoryStore._get_st_model().encode(text, normalize_embeddings=True).astype(np.float64)


# ---------------------------------------------------------------------------
# Prompt helpers (unchanged)
# ---------------------------------------------------------------------------

RETRIEVAL_REFLECTION_HEADER = """\
You are an expert at diagnosing failures in household task agents.
Below you will find:
  1. SIMILAR PAST TRAJECTORIES (labelled as success or failure).
  2. The CURRENT FAILED TRAJECTORY.

Study similarities and differences. Use successful trajectories as a
reference for the correct approach. Use failed ones to spot recurring mistakes.
Then write a concise, actionable reflection for the CURRENT trajectory ONLY.

"""


def format_retrieved_trajectories(records):
    if not records: return ''
    lines     = []
    successes = [r for r in records if r.success]
    failures  = [r for r in records if not r.success]
    if successes:
        lines.append("=== SIMILAR SUCCESSFUL TRAJECTORIES ===")
        for i, r in enumerate(successes, 1):
            lines.append(f"\n[Success {i}] Task type: {r.task_type} | Task: {r.task_desc}")
            lines.append(_truncate_history(r.history_str).strip())
            if r.reflection:
                lines.append(f"Reflection: {r.reflection}")
    if failures:
        lines.append("\n=== SIMILAR FAILED TRAJECTORIES ===")
        for i, r in enumerate(failures, 1):
            lines.append(f"\n[Failure {i}] Task type: {r.task_type} | Error: {r.error_class}")
            lines.append(f"Task: {r.task_desc}")
            lines.append(_truncate_history(r.history_str).strip())
            if r.reflection:
                lines.append(f"Reflection: {r.reflection}")
    return '\n'.join(lines)


def _truncate_history(history_str, max_chars=1500):
    if len(history_str) <= max_chars:
        return history_str
    half = max_chars // 2
    return history_str[:half] + '\n...[truncated]...\n' + history_str[-half:]


def classify_alfworld_error(task_desc, history_str, llm_fn):
    prompt = (
        "Classify the following failed household-task trajectory into exactly ONE of these error types:\n"
        f"{', '.join(ALFWORLD_ERROR_TAXONOMY)}\n\n"
        f"Task: {task_desc}\n\nTrajectory:\n{_truncate_history(history_str)}\n\n"
        "Reply with only the error type label, nothing else."
    )
    raw = llm_fn(prompt).strip().upper()
    for label in ALFWORLD_ERROR_TAXONOMY:
        if label in raw:
            return label
    return "UNKNOWN"


def build_retrieval_reflection_prompt(task_type, task_desc, history_str,
                                       error_class, retrieved,
                                       policy_str: str = ""):   # ← new param
    retrieved_context = format_retrieved_trajectories(retrieved)
    current_block = (
        "\n=== CURRENT FAILED TRAJECTORY ===\n"
        f"Task type:   {task_type}\n"
        f"Task:        {task_desc}\n"
        f"Error class: {error_class}\n\n"
        f"{_truncate_history(history_str).strip()}\n"
    )
    instruction = (
        "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
        "FAILED_STEP: <the action where things went wrong>\n"
        "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
        "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
        "e.g. 'go to fridge 1' instead of 'go to countertop 1'>\n"
        "GENERALISATION: <one sentence on when this fix applies beyond this exact task>\n"
    )
    # Policy block prepended before retrieved trajectories
    policy_block = (policy_str + "\n") if policy_str else ""
    return RETRIEVAL_REFLECTION_HEADER + policy_block + retrieved_context + current_block + instruction


def build_tapas_base_prompt_prefix(task_type: str,
                                    policy_store: Optional[PolicyStore]) -> str:
    """
    Inject the current policy for this task_type into the agent's base prompt.
    Called from alfworld_trial.py before alfworld_run().
    Returns empty string if no policy exists yet.
    """
    if policy_store is None:
        return ""
    policy = policy_store.get(task_type)
    return policy.to_prompt_str()