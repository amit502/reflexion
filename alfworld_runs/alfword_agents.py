"""
ALFWorld Agents — mirrors the HotpotQA agents.py structure.

Strategies:
    NONE                          -> ReAct only (base run)
    REFLEXION                     -> ReAct + Reflexion (standard)
    EXPERT_CONTEXT                -> CoT + Expert context (analog of CoT+Context)
    RETRIEVED_TRAJECTORY_REFLEXION -> Retrieval-augmented reflexion (novel method)
"""

import re
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class ReflexionStrategy(Enum):
    """
    NONE:                           ReAct only — no memory, no reflection
    REFLEXION:                      ReAct + standard reflexion (last 3 reflections in memory)
    EXPERT_CONTEXT:                 CoT + Expert context (ALFWorld analog of CoT+Context)
    RETRIEVED_TRAJECTORY_REFLEXION: Retrieve top-k similar past trajectories by task type
                                    + embedding similarity, diversified via MMR, then reflect
    """
    NONE                           = 'base'
    REFLEXION                      = 'reflexion'
    EXPERT_CONTEXT                 = 'expert_context'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# ---------------------------------------------------------------------------
# ALFWorld error taxonomy
# ---------------------------------------------------------------------------

ALFWORLD_ERROR_TAXONOMY = [
    "WRONG_LOCATION",        # looked in wrong receptacles
    "WRONG_OBJECT",          # picked up / interacted with wrong object
    "LOOP_DETECTED",         # repeated same action cycle
    "INEFFICIENT_PLAN",      # exceeded step budget without progress
    "MISSING_STEP",          # skipped a required sub-task (e.g. forgot to clean before place)
    "OBJECT_NOT_IN_HAND",    # tried to use object not currently held
    "RECEPTACLE_NOT_OPEN",   # tried to place in closed container
    "UNKNOWN",
]

ALFWORLD_TASK_TYPES = {
    'pick_and_place':        'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place':  'heat',
    'pick_cool_then_place':  'cool',
    'look_at_obj':           'examine',
    'pick_two_obj':          'puttwo',
}


# ---------------------------------------------------------------------------
# TrajectoryRecord + TrajectoryStore  (ALFWorld-adapted)
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    """Stores a single ALFWorld episode for later retrieval."""

    def __init__(self,
                 task_type: str,
                 task_desc: str,
                 history_str: str,
                 reflection: str,
                 success: bool,
                 error_class: str = "UNKNOWN"):
        self.task_type   = task_type    # e.g. 'pick_and_place'
        self.task_desc   = task_desc    # the natural-language task string
        self.history_str = history_str  # full action/observation trace
        self.reflection  = reflection
        self.success     = success
        self.error_class = error_class
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.task_desc)
        return self._embedding


class TrajectoryStore:
    """
    Episodic memory of past ALFWorld trajectories.

    Retrieval scoring:
        score = λ_type * (task_type_i == task_type_curr)   # same task category
              + λ_q    * cos_sim(task_desc_i, task_desc_curr)  # semantic similarity
              + λ_err  * (error_class_i == error_class_curr)   # same failure mode

    Then MMR is applied for diversity, and results are balanced between
    failures and successes.
    """

    _st_model = None  # class-level cache

    def __init__(self,
                 embed_fn=None,
                 lambda_type: float = 0.4,
                 lambda_q:    float = 0.4,
                 lambda_err:  float = 0.2,
                 mmr_lambda:  float = 0.5):
        self.records:     List[TrajectoryRecord] = []
        self.embed_fn     = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.lambda_type  = lambda_type
        self.lambda_q     = lambda_q
        self.lambda_err   = lambda_err
        self.mmr_lambda   = mmr_lambda

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self,
                 task_type:   str,
                 task_desc:   str,
                 error_class: str,
                 k:               int = 5,
                 max_failures:    int = 3,
                 max_successes:   int = 2) -> List[TrajectoryRecord]:
        if not self.records:
            return []

        q_emb = self.embed_fn(task_desc)

        scored: List[Tuple[float, TrajectoryRecord]] = []
        for rec in self.records:
            sim_type = float(rec.task_type == task_type)
            sim_q    = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
            sim_err  = float(rec.error_class == error_class)
            score    = (self.lambda_type * sim_type
                        + self.lambda_q  * sim_q
                        + self.lambda_err * sim_err)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        failures  = [(s, r) for s, r in scored if not r.success]
        successes = [(s, r) for s, r in scored if r.success]

        selected_failures  = self._mmr_select(failures,  max_failures)
        selected_successes = self._mmr_select(successes, max_successes)

        return selected_failures + selected_successes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mmr_select(self,
                    scored: List[Tuple[float, TrajectoryRecord]],
                    k: int) -> List[TrajectoryRecord]:
        if not scored:
            return []
        selected: List[TrajectoryRecord] = []
        candidates = list(scored)
        while len(selected) < k and candidates:
            if not selected:
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score = -1e9
                best = None
                sel_embs = [r.embedding(self.embed_fn) for r in selected]
                for rel_score, rec in candidates:
                    max_sim = max(
                        float(np.dot(rec.embedding(self.embed_fn), se))
                        for se in sel_embs
                    )
                    mmr = (self.mmr_lambda * rel_score
                           - (1 - self.mmr_lambda) * max_sim)
                    if mmr > best_score:
                        best_score = mmr
                        best = rec
            selected.append(best)
            candidates = [(s, r) for s, r in candidates if r is not best]
        return selected

    @staticmethod
    def _get_st_model():
        if TrajectoryStore._st_model is None:
            from sentence_transformers import SentenceTransformer
            TrajectoryStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return TrajectoryStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        model = TrajectoryStore._get_st_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float64)


# ---------------------------------------------------------------------------
# Prompt helpers
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

def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
    if not records:
        return ''
    lines = []
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


def _truncate_history(history_str: str, max_chars: int = 1500) -> str:
    """Truncate long trajectories from the middle to fit context."""
    if len(history_str) <= max_chars:
        return history_str
    half = max_chars // 2
    return history_str[:half] + '\n...[truncated]...\n' + history_str[-half:]


def classify_alfworld_error(task_desc: str,
                             history_str: str,
                             llm_fn) -> str:
    """
    Ask the LLM to classify the ALFWorld failure into one of the known error types.
    llm_fn: callable(prompt: str) -> str
    """
    prompt = (
        "Classify the following failed household-task trajectory into exactly ONE of these error types:\n"
        f"{', '.join(ALFWORLD_ERROR_TAXONOMY)}\n\n"
        f"Task: {task_desc}\n\n"
        f"Trajectory:\n{_truncate_history(history_str)}\n\n"
        "Reply with only the error type label, nothing else."
    )
    raw = llm_fn(prompt).strip().upper()
    for label in ALFWORLD_ERROR_TAXONOMY:
        if label in raw:
            return label
    return "UNKNOWN"


def build_retrieval_reflection_prompt(task_type: str,
                                       task_desc: str,
                                       history_str: str,
                                       error_class: str,
                                       retrieved: List[TrajectoryRecord]) -> str:
    retrieved_context = format_retrieved_trajectories(retrieved)

    current_block = (
        "\n=== CURRENT FAILED TRAJECTORY ===\n"
        f"Task type:  {task_type}\n"
        f"Task:       {task_desc}\n"
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

    return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_block + instruction