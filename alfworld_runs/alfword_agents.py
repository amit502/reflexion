"""
ALFWorld Agents — Attention-Weighted Retrieval.

Strategies:
    NONE                          -> ReAct only (base run)
    REFLEXION                     -> ReAct + Reflexion (standard)
    EXPERT_CONTEXT                -> CoT + Expert context
    RETRIEVED_TRAJECTORY_REFLEXION -> Attention-weighted retrieval-augmented reflexion
"""

import re
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional


class ReflexionStrategy(Enum):
    NONE                           = 'base'
    REFLEXION                      = 'reflexion'
    EXPERT_CONTEXT                 = 'expert_context'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


ALFWORLD_ERROR_TAXONOMY = [
    "WRONG_LOCATION",
    "WRONG_OBJECT",
    "LOOP_DETECTED",
    "INEFFICIENT_PLAN",
    "MISSING_STEP",
    "OBJECT_NOT_IN_HAND",
    "RECEPTACLE_NOT_OPEN",
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


class TrajectoryRecord:
    """Stores a single ALFWorld episode for later retrieval."""

    def __init__(self,
                 task_type: str,
                 task_desc: str,
                 history_str: str,
                 reflection: str,
                 success: bool,
                 error_class: str = "UNKNOWN"):
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
    """
    Episodic memory of past ALFWorld trajectories.

    Retrieval uses Attention-Weighted Selection:

        1. Compute raw dot-product scores between query and all stored keys
        2. Apply task-type bonus before softmax (same type = +1.0, structurally
           similar = +0.5) to bias attention toward relevant task types
        3. Softmax with temperature τ to get attention weights α_i:
               α_i = softmax(( q·k_i / sqrt(d) + type_bonus_i ) / τ)
        4. Select top-k by α_i — split into failure/success pools
        5. MMR applied within each pool for diversity

    Temperature τ controls selection sharpness:
        - Low τ  (0.05): sharp — only top trajectory gets high weight
                         good for early trials when store is sparse
        - High τ (0.5):  soft  — spreads weight across many trajectories
                         good for later trials with rich store
    Adaptive τ scales with store size for best of both.
    """

    _st_model = None

    def __init__(self,
                 embed_fn=None,
                 tau: float = 0.1,          # attention temperature
                 adaptive_tau: bool = True,  # scale tau with store size
                 mmr_lambda: float = 0.5):
        self.records:      List[TrajectoryRecord] = []
        self.embed_fn      = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.tau           = tau
        self.adaptive_tau  = adaptive_tau
        self.mmr_lambda    = mmr_lambda

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
        d     = q_emb.shape[0]

        # ── Adaptive temperature — sharper when store is small ───────────────
        if self.adaptive_tau:
            n = len(self.records)
            # tau scales from 0.05 (sparse) to 0.3 (rich)
            tau = 0.05 + 0.25 * min(n / 100.0, 1.0)
        else:
            tau = self.tau
        # ─────────────────────────────────────────────────────────────────────

        # ── Compute attention logits ─────────────────────────────────────────
        # logit_i = (q · k_i / sqrt(d) + type_bonus_i) / tau
        logits = []
        for rec in self.records:
            k_emb      = rec.embedding(self.embed_fn)
            dot        = float(np.dot(q_emb, k_emb)) / np.sqrt(d)

            # Task-type bonus biases attention toward same/similar task types
            if rec.task_type == task_type:
                type_bonus = 1.0
            elif rec.task_type in _STRUCTURALLY_SIMILAR.get(task_type, []):
                type_bonus = 0.5
            else:
                type_bonus = 0.0

            logits.append((dot + type_bonus) / tau)

        # Softmax over logits → attention weights
        logits_arr = np.array(logits)
        logits_arr -= logits_arr.max()          # numerical stability
        exp_logits  = np.exp(logits_arr)
        alphas      = exp_logits / exp_logits.sum()
        # ─────────────────────────────────────────────────────────────────────

        # Pair each record with its attention weight
        scored = list(zip(alphas.tolist(), self.records))
        scored.sort(key=lambda x: x[0], reverse=True)

        failures  = [(a, r) for a, r in scored if not r.success]
        successes = [(a, r) for a, r in scored if r.success]

        selected_failures  = self._mmr_select(failures,  max_failures)
        selected_successes = self._mmr_select(successes, max_successes)

        return selected_failures + selected_successes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mmr_select(self,
                    scored: List[Tuple[float, TrajectoryRecord]],
                    k: int) -> List[TrajectoryRecord]:
        """MMR using cosine similarity for diversity (unchanged logic)."""
        if not scored:
            return []
        selected   = []
        candidates = list(scored)
        while len(selected) < k and candidates:
            if not selected:
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score = -1e9
                best       = None
                sel_embs   = [r.embedding(self.embed_fn) for r in selected]
                for attn_score, rec in candidates:
                    max_sim = max(
                        float(np.dot(rec.embedding(self.embed_fn), se))
                        for se in sel_embs
                    )
                    mmr = (self.mmr_lambda * attn_score
                           - (1 - self.mmr_lambda) * max_sim)
                    if mmr > best_score:
                        best_score = mmr
                        best       = rec
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
        model = TrajectoryStore._get_st_model()
        vec   = model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float64)


# Structurally similar task types for attention type bonus
_STRUCTURALLY_SIMILAR = {
    'pick_clean_then_place': ['pick_heat_then_place', 'pick_cool_then_place'],
    'pick_heat_then_place':  ['pick_clean_then_place', 'pick_cool_then_place'],
    'pick_cool_then_place':  ['pick_heat_then_place', 'pick_clean_then_place'],
    'pick_and_place':        ['pick_two_obj'],
    'pick_two_obj':          ['pick_and_place'],
    'look_at_obj':           [],
}


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


def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
    if not records:
        return ''
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


def _truncate_history(history_str: str, max_chars: int = 1500) -> str:
    if len(history_str) <= max_chars:
        return history_str
    half = max_chars // 2
    return history_str[:half] + '\n...[truncated]...\n' + history_str[-half:]


def classify_alfworld_error(task_desc: str,
                             history_str: str,
                             llm_fn) -> str:
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