# """
# ALFWorld Agents — mirrors the HotpotQA agents.py structure.

# Strategies:
#     NONE                          -> ReAct only (base run)
#     REFLEXION                     -> ReAct + Reflexion (standard)
#     EXPERT_CONTEXT                -> CoT + Expert context (analog of CoT+Context)
#     RETRIEVED_TRAJECTORY_REFLEXION -> Retrieval-augmented reflexion (novel method)
# """

# import re
# import numpy as np
# from enum import Enum
# from typing import List, Dict, Any, Tuple, Optional


# # ---------------------------------------------------------------------------
# # Strategy enum
# # ---------------------------------------------------------------------------

# class ReflexionStrategy(Enum):
#     """
#     NONE:                           ReAct only — no memory, no reflection
#     REFLEXION:                      ReAct + standard reflexion (last 3 reflections in memory)
#     EXPERT_CONTEXT:                 CoT + Expert context (ALFWorld analog of CoT+Context)
#     RETRIEVED_TRAJECTORY_REFLEXION: Retrieve top-k similar past trajectories by task type
#                                     + embedding similarity, diversified via MMR, then reflect
#     """
#     NONE                           = 'base'
#     REFLEXION                      = 'reflexion'
#     EXPERT_CONTEXT                 = 'expert_context'
#     RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# # ---------------------------------------------------------------------------
# # ALFWorld error taxonomy
# # ---------------------------------------------------------------------------

# ALFWORLD_ERROR_TAXONOMY = [
#     "WRONG_LOCATION",        # looked in wrong receptacles
#     "WRONG_OBJECT",          # picked up / interacted with wrong object
#     "LOOP_DETECTED",         # repeated same action cycle
#     "INEFFICIENT_PLAN",      # exceeded step budget without progress
#     "MISSING_STEP",          # skipped a required sub-task (e.g. forgot to clean before place)
#     "OBJECT_NOT_IN_HAND",    # tried to use object not currently held
#     "RECEPTACLE_NOT_OPEN",   # tried to place in closed container
#     "UNKNOWN",
# ]

# ALFWORLD_TASK_TYPES = {
#     'pick_and_place':        'put',
#     'pick_clean_then_place': 'clean',
#     'pick_heat_then_place':  'heat',
#     'pick_cool_then_place':  'cool',
#     'look_at_obj':           'examine',
#     'pick_two_obj':          'puttwo',
# }


# # ---------------------------------------------------------------------------
# # TrajectoryRecord + TrajectoryStore  (ALFWorld-adapted)
# # ---------------------------------------------------------------------------

# class TrajectoryRecord:
#     """Stores a single ALFWorld episode for later retrieval."""

#     def __init__(self,
#                  task_type: str,
#                  task_desc: str,
#                  history_str: str,
#                  reflection: str,
#                  success: bool,
#                  error_class: str = "UNKNOWN"):
#         self.task_type   = task_type    # e.g. 'pick_and_place'
#         self.task_desc   = task_desc    # the natural-language task string
#         self.history_str = history_str  # full action/observation trace
#         self.reflection  = reflection
#         self.success     = success
#         self.error_class = error_class
#         self._embedding: Optional[np.ndarray] = None

#     def embedding(self, embed_fn) -> np.ndarray:
#         if self._embedding is None:
#             self._embedding = embed_fn(self.task_desc)
#         return self._embedding


# class TrajectoryStore:
#     """
#     Episodic memory of past ALFWorld trajectories.

#     Retrieval scoring:
#         score = λ_type * (task_type_i == task_type_curr)   # same task category
#               + λ_q    * cos_sim(task_desc_i, task_desc_curr)  # semantic similarity
#               + λ_err  * (error_class_i == error_class_curr)   # same failure mode

#     Then MMR is applied for diversity, and results are balanced between
#     failures and successes.
#     """

#     _st_model = None  # class-level cache

#     def __init__(self,
#                  embed_fn=None,
#                  lambda_type: float = 0.4,
#                  lambda_q:    float = 0.4,
#                  lambda_err:  float = 0.2,
#                  mmr_lambda:  float = 0.5):
#         self.records:     List[TrajectoryRecord] = []
#         self.embed_fn     = embed_fn if embed_fn is not None else self._sentence_transformer_embed
#         self.lambda_type  = lambda_type
#         self.lambda_q     = lambda_q
#         self.lambda_err   = lambda_err
#         self.mmr_lambda   = mmr_lambda

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def add(self, record: TrajectoryRecord) -> None:
#         self.records.append(record)

#     def retrieve(self,
#                  task_type:   str,
#                  task_desc:   str,
#                  error_class: str,
#                  k:               int = 5,
#                  max_failures:    int = 3,
#                  max_successes:   int = 2) -> List[TrajectoryRecord]:
#         if not self.records:
#             return []

#         q_emb = self.embed_fn(task_desc)

#         scored: List[Tuple[float, TrajectoryRecord]] = []
#         for rec in self.records:
#             sim_type = float(rec.task_type == task_type)
#             sim_q    = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
#             sim_err  = float(rec.error_class == error_class)
#             score    = (self.lambda_type * sim_type
#                         + self.lambda_q  * sim_q
#                         + self.lambda_err * sim_err)
#             scored.append((score, rec))

#         scored.sort(key=lambda x: x[0], reverse=True)

#         failures  = [(s, r) for s, r in scored if not r.success]
#         successes = [(s, r) for s, r in scored if r.success]

#         selected_failures  = self._mmr_select(failures,  max_failures)
#         selected_successes = self._mmr_select(successes, max_successes)

#         return selected_failures + selected_successes

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _mmr_select(self,
#                     scored: List[Tuple[float, TrajectoryRecord]],
#                     k: int) -> List[TrajectoryRecord]:
#         if not scored:
#             return []
#         selected: List[TrajectoryRecord] = []
#         candidates = list(scored)
#         while len(selected) < k and candidates:
#             if not selected:
#                 _, best = max(candidates, key=lambda x: x[0])
#             else:
#                 best_score = -1e9
#                 best = None
#                 sel_embs = [r.embedding(self.embed_fn) for r in selected]
#                 for rel_score, rec in candidates:
#                     max_sim = max(
#                         float(np.dot(rec.embedding(self.embed_fn), se))
#                         for se in sel_embs
#                     )
#                     mmr = (self.mmr_lambda * rel_score
#                            - (1 - self.mmr_lambda) * max_sim)
#                     if mmr > best_score:
#                         best_score = mmr
#                         best = rec
#             selected.append(best)
#             candidates = [(s, r) for s, r in candidates if r is not best]
#         return selected

#     @staticmethod
#     def _get_st_model():
#         if TrajectoryStore._st_model is None:
#             from sentence_transformers import SentenceTransformer
#             TrajectoryStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
#         return TrajectoryStore._st_model

#     @staticmethod
#     def _sentence_transformer_embed(text: str) -> np.ndarray:
#         model = TrajectoryStore._get_st_model()
#         vec = model.encode(text, normalize_embeddings=True)
#         return vec.astype(np.float64)


# # ---------------------------------------------------------------------------
# # Prompt helpers
# # ---------------------------------------------------------------------------

# RETRIEVAL_REFLECTION_HEADER = """\
# You are an expert at diagnosing failures in household task agents.
# Below you will find:
#   1. SIMILAR PAST TRAJECTORIES (labelled as success or failure).
#   2. The CURRENT FAILED TRAJECTORY.

# Study similarities and differences. Use successful trajectories as a
# reference for the correct approach. Use failed ones to spot recurring mistakes.
# Then write a concise, actionable reflection for the CURRENT trajectory ONLY.

# """

# def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
#     if not records:
#         return ''
#     lines = []
#     successes = [r for r in records if r.success]
#     failures  = [r for r in records if not r.success]

#     if successes:
#         lines.append("=== SIMILAR SUCCESSFUL TRAJECTORIES ===")
#         for i, r in enumerate(successes, 1):
#             lines.append(f"\n[Success {i}] Task type: {r.task_type} | Task: {r.task_desc}")
#             lines.append(_truncate_history(r.history_str).strip())
#             if r.reflection:
#                 lines.append(f"Reflection: {r.reflection}")

#     if failures:
#         lines.append("\n=== SIMILAR FAILED TRAJECTORIES ===")
#         for i, r in enumerate(failures, 1):
#             lines.append(f"\n[Failure {i}] Task type: {r.task_type} | Error: {r.error_class}")
#             lines.append(f"Task: {r.task_desc}")
#             lines.append(_truncate_history(r.history_str).strip())
#             if r.reflection:
#                 lines.append(f"Reflection: {r.reflection}")

#     return '\n'.join(lines)


# def _truncate_history(history_str: str, max_chars: int = 1500) -> str:
#     """Truncate long trajectories from the middle to fit context."""
#     if len(history_str) <= max_chars:
#         return history_str
#     half = max_chars // 2
#     return history_str[:half] + '\n...[truncated]...\n' + history_str[-half:]


# def classify_alfworld_error(task_desc: str,
#                              history_str: str,
#                              llm_fn) -> str:
#     """
#     Ask the LLM to classify the ALFWorld failure into one of the known error types.
#     llm_fn: callable(prompt: str) -> str
#     """
#     prompt = (
#         "Classify the following failed household-task trajectory into exactly ONE of these error types:\n"
#         f"{', '.join(ALFWORLD_ERROR_TAXONOMY)}\n\n"
#         f"Task: {task_desc}\n\n"
#         f"Trajectory:\n{_truncate_history(history_str)}\n\n"
#         "Reply with only the error type label, nothing else."
#     )
#     raw = llm_fn(prompt).strip().upper()
#     for label in ALFWORLD_ERROR_TAXONOMY:
#         if label in raw:
#             return label
#     return "UNKNOWN"


# def build_retrieval_reflection_prompt(task_type: str,
#                                        task_desc: str,
#                                        history_str: str,
#                                        error_class: str,
#                                        retrieved: List[TrajectoryRecord]) -> str:
#     retrieved_context = format_retrieved_trajectories(retrieved)

#     current_block = (
#         "\n=== CURRENT FAILED TRAJECTORY ===\n"
#         f"Task type:  {task_type}\n"
#         f"Task:       {task_desc}\n"
#         f"Error class: {error_class}\n\n"
#         f"{_truncate_history(history_str).strip()}\n"
#     )

#     instruction = (
#         "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
#         "FAILED_STEP: <the action where things went wrong>\n"
#         "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
#         "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
#         "e.g. 'go to fridge 1' instead of 'go to countertop 1'>\n"
#         "GENERALISATION: <one sentence on when this fix applies beyond this exact task>\n"
#     )

#     return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_block + instruction


"""
ALFWorld Agents — Quantum Density Matrix Retrieval.

Strategies:
    NONE                          -> ReAct only (base run)
    REFLEXION                     -> ReAct + Reflexion (standard)
    EXPERT_CONTEXT                -> CoT + Expert context
    RETRIEVED_TRAJECTORY_REFLEXION -> Quantum density matrix retrieval
"""

import re
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Strategy enum (unchanged)
# ---------------------------------------------------------------------------

class ReflexionStrategy(Enum):
    NONE                           = 'base'
    REFLEXION                      = 'reflexion'
    EXPERT_CONTEXT                 = 'expert_context'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# ---------------------------------------------------------------------------
# ALFWorld error taxonomy (unchanged)
# ---------------------------------------------------------------------------

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

# Structurally similar task types for cross-type retrieval bonus
TASK_TYPE_STRUCTURAL_SIMILARITY = {
    'pick_clean_then_place': ['pick_heat_then_place', 'pick_cool_then_place'],
    'pick_heat_then_place':  ['pick_clean_then_place', 'pick_cool_then_place'],
    'pick_cool_then_place':  ['pick_heat_then_place', 'pick_clean_then_place'],
    'pick_and_place':        ['pick_two_obj'],
    'pick_two_obj':          ['pick_and_place'],
    'look_at_obj':           [],
}


# ---------------------------------------------------------------------------
# Quantum helpers
# ---------------------------------------------------------------------------

def _build_density_matrix(vecs: List[np.ndarray],
                           weights: List[float]) -> np.ndarray:
    """
    Build a density matrix ρ = Σ_k w_k |φ_k><φ_k| from aspect vectors.

    For ALFWorld, the 3 aspects are:
        φ_0 = task description embedding   (what the task is)
        φ_1 = error/reflection embedding   (how it failed)
        φ_2 = action history embedding     (what actions were taken)

    The density matrix superimposes all 3 aspects, capturing that a
    trajectory simultaneously encodes task identity, failure mode, and
    action pattern — rather than collapsing all into one vector.

    Returns:
        ρ: (d, d) density matrix, trace = 1
    """
    assert len(vecs) == len(weights)
    d = vecs[0].shape[0]
    rho = np.zeros((d, d), dtype=np.float64)
    for w, v in zip(weights, vecs):
        rho += w * np.outer(v, v)
    tr = np.trace(rho)
    if tr > 1e-10:
        rho /= tr
    return rho


def _quantum_similarity(rho_q: np.ndarray, rho_i: np.ndarray) -> float:
    """
    Von Neumann trace overlap: sim(ρ_q, ρ_i) = Tr(ρ_q · ρ_i)

    For pure states reduces to squared cosine similarity.
    For mixed states captures uncertainty over multiple semantic aspects.
    Range: [0, 1].
    """
    return float(np.trace(rho_q @ rho_i))


# ---------------------------------------------------------------------------
# TrajectoryRecord — quantum density matrix representation
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    """Stores a single ALFWorld episode with quantum density matrix."""

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
        # Cached aspect embeddings
        self._task_emb:   Optional[np.ndarray] = None
        self._error_emb:  Optional[np.ndarray] = None
        self._action_emb: Optional[np.ndarray] = None
        # Cached density matrix
        self._density_matrix: Optional[np.ndarray] = None

    def _get_error_text(self) -> str:
        """Text for failure mode aspect."""
        parts = []
        if self.error_class and self.error_class not in ("UNKNOWN", "SUCCESS"):
            parts.append(self.error_class)
        if self.reflection:
            parts.append(self.reflection[:200])
        return ' '.join(parts) if parts else self.task_type

    def _get_action_text(self) -> str:
        """Text for action pattern aspect — first 300 chars of history."""
        return self.history_str[:300] if self.history_str else self.task_desc

    def density_matrix(self, embed_fn,
                       w_task: float = 0.5,
                       w_error: float = 0.3,
                       w_action: float = 0.2) -> np.ndarray:
        """
        Returns ρ = w_t |task><task| + w_e |error><error| + w_a |action><action|
        Cached after first computation.
        """
        if self._density_matrix is not None:
            return self._density_matrix

        if self._task_emb is None:
            self._task_emb = embed_fn(self.task_desc)
        if self._error_emb is None:
            self._error_emb = embed_fn(self._get_error_text())
        if self._action_emb is None:
            self._action_emb = embed_fn(self._get_action_text())

        self._density_matrix = _build_density_matrix(
            vecs=[self._task_emb, self._error_emb, self._action_emb],
            weights=[w_task, w_error, w_action]
        )
        return self._density_matrix

    # Backward-compatible
    def embedding(self, embed_fn) -> np.ndarray:
        if self._task_emb is None:
            self._task_emb = embed_fn(self.task_desc)
        return self._task_emb


# ---------------------------------------------------------------------------
# TrajectoryStore — quantum density matrix episodic memory
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """
    Episodic memory with Quantum Density Matrix retrieval for ALFWorld.

    Retrieval scoring:
        1. Build query density matrix ρ_q from (task_desc, error_class, history)
        2. Score each stored trajectory via Tr(ρ_q · ρ_i)
        3. Add task-type structural similarity bonus (same type = 1.0,
           structurally similar = 0.5, different = 0.0)
        4. Apply MMR with quantum similarity for diversity
        5. Balance failures and successes in output

    Quality filter: only store reflections with len >= min_reflection_length.
    """

    _st_model = None

    def __init__(self,
                 embed_fn=None,
                 # Density matrix aspect weights (must sum to 1)
                 w_task:   float = 0.5,
                 w_error:  float = 0.3,
                 w_action: float = 0.2,
                 # Task type structural similarity bonus weight
                 lambda_type: float = 0.3,
                 # Quantum similarity weight (after type bonus)
                 lambda_quantum: float = 0.7,
                 # MMR trade-off
                 mmr_lambda: float = 0.5,
                 # Quality filter
                 min_reflection_length: int = 50):

        assert abs(w_task + w_error + w_action - 1.0) < 1e-6, \
            "Density matrix weights must sum to 1"

        self.records:    List[TrajectoryRecord] = []
        self.embed_fn    = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.w_task      = w_task
        self.w_error     = w_error
        self.w_action    = w_action
        self.lambda_type = lambda_type
        self.lambda_quantum = lambda_quantum
        self.mmr_lambda  = mmr_lambda
        self.min_reflection_length = min_reflection_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: TrajectoryRecord) -> None:
        """Add with quality filter — successes always stored, failures filtered."""
        if record.success:
            self.records.append(record)
        elif (record.reflection and
              len(record.reflection) >= self.min_reflection_length):
            self.records.append(record)
        else:
            print(f"  [Store] Skipped low-quality reflection "
                  f"(len={len(record.reflection) if record.reflection else 0})")

    def retrieve(self,
                 task_type:   str,
                 task_desc:   str,
                 error_class: str,
                 history_str: str = "",
                 k:               int = 5,
                 max_failures:    int = 3,
                 max_successes:   int = 2) -> List[TrajectoryRecord]:
        """
        Quantum retrieval with task-type structural similarity bonus.

        score = λ_type * type_sim(task_type_i, task_type_curr)
              + λ_quantum * Tr(ρ_query · ρ_i)

        where type_sim = 1.0 (same), 0.5 (structurally similar), 0.0 (different)
        """
        if not self.records:
            return []

        # Build query density matrix from current context
        rho_query = self._build_query_density_matrix(
            task_desc, error_class, history_str
        )

        scored: List[Tuple[float, TrajectoryRecord]] = []
        for rec in self.records:
            # Task type structural similarity
            if rec.task_type == task_type:
                type_sim = 1.0
            elif rec.task_type in TASK_TYPE_STRUCTURAL_SIMILARITY.get(task_type, []):
                type_sim = 0.5   # structurally similar task types
            else:
                type_sim = 0.0

            # Quantum trace overlap
            rho_i = rec.density_matrix(
                self.embed_fn, self.w_task, self.w_error, self.w_action
            )
            q_sim = _quantum_similarity(rho_query, rho_i)

            # Combined score
            score = self.lambda_type * type_sim + self.lambda_quantum * q_sim
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

    def _build_query_density_matrix(self,
                                     task_desc: str,
                                     error_class: str,
                                     history_str: str) -> np.ndarray:
        """Build query density matrix from current task context."""
        error_text  = error_class if error_class not in ("UNKNOWN", "") else task_desc
        action_text = history_str[:300] if history_str else task_desc

        task_emb   = self.embed_fn(task_desc)
        error_emb  = self.embed_fn(error_text)
        action_emb = self.embed_fn(action_text)

        return _build_density_matrix(
            vecs=[task_emb, error_emb, action_emb],
            weights=[self.w_task, self.w_error, self.w_action]
        )

    def _mmr_select(self,
                    scored: List[Tuple[float, TrajectoryRecord]],
                    k: int) -> List[TrajectoryRecord]:
        """Greedy MMR using quantum similarity for diversity."""
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
                sel_matrices = [
                    r.density_matrix(self.embed_fn, self.w_task, self.w_error, self.w_action)
                    for r in selected
                ]
                for rel_score, rec in candidates:
                    rho_i = rec.density_matrix(
                        self.embed_fn, self.w_task, self.w_error, self.w_action
                    )
                    max_sim = max(
                        _quantum_similarity(rho_i, rho_s)
                        for rho_s in sel_matrices
                    )
                    mmr_score = (self.mmr_lambda * rel_score
                                 - (1 - self.mmr_lambda) * max_sim)
                    if mmr_score > best_score:
                        best_score = mmr_score
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

    # instruction = (
    #     "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
    #     "FAILED_STEP: <the action where things went wrong>\n"
    #     "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
    #     "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
    #     "e.g. 'go to fridge 1' instead of 'go to countertop 1'>\n"
    #     "FULL_PLAN: <complete step-by-step plan for the entire next trial>\n"
    #     "GENERALISATION: <one sentence on when this fix applies beyond this exact task>\n"
    # )

    instruction = (
    "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
    "FAILED_STEP: <the action where things went wrong>\n"
    "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
    "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
    "e.g. 'go to fridge 1' instead of 'go to countertop 1'>\n"
    "GENERALISATION: <one sentence on when this fix applies beyond this exact task>\n"
)

    return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_block + instruction