"""
Programming Agents — mirrors HotpotQA and ALFWorld agents structure.

Strategies:
    SIMPLE                         -> Experiment 1: Simple generation (baseline)
    COT_GT                         -> Experiment 2: CoT + docstring context
    REFLEXION                      -> Experiment 3: Standard Reflexion
    RETRIEVED_TRAJECTORY_REFLEXION -> Experiment 4: Retrieval-augmented Reflexion
                                      (error-first filtering + sentence transformer)
"""

import re
import difflib
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Strategy enum (unchanged)
# ---------------------------------------------------------------------------

class ReflexionStrategy(Enum):
    SIMPLE                         = 'simple'
    COT_GT                         = 'cot_gt'
    REFLEXION                      = 'reflexion'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# ---------------------------------------------------------------------------
# Programming error taxonomy (unchanged)
# ---------------------------------------------------------------------------

PROGRAMMING_ERROR_TAXONOMY = [
    "WRONG_ALGORITHM",
    "EDGE_CASE_MISSING",
    "OFF_BY_ONE",
    "WRONG_DATA_STRUCTURE",
    "LOGIC_ERROR",
    "TYPE_ERROR",
    "TIMEOUT",
    "SYNTAX_ERROR",
    "UNKNOWN",
]


# ---------------------------------------------------------------------------
# CoT + GT context helpers (unchanged)
# ---------------------------------------------------------------------------

PY_COT_GT_INSTRUCTION = """You are an AI Python assistant. You will be given a function signature and docstring.
Before writing code, reason step by step about:
1. What the function needs to do based on the docstring
2. What edge cases need to be handled
3. What algorithm or data structure is most appropriate

Then write your full implementation (restate the function signature).
Use a Python code block to write your response. For example:
```python
print('Hello world!')
```"""


def build_cot_gt_prompt(func_sig: str) -> str:
    docstring = ""
    match = re.search(r'"""(.*?)"""', func_sig, re.DOTALL)
    if match:
        docstring = match.group(1).strip()
    if docstring:
        context = (
            f"Function specification (ground truth):\n{docstring}\n\n"
            f"Use this specification to reason step by step before implementing."
        )
        return f"{context}\n\n{func_sig}"
    return func_sig


# ---------------------------------------------------------------------------
# TrajectoryRecord (unchanged)
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    """Stores a single programming episode for later retrieval."""

    def __init__(self,
                 func_sig: str,
                 implementation: str,
                 feedback: str,
                 reflection: str,
                 success: bool,
                 error_class: str = "UNKNOWN"):
        self.func_sig       = func_sig
        self.implementation = implementation
        self.feedback       = feedback
        self.reflection     = reflection
        self.success        = success
        self.error_class    = error_class
        self._embedding: Optional[np.ndarray] = None

    def embedding_text(self) -> str:
        return self.func_sig

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.func_sig)
        return self._embedding


# ---------------------------------------------------------------------------
# TrajectoryStore — sentence transformer + error-first filtering
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """
    Episodic memory of past programming trajectories.

    Retrieval uses error-first filtering with fallback:
        Step 1 — Filter failures by error_class (hard filter)
                 Fallback to all failures if fewer than k match
        Step 2 — Rank filtered failures by sentence transformer cosine similarity
        Step 3 — Successes always retrieved separately by cosine similarity
        Step 4 — MMR applied within each pool for diversity
    """

    _st_model = None  # class-level cache

    def __init__(self, mmr_lambda: float = 0.5):
        self.records:   List[TrajectoryRecord] = []
        self.mmr_lambda = mmr_lambda
        self.embed_fn   = self._sentence_transformer_embed

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self,
                 func_sig:    str,
                 error_class: str,
                 k:               int = 3,
                 max_failures:    int = 2,
                 max_successes:   int = 1) -> List[TrajectoryRecord]:
        if not self.records:
            return []

        q_emb = self.embed_fn(func_sig)

        # ── Error-first filtering ────────────────────────────────────────────
        # Successes always retrieved separately as contrastive anchors
        successes_pool = [r for r in self.records if r.success]

        # Filter failures by error class first (strongest signal)
        failure_candidates = [
            r for r in self.records
            if not r.success and r.error_class == error_class
        ]
        # Fallback: all failures if not enough same-error matches
        if len(failure_candidates) < k:
            failure_candidates = [r for r in self.records if not r.success]
        # ─────────────────────────────────────────────────────────────────────

        # Rank filtered failures by cosine similarity
        scored_failures: List[Tuple[float, TrajectoryRecord]] = []
        for rec in failure_candidates:
            sim_q = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
            scored_failures.append((sim_q, rec))
        scored_failures.sort(key=lambda x: x[0], reverse=True)

        # Rank successes by cosine similarity
        scored_successes: List[Tuple[float, TrajectoryRecord]] = []
        for rec in successes_pool:
            sim_q = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
            scored_successes.append((sim_q, rec))
        scored_successes.sort(key=lambda x: x[0], reverse=True)

        selected_failures  = self._mmr_select(scored_failures,  max_failures)
        selected_successes = self._mmr_select(scored_successes, max_successes)

        return selected_failures + selected_successes

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

    # Keep for backward compatibility
    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Prompt helpers (unchanged)
# ---------------------------------------------------------------------------

RETRIEVAL_REFLECTION_HEADER = """\
You are an expert Python programming assistant analyzing why a function implementation failed.
Below you will find:
  1. SIMILAR PAST TRAJECTORIES (labelled as success or failure).
  2. The CURRENT FAILED IMPLEMENTATION.

Study similarities and differences. Use successful implementations as reference.
Use failed ones to identify recurring mistakes.
Write a concise, actionable reflection for the CURRENT implementation ONLY.

"""


def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
    if not records:
        return ''
    lines = []
    successes = [r for r in records if r.success]
    failures  = [r for r in records if not r.success]

    if successes:
        lines.append("=== SIMILAR SUCCESSFUL IMPLEMENTATIONS ===")
        for i, r in enumerate(successes, 1):
            lines.append(f"\n[Success {i}]")
            lines.append(f"Function: {r.func_sig[:200].strip()}")
            lines.append(f"Implementation:\n{r.implementation[:500].strip()}")

    if failures:
        lines.append("\n=== SIMILAR FAILED IMPLEMENTATIONS ===")
        for i, r in enumerate(failures, 1):
            lines.append(f"\n[Failure {i}] Error class: {r.error_class}")
            lines.append(f"Function: {r.func_sig[:200].strip()}")
            lines.append(f"Implementation:\n{r.implementation[:500].strip()}")
            lines.append(f"Feedback:\n{r.feedback[:300].strip()}")
            if r.reflection:
                lines.append(f"Reflection: {r.reflection}")

    return '\n'.join(lines)


def classify_programming_error(func_sig: str,
                                implementation: str,
                                feedback: str,
                                llm_fn) -> str:
    prompt = (
        "Classify the following failed Python implementation into exactly ONE of these error types:\n"
        f"{', '.join(PROGRAMMING_ERROR_TAXONOMY)}\n\n"
        f"Function signature:\n{func_sig[:300]}\n\n"
        f"Failed implementation:\n{implementation[:500]}\n\n"
        f"Test feedback:\n{feedback[:300]}\n\n"
        "Reply with only the error type label, nothing else."
    )
    raw = llm_fn(prompt).strip().upper()
    for label in PROGRAMMING_ERROR_TAXONOMY:
        if label in raw:
            return label
    return "UNKNOWN"


def build_retrieval_reflection_prompt(func_sig: str,
                                       implementation: str,
                                       feedback: str,
                                       error_class: str,
                                       retrieved: List[TrajectoryRecord]) -> str:
    retrieved_context = format_retrieved_trajectories(retrieved)
    current_block = (
        "\n=== CURRENT FAILED IMPLEMENTATION ===\n"
        f"Error class: {error_class}\n\n"
        f"Function:\n{func_sig[:300].strip()}\n\n"
        f"Implementation:\n{implementation[:500].strip()}\n\n"
        f"Test feedback:\n{feedback[:300].strip()}\n"
    )
    instruction = (
        "\nWrite a reflection for the CURRENT FAILED IMPLEMENTATION in EXACTLY this format:\n\n"
        "FAILED_STEP: <what specifically went wrong in the implementation>\n"
        "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
        "WHAT_TO_DO_DIFFERENTLY: <exact change to make in the next implementation>\n"
        "GENERALISATION: <one sentence on when this fix applies beyond this exact problem>\n"
    )
    return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_block + instruction