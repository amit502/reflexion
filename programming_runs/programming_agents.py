"""
Programming Agents — mirrors HotpotQA and ALFWorld agents structure.

Strategies:
    SIMPLE                         -> Experiment 1: Simple generation (baseline)
    COT_GT                         -> Experiment 2: CoT + docstring context
    REFLEXION                      -> Experiment 3: Standard Reflexion
    RETRIEVED_TRAJECTORY_REFLEXION -> Experiment 4: Retrieval-augmented Reflexion
"""

import re
import difflib
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class ReflexionStrategy(Enum):
    """
    SIMPLE:                         No reflection, single generation attempt
    COT_GT:                         CoT with explicit docstring context injection
    REFLEXION:                      Standard reflexion with self-reflection
    RETRIEVED_TRAJECTORY_REFLEXION: Retrieve top-k similar past failed problems,
                                    use as contrastive context for reflection
    """
    SIMPLE                         = 'simple'
    COT_GT                         = 'cot_gt'
    REFLEXION                      = 'reflexion'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# ---------------------------------------------------------------------------
# Programming error taxonomy
# ---------------------------------------------------------------------------

PROGRAMMING_ERROR_TAXONOMY = [
    "WRONG_ALGORITHM",       # fundamentally wrong approach
    "EDGE_CASE_MISSING",     # fails on empty input, zero, negative numbers etc.
    "OFF_BY_ONE",            # index/boundary error
    "WRONG_DATA_STRUCTURE",  # used wrong data structure
    "LOGIC_ERROR",           # correct approach but wrong logic
    "TYPE_ERROR",            # wrong type handling
    "TIMEOUT",               # correct but too slow
    "SYNTAX_ERROR",          # code doesn't parse
    "UNKNOWN",
]


# ---------------------------------------------------------------------------
# CoT + GT context helpers (Experiment 2)
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
    """
    Build a CoT+GT prompt by extracting the docstring as structured context
    and prepending explicit reasoning instructions.

    Analogous to HotpotQA CoT+GT where ground truth passage is given —
    here the docstring IS the ground truth specification.
    """
    # Extract docstring if present
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
# TrajectoryRecord + TrajectoryStore (Programming-adapted)
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
        self.func_sig       = func_sig        # function signature + docstring
        self.implementation = implementation  # the generated code
        self.feedback       = feedback        # test feedback
        self.reflection     = reflection      # self-reflection text
        self.success        = success
        self.error_class    = error_class

    def embedding_text(self) -> str:
        """Text used for similarity — function signature is the key signal."""
        return self.func_sig


class TrajectoryStore:
    """
    Episodic memory of past programming trajectories.

    Retrieval scoring:
        score = λ_q   * sim(func_sig_i, func_sig_curr)  # semantic similarity
              + λ_err * (error_class_i == error_class_curr)  # same failure mode

    Uses edit distance similarity — no heavy dependencies.
    MMR applied for diversity, balanced between failures and successes.
    """

    def __init__(self,
                 lambda_q:   float = 0.7,
                 lambda_err: float = 0.3,
                 mmr_lambda: float = 0.5):
        self.records:    List[TrajectoryRecord] = []
        self.lambda_q    = lambda_q
        self.lambda_err  = lambda_err
        self.mmr_lambda  = mmr_lambda

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self,
                 func_sig:   str,
                 error_class: str,
                 k:               int = 3,
                 max_failures:    int = 2,
                 max_successes:   int = 1) -> List[TrajectoryRecord]:
        if not self.records:
            return []

        scored: List[Tuple[float, TrajectoryRecord]] = []
        for rec in self.records:
            sim_q   = self._text_similarity(func_sig, rec.func_sig)
            sim_err = float(rec.error_class == error_class)
            score   = self.lambda_q * sim_q + self.lambda_err * sim_err
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        failures  = [(s, r) for s, r in scored if not r.success]
        successes = [(s, r) for s, r in scored if r.success]

        selected_failures  = self._mmr_select(failures,  max_failures)
        selected_successes = self._mmr_select(successes, max_successes)

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
                for rel_score, rec in candidates:
                    max_sim = max(
                        self._text_similarity(rec.func_sig, s.func_sig)
                        for s in selected
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
    def _text_similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Prompt helpers for retrieval-augmented reflection
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
    """
    Ask the LLM to classify the programming failure into one of the known error types.
    llm_fn: callable(prompt: str) -> str
    """
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