# """
# Programming Agents — mirrors HotpotQA and ALFWorld agents structure.

# Strategies:
#     SIMPLE                         -> Experiment 1: Simple generation (baseline)
#     COT_GT                         -> Experiment 2: CoT + docstring context
#     REFLEXION                      -> Experiment 3: Standard Reflexion
#     RETRIEVED_TRAJECTORY_REFLEXION -> Experiment 4: Retrieval-augmented Reflexion
# """

# import re
# import difflib
# import numpy as np
# from enum import Enum
# from typing import List, Tuple, Optional


# # ---------------------------------------------------------------------------
# # Strategy enum
# # ---------------------------------------------------------------------------

# class ReflexionStrategy(Enum):
#     """
#     SIMPLE:                         No reflection, single generation attempt
#     COT_GT:                         CoT with explicit docstring context injection
#     REFLEXION:                      Standard reflexion with self-reflection
#     RETRIEVED_TRAJECTORY_REFLEXION: Retrieve top-k similar past failed problems,
#                                     use as contrastive context for reflection
#     """
#     SIMPLE                         = 'simple'
#     COT_GT                         = 'cot_gt'
#     REFLEXION                      = 'reflexion'
#     RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# # ---------------------------------------------------------------------------
# # Programming error taxonomy
# # ---------------------------------------------------------------------------

# PROGRAMMING_ERROR_TAXONOMY = [
#     "WRONG_ALGORITHM",       # fundamentally wrong approach
#     "EDGE_CASE_MISSING",     # fails on empty input, zero, negative numbers etc.
#     "OFF_BY_ONE",            # index/boundary error
#     "WRONG_DATA_STRUCTURE",  # used wrong data structure
#     "LOGIC_ERROR",           # correct approach but wrong logic
#     "TYPE_ERROR",            # wrong type handling
#     "TIMEOUT",               # correct but too slow
#     "SYNTAX_ERROR",          # code doesn't parse
#     "UNKNOWN",
# ]


# # ---------------------------------------------------------------------------
# # CoT + GT context helpers (Experiment 2)
# # ---------------------------------------------------------------------------

# PY_COT_GT_INSTRUCTION = """You are an AI Python assistant. You will be given a function signature and docstring.
# Before writing code, reason step by step about:
# 1. What the function needs to do based on the docstring
# 2. What edge cases need to be handled
# 3. What algorithm or data structure is most appropriate

# Then write your full implementation (restate the function signature).
# Use a Python code block to write your response. For example:
# ```python
# print('Hello world!')
# ```"""

# def build_cot_gt_prompt(func_sig: str) -> str:
#     """
#     Build a CoT+GT prompt by extracting the docstring as structured context
#     and prepending explicit reasoning instructions.

#     Analogous to HotpotQA CoT+GT where ground truth passage is given —
#     here the docstring IS the ground truth specification.
#     """
#     # Extract docstring if present
#     docstring = ""
#     match = re.search(r'"""(.*?)"""', func_sig, re.DOTALL)
#     if match:
#         docstring = match.group(1).strip()

#     if docstring:
#         context = (
#             f"Function specification (ground truth):\n{docstring}\n\n"
#             f"Use this specification to reason step by step before implementing."
#         )
#         return f"{context}\n\n{func_sig}"
#     return func_sig


# # ---------------------------------------------------------------------------
# # TrajectoryRecord + TrajectoryStore (Programming-adapted)
# # ---------------------------------------------------------------------------

# class TrajectoryRecord:
#     """Stores a single programming episode for later retrieval."""

#     def __init__(self,
#                  func_sig: str,
#                  implementation: str,
#                  feedback: str,
#                  reflection: str,
#                  success: bool,
#                  error_class: str = "UNKNOWN"):
#         self.func_sig       = func_sig        # function signature + docstring
#         self.implementation = implementation  # the generated code
#         self.feedback       = feedback        # test feedback
#         self.reflection     = reflection      # self-reflection text
#         self.success        = success
#         self.error_class    = error_class

#     def embedding_text(self) -> str:
#         """Text used for similarity — function signature is the key signal."""
#         return self.func_sig


# class TrajectoryStore:
#     """
#     Episodic memory of past programming trajectories.

#     Retrieval scoring:
#         score = λ_q   * sim(func_sig_i, func_sig_curr)  # semantic similarity
#               + λ_err * (error_class_i == error_class_curr)  # same failure mode

#     Uses edit distance similarity — no heavy dependencies.
#     MMR applied for diversity, balanced between failures and successes.
#     """

#     def __init__(self,
#                  lambda_q:   float = 0.7,
#                  lambda_err: float = 0.3,
#                  mmr_lambda: float = 0.5):
#         self.records:    List[TrajectoryRecord] = []
#         self.lambda_q    = lambda_q
#         self.lambda_err  = lambda_err
#         self.mmr_lambda  = mmr_lambda

#     def add(self, record: TrajectoryRecord) -> None:
#         self.records.append(record)

#     def retrieve(self,
#                  func_sig:   str,
#                  error_class: str,
#                  k:               int = 3,
#                  max_failures:    int = 2,
#                  max_successes:   int = 1) -> List[TrajectoryRecord]:
#         if not self.records:
#             return []

#         scored: List[Tuple[float, TrajectoryRecord]] = []
#         for rec in self.records:
#             sim_q   = self._text_similarity(func_sig, rec.func_sig)
#             sim_err = float(rec.error_class == error_class)
#             score   = self.lambda_q * sim_q + self.lambda_err * sim_err
#             scored.append((score, rec))

#         scored.sort(key=lambda x: x[0], reverse=True)

#         failures  = [(s, r) for s, r in scored if not r.success]
#         successes = [(s, r) for s, r in scored if r.success]

#         selected_failures  = self._mmr_select(failures,  max_failures)
#         selected_successes = self._mmr_select(successes, max_successes)

#         return selected_failures + selected_successes

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
#                 for rel_score, rec in candidates:
#                     max_sim = max(
#                         self._text_similarity(rec.func_sig, s.func_sig)
#                         for s in selected
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
#     def _text_similarity(a: str, b: str) -> float:
#         return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# # ---------------------------------------------------------------------------
# # Prompt helpers for retrieval-augmented reflection
# # ---------------------------------------------------------------------------

# RETRIEVAL_REFLECTION_HEADER = """\
# You are an expert Python programming assistant analyzing why a function implementation failed.
# Below you will find:
#   1. SIMILAR PAST TRAJECTORIES (labelled as success or failure).
#   2. The CURRENT FAILED IMPLEMENTATION.

# Study similarities and differences. Use successful implementations as reference.
# Use failed ones to identify recurring mistakes.
# Write a concise, actionable reflection for the CURRENT implementation ONLY.

# """

# def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
#     if not records:
#         return ''
#     lines = []
#     successes = [r for r in records if r.success]
#     failures  = [r for r in records if not r.success]

#     if successes:
#         lines.append("=== SIMILAR SUCCESSFUL IMPLEMENTATIONS ===")
#         for i, r in enumerate(successes, 1):
#             lines.append(f"\n[Success {i}]")
#             lines.append(f"Function: {r.func_sig[:200].strip()}")
#             lines.append(f"Implementation:\n{r.implementation[:500].strip()}")

#     if failures:
#         lines.append("\n=== SIMILAR FAILED IMPLEMENTATIONS ===")
#         for i, r in enumerate(failures, 1):
#             lines.append(f"\n[Failure {i}] Error class: {r.error_class}")
#             lines.append(f"Function: {r.func_sig[:200].strip()}")
#             lines.append(f"Implementation:\n{r.implementation[:500].strip()}")
#             lines.append(f"Feedback:\n{r.feedback[:300].strip()}")
#             if r.reflection:
#                 lines.append(f"Reflection: {r.reflection}")

#     return '\n'.join(lines)


# def classify_programming_error(func_sig: str,
#                                 implementation: str,
#                                 feedback: str,
#                                 llm_fn) -> str:
#     """
#     Ask the LLM to classify the programming failure into one of the known error types.
#     llm_fn: callable(prompt: str) -> str
#     """
#     prompt = (
#         "Classify the following failed Python implementation into exactly ONE of these error types:\n"
#         f"{', '.join(PROGRAMMING_ERROR_TAXONOMY)}\n\n"
#         f"Function signature:\n{func_sig[:300]}\n\n"
#         f"Failed implementation:\n{implementation[:500]}\n\n"
#         f"Test feedback:\n{feedback[:300]}\n\n"
#         "Reply with only the error type label, nothing else."
#     )
#     raw = llm_fn(prompt).strip().upper()
#     for label in PROGRAMMING_ERROR_TAXONOMY:
#         if label in raw:
#             return label
#     return "UNKNOWN"


# def build_retrieval_reflection_prompt(func_sig: str,
#                                        implementation: str,
#                                        feedback: str,
#                                        error_class: str,
#                                        retrieved: List[TrajectoryRecord]) -> str:
#     retrieved_context = format_retrieved_trajectories(retrieved)

#     current_block = (
#         "\n=== CURRENT FAILED IMPLEMENTATION ===\n"
#         f"Error class: {error_class}\n\n"
#         f"Function:\n{func_sig[:300].strip()}\n\n"
#         f"Implementation:\n{implementation[:500].strip()}\n\n"
#         f"Test feedback:\n{feedback[:300].strip()}\n"
#     )

#     instruction = (
#         "\nWrite a reflection for the CURRENT FAILED IMPLEMENTATION in EXACTLY this format:\n\n"
#         "FAILED_STEP: <what specifically went wrong in the implementation>\n"
#         "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
#         "WHAT_TO_DO_DIFFERENTLY: <exact change to make in the next implementation>\n"
#         "GENERALISATION: <one sentence on when this fix applies beyond this exact problem>\n"
#     )

#     return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_block + instruction

"""
Programming Agents — Quantum Density Matrix Retrieval.

Strategies:
    SIMPLE                         -> Experiment 1: Simple generation (baseline)
    COT_GT                         -> Experiment 2: CoT + docstring context
    REFLEXION                      -> Experiment 3: Standard Reflexion
    RETRIEVED_TRAJECTORY_REFLEXION -> Experiment 4: Quantum density matrix retrieval
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
# Quantum helpers
# ---------------------------------------------------------------------------

def _text_to_vector(text: str, vocab: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert text to a normalised bag-of-words vector.
    If vocab is None, builds from the text itself (used for single-text embedding).
    Used as a lightweight alternative to sentence transformers for programming tasks
    where token overlap is highly informative (function names, keywords, types).
    """
    tokens = re.findall(r'\w+', text.lower())
    if not tokens:
        tokens = ['<empty>']
    if vocab is None:
        vocab = sorted(set(tokens))
    vec = np.array([tokens.count(w) for w in vocab], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def _build_shared_vocab(*texts: str) -> List[str]:
    """Build a shared vocabulary from multiple texts for comparable vectors."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(re.findall(r'\w+', text.lower()))
    return sorted(set(all_tokens)) if all_tokens else ['<empty>']


def _build_density_matrix(vecs: List[np.ndarray],
                           weights: List[float]) -> np.ndarray:
    """
    Build density matrix ρ = Σ_k w_k |φ_k><φ_k|

    For programming tasks, the 3 aspects are:
        φ_0 = function signature embedding  (what the function does)
        φ_1 = error + feedback embedding    (how it failed)
        φ_2 = implementation embedding      (what code was written)

    Returns: (d, d) density matrix, trace = 1
    """
    assert len(vecs) == len(weights)
    # Pad all vectors to same dimension
    max_dim = max(v.shape[0] for v in vecs)
    padded = []
    for v in vecs:
        if v.shape[0] < max_dim:
            v = np.pad(v, (0, max_dim - v.shape[0]))
        # Re-normalize after padding
        norm = np.linalg.norm(v)
        padded.append(v / norm if norm > 1e-10 else v)

    d = max_dim
    rho = np.zeros((d, d), dtype=np.float64)
    for w, v in zip(weights, padded):
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
    # Handle dimension mismatch — pad smaller matrix
    d_q, d_i = rho_q.shape[0], rho_i.shape[0]
    if d_q != d_i:
        d = max(d_q, d_i)
        if d_q < d:
            rho_q = np.pad(rho_q, ((0, d - d_q), (0, d - d_q)))
        else:
            rho_i = np.pad(rho_i, ((0, d - d_i), (0, d - d_i)))
    return float(np.trace(rho_q @ rho_i))


# ---------------------------------------------------------------------------
# TrajectoryRecord — quantum density matrix representation
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    """Stores a single programming episode with quantum density matrix."""

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
        # Cached density matrix
        self._density_matrix: Optional[np.ndarray] = None

    def _get_error_text(self) -> str:
        """Text for failure mode aspect."""
        parts = []
        if self.error_class and self.error_class not in ("UNKNOWN", "SUCCESS"):
            parts.append(self.error_class)
        if self.feedback:
            parts.append(self.feedback[:200])
        if self.reflection:
            parts.append(self.reflection[:100])
        return ' '.join(parts) if parts else self.error_class

    def density_matrix(self,
                       w_sig: float = 0.5,
                       w_error: float = 0.3,
                       w_impl: float = 0.2) -> np.ndarray:
        """
        Returns ρ = w_s |sig><sig| + w_e |error><error| + w_i |impl><impl|

        Uses shared vocabulary across all 3 aspects for comparable vectors.
        Cached after first computation.
        """
        if self._density_matrix is not None:
            return self._density_matrix

        sig_text   = self.func_sig[:500]
        error_text = self._get_error_text()
        impl_text  = self.implementation[:500]

        # Shared vocabulary ensures vectors are in same space
        vocab = _build_shared_vocab(sig_text, error_text, impl_text)

        sig_vec   = _text_to_vector(sig_text,   vocab)
        error_vec = _text_to_vector(error_text, vocab)
        impl_vec  = _text_to_vector(impl_text,  vocab)

        self._density_matrix = _build_density_matrix(
            vecs=[sig_vec, error_vec, impl_vec],
            weights=[w_sig, w_error, w_impl]
        )
        return self._density_matrix

    def embedding_text(self) -> str:
        """Backward-compatible."""
        return self.func_sig


# ---------------------------------------------------------------------------
# TrajectoryStore — quantum density matrix episodic memory
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """
    Episodic memory with Quantum Density Matrix retrieval for programming tasks.

    Retrieval:
        1. Build query density matrix from (func_sig, error_class+feedback, implementation)
        2. Score each stored trajectory via Tr(ρ_query · ρ_i)
        3. Apply MMR with quantum similarity for diversity
        4. Balance failures and successes

    Quality filter: only store reflections with len >= min_reflection_length.

    Note: Uses bag-of-words vectors (no sentence transformers) — appropriate for
    programming tasks where token overlap (function names, keywords, types) is
    highly informative and lightweight computation is preferred.
    """

    def __init__(self,
                 # Density matrix aspect weights (must sum to 1)
                 w_sig:   float = 0.5,
                 w_error: float = 0.3,
                 w_impl:  float = 0.2,
                 # MMR trade-off
                 mmr_lambda: float = 0.5,
                 # Quality filter
                 min_reflection_length: int = 50):

        assert abs(w_sig + w_error + w_impl - 1.0) < 1e-6, \
            "Density matrix weights must sum to 1"

        self.records:    List[TrajectoryRecord] = []
        self.w_sig       = w_sig
        self.w_error     = w_error
        self.w_impl      = w_impl
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
                 func_sig:    str,
                 error_class: str,
                 feedback:    str = "",
                 implementation: str = "",
                 k:               int = 3,
                 max_failures:    int = 2,
                 max_successes:   int = 1) -> List[TrajectoryRecord]:
        """
        Quantum retrieval using Von Neumann trace overlap.

        Query density matrix built from:
            φ_0 = func_sig embedding        (what function does)
            φ_1 = error_class + feedback     (how it failed)
            φ_2 = implementation embedding   (what code was attempted)
        """
        if not self.records:
            return []

        # Build query density matrix
        rho_query = self._build_query_density_matrix(
            func_sig, error_class, feedback, implementation
        )

        scored: List[Tuple[float, TrajectoryRecord]] = []
        for rec in self.records:
            rho_i = rec.density_matrix(self.w_sig, self.w_error, self.w_impl)
            score = _quantum_similarity(rho_query, rho_i)
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
                                     func_sig: str,
                                     error_class: str,
                                     feedback: str,
                                     implementation: str) -> np.ndarray:
        """Build query density matrix from current context."""
        sig_text   = func_sig[:500]
        error_text = f"{error_class} {feedback[:200]}" if feedback else error_class
        impl_text  = implementation[:500] if implementation else func_sig[:200]

        vocab = _build_shared_vocab(sig_text, error_text, impl_text)

        sig_vec   = _text_to_vector(sig_text,   vocab)
        error_vec = _text_to_vector(error_text, vocab)
        impl_vec  = _text_to_vector(impl_text,  vocab)

        return _build_density_matrix(
            vecs=[sig_vec, error_vec, impl_vec],
            weights=[self.w_sig, self.w_error, self.w_impl]
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
                    r.density_matrix(self.w_sig, self.w_error, self.w_impl)
                    for r in selected
                ]
                for rel_score, rec in candidates:
                    rho_i = rec.density_matrix(self.w_sig, self.w_error, self.w_impl)
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