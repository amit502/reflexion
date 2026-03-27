"""
ExpeL Experience Pool — shared across all 3 tasks.

Stores successful and failed trajectories from a training/gathering phase.
After gathering, extracts generalised insights via batch LLM call.
At inference, retrieves top-k successful trajectories by cosine similarity
and injects insights + successes into the agent prompt.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict


class ExperienceRecord:
    """A single trajectory record in the ExpeL experience pool."""

    def __init__(self,
                 task_id: str,          # unique identifier for the task
                 task_desc: str,        # question / task description / func_sig
                 trajectory: str,       # full action-observation trace
                 success: bool,
                 answer: str = ""):     # correct answer if known
        self.task_id    = task_id
        self.task_desc  = task_desc
        self.trajectory = trajectory
        self.success    = success
        self.answer     = answer
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.task_desc)
        return self._embedding


class ExpeL:
    """
    ExpeL experience pool.

    Two phases:
        Phase 1 — gather(): called after each trial during experience gathering
        Phase 2 — extract_insights(): called once after gathering is complete
                   inference(): called at eval time for each new task

    At inference:
        1. Retrieve top-k successful trajectories by cosine similarity
        2. Inject insights + retrieved successes into prompt
        3. Single attempt — no reflection
    """

    _st_model = None

    def __init__(self,
                 embed_fn=None,
                 max_insights: int = 10,
                 retrieval_k: int = 3):
        self.pool:          List[ExperienceRecord] = []
        self.insights:      List[str] = []          # extracted after gathering
        self.embed_fn       = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.max_insights   = max_insights
        self.retrieval_k    = retrieval_k

    # ------------------------------------------------------------------
    # Phase 1 — Gathering
    # ------------------------------------------------------------------

    def add(self, record: ExperienceRecord) -> None:
        """Add a trajectory to the pool during gathering phase."""
        self.pool.append(record)

    # ------------------------------------------------------------------
    # Phase 2 — Insight Extraction (call once after gathering)
    # ------------------------------------------------------------------

    def extract_insights(self, llm_fn) -> List[str]:
        """
        Batch insight extraction — compare successes vs failures and
        extract generalised rules that explain what distinguishes them.

        llm_fn: callable(prompt: str) -> str
        """
        if not self.pool:
            return []

        successes = [r for r in self.pool if r.success]
        failures  = [r for r in self.pool if not r.success]

        if not successes:
            print("ExpeL: No successful trajectories to extract insights from.")
            return []

        # Format a sample of successes and failures for the LLM
        success_block = "\n\n".join([
            f"[Success] Task: {r.task_desc[:200]}\nTrajectory: {r.trajectory[:500]}"
            for r in successes[:5]
        ])
        failure_block = "\n\n".join([
            f"[Failure] Task: {r.task_desc[:200]}\nTrajectory: {r.trajectory[:500]}"
            for r in failures[:5]
        ])

        prompt = (
            "You are analyzing trajectories of an AI agent to extract generalised lessons.\n\n"
            "=== SUCCESSFUL TRAJECTORIES ===\n"
            f"{success_block}\n\n"
            "=== FAILED TRAJECTORIES ===\n"
            f"{failure_block}\n\n"
            f"Extract exactly {self.max_insights} generalised insights that explain "
            "what distinguishes successful from failed trajectories. "
            "Each insight should be a single actionable rule that applies broadly, "
            "not just to these specific examples.\n\n"
            "Format your response as a numbered list:\n"
            "1. <insight>\n"
            "2. <insight>\n"
            "..."
        )

        raw = llm_fn(prompt)
        insights = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            # Parse numbered list items
            if line and line[0].isdigit() and '.' in line:
                insight = line.split('.', 1)[-1].strip()
                if len(insight) > 20:
                    insights.append(insight)

        self.insights = insights[:self.max_insights]
        print(f"ExpeL: Extracted {len(self.insights)} insights.")
        return self.insights

    # ------------------------------------------------------------------
    # Phase 2 — Inference Retrieval
    # ------------------------------------------------------------------

    def retrieve_successes(self, task_desc: str) -> List[ExperienceRecord]:
        """
        Retrieve top-k successful trajectories most similar to task_desc.
        Used at inference time.
        """
        successes = [r for r in self.pool if r.success]
        if not successes:
            return []

        q_emb = self.embed_fn(task_desc)
        scored = []
        for rec in successes:
            sim = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:self.retrieval_k]]

    def format_inference_context(self, task_desc: str) -> str:
        """
        Build the full ExpeL context block to inject at inference:
            [Extracted insights] + [Retrieved successful trajectories]
        """
        retrieved = self.retrieve_successes(task_desc)
        lines = []

        # Insights block
        if self.insights:
            lines.append("=== GENERALISED INSIGHTS FROM PAST EXPERIENCE ===")
            for i, ins in enumerate(self.insights, 1):
                lines.append(f"{i}. {ins}")
            lines.append("")

        # Retrieved successes block
        if retrieved:
            lines.append("=== SIMILAR SUCCESSFUL PAST TRAJECTORIES ===")
            for i, rec in enumerate(retrieved, 1):
                lines.append(f"\n[Example {i}] Task: {rec.task_desc[:200]}")
                lines.append(rec.trajectory[:800].strip())

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Sentence transformer embedding (same as other agents)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_st_model():
        if ExpeL._st_model is None:
            from sentence_transformers import SentenceTransformer
            ExpeL._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return ExpeL._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        model = ExpeL._get_st_model()
        vec   = model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float64)