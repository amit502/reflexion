"""
STAR — Step-level Trajectory-Agnostic Retrieval

STAR = RAR + StepKnowledgeStore

RAR component (unchanged):
  - TrajectoryStore: stores full episodes with reflections
  - After failure: retrieve top-k similar trajectories + their reflections
  - Use retrieved trajectories as context to generate new reflection
  - Agent sees reflections from retrieved trajectories only (not all)

STAR addition (on top of RAR):
  - StepKnowledgeStore: stores atomic step-level rules
  - Before every action: retrieve top-k relevant rules via attention
  - LLM generates EXPECTED + NEXT_INTENT + optional CORRECTION per step
  - Zero extra LLM calls vs RAR
"""

import re
import string
from typing import List, Tuple, Optional
from enum import Enum
import numpy as np
import tiktoken
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    reflect_prompt,
    react_reflect_agent_prompt,
    REFLECTION_HEADER,
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    CLASSIFY_ERROR_SYSTEM_PROMPT,
)
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, WEBTHINK_SIMPLE2
from sentence_transformers import SentenceTransformer

# ── Import RAR components — exact names from retrieval_agents.py ─────────────
from retrieval_agents import (
    TrajectoryRecord,
    TrajectoryStore,
    classify_error,
    format_retrieved_trajectories,
    RETRIEVAL_REFLECTION_HEADER,
    format_last_attempt,
    format_reflections,
)




NONE                           = 'base'
REFLEXION                      = 'reflexion'
RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'
STAR                           = 'star'


# ---------------------------------------------------------------------------
# Step Knowledge — STAR addition
# ---------------------------------------------------------------------------

class StepKnowledge:
    """
    One generalizable rule extracted from a single step.
    positive=True  → prediction matched actual (confirmed rule)
    positive=False → prediction differed from actual (corrective rule)
    """
    def __init__(self,
                 action_intent: str,
                 rule: str,
                 positive: bool):
        self.action_intent = action_intent
        self.rule          = rule
        self.positive      = positive
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.action_intent + " " + self.rule)
        return self._embedding


class StepKnowledgeStore:
    """Attention-weighted retrieval over step-level knowledge."""

    _st_model = None

    def __init__(self, tau=0.1, adaptive_tau=True, mmr_lambda=0.5):
        self.knowledge:   List[StepKnowledge] = []
        self.embed_fn     = self._sentence_transformer_embed
        self.tau          = tau
        self.adaptive_tau = adaptive_tau
        self.mmr_lambda   = mmr_lambda

    def add(self, knowledge: StepKnowledge) -> None:
        # Deduplication: skip if very similar rule already exists
        if self.knowledge:
            q_emb = self.embed_fn(knowledge.action_intent + " " + knowledge.rule)
            for existing in self.knowledge[-20:]:
                sim = float(np.dot(q_emb, existing.embedding(self.embed_fn)))
                if sim > 0.92:
                    return
        self.knowledge.append(knowledge)

    def retrieve(self, action_intent: str, k: int = 2) -> List[StepKnowledge]:
        if not self.knowledge:
            return []
        q_emb = self.embed_fn(action_intent)
        d     = q_emb.shape[0]
        tau   = (0.05 + 0.25 * min(len(self.knowledge) / 100.0, 1.0)
                 if self.adaptive_tau else self.tau)
        logits = [
            float(np.dot(q_emb, sk.embedding(self.embed_fn))) / np.sqrt(d) / tau
            for sk in self.knowledge
        ]
        logits_arr  = np.array(logits)
        logits_arr -= logits_arr.max()
        alphas      = np.exp(logits_arr) / np.exp(logits_arr).sum()
        scored = sorted(zip(alphas.tolist(), self.knowledge),
                        key=lambda x: x[0], reverse=True)
        return self._mmr_select(scored, k)

    def _mmr_select(self, scored, k):
        if not scored:
            return []
        selected, candidates = [], list(scored)
        while len(selected) < k and candidates:
            if not selected:
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score, best = -1e9, None
                sel_embs = [s.embedding(self.embed_fn) for s in selected]
                for attn_score, sk in candidates:
                    max_sim = max(
                        float(np.dot(sk.embedding(self.embed_fn), se))
                        for se in sel_embs
                    )
                    mmr = self.mmr_lambda * attn_score - (1 - self.mmr_lambda) * max_sim
                    if mmr > best_score:
                        best_score, best = mmr, sk
            selected.append(best)
            candidates = [(a, s) for a, s in candidates if s is not best]
        return selected

    @staticmethod
    def _get_st_model():
        if StepKnowledgeStore._st_model is None:
            StepKnowledgeStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return StepKnowledgeStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        return StepKnowledgeStore._get_st_model().encode(
            text, normalize_embeddings=True).astype(np.float64)


# ---------------------------------------------------------------------------
# Structured output parser
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> dict:
    result = {
        'thought': '', 'action': '', 'expected': '',
        'next_intent': '', 'correction': '',
    }
    if not raw:
        return result
    current_key = None
    for line in raw.split('\n'):
        line = line.strip()
        matched = False
        for key in ['THOUGHT', 'ACTION', 'EXPECTED', 'NEXT_INTENT', 'CORRECTION']:
            if line.upper().startswith(f'{key}:'):
                current_key = key.lower()
                result[current_key] = line[len(key)+1:].strip()
                matched = True
                break
        if not matched and current_key and line:
            result[current_key] += ' ' + line
    for k in result:
        result[k] = result[k].strip()
    return result


def format_step_knowledge(knowledge: List[StepKnowledge]) -> str:
    if not knowledge:
        return ""
    lines = ["=== STEP KNOWLEDGE FROM PAST EXPERIENCE ==="]
    for sk in knowledge:
        icon = "CONFIRMED" if sk.positive else "CORRECTION"
        lines.append(f"[{icon}] {sk.rule}")
    lines.append("=== END STEP KNOWLEDGE ===\n")
    return '\n'.join(lines)


STAR_STEP_INSTRUCTION = (
    "\n\nNow respond using EXACTLY these labels (as shown in examples above):\n"
    "THOUGHT: <your reasoning>\n"
    "ACTION: <Search[X] or Lookup[X] or Finish[X]>\n"
    "EXPECTED: <what you expect this action to return>\n"
    "NEXT_INTENT: <what you plan to do after this, e.g. 'lookup birth year'>\n"
    "CORRECTION: <only if previous EXPECTED was wrong — one generalizable rule. "
    "Skip this line entirely if first step or prediction was accurate.>\n"
)


# ---------------------------------------------------------------------------
# STAR ReactAgent — RAR + StepKnowledgeStore
# ---------------------------------------------------------------------------

class STARReactAgent:
    """
    STAR = RAR trajectory retrieval + step-level knowledge retrieval.

    RAR component:
      - TrajectoryStore shared across all agents
      - After failure: retrieve top-k trajectories by attention
      - Use retrieved trajectories + their reflections to generate new reflection
      - Agent sees reflections from retrieved trajectories only (not all)

    STAR addition:
      - StepKnowledgeStore shared across all agents
      - Before every action: retrieve top-k relevant rules
      - One LLM call per step: THOUGHT + ACTION + EXPECTED + NEXT_INTENT + CORRECTION
      - Mismatch injection: NOTE added when prev EXPECTED != actual
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 react_llm: AnyOpenAILLM = None,
                 reflect_llm: AnyOpenAILLM = None,
                 trajectory_store: TrajectoryStore = None,   # ← RAR
                 knowledge_store: StepKnowledgeStore = None,  # ← STAR
                 retrieval_k: int = 3,
                 use_reflection: bool = True):

        self.question          = question
        self.key               = key
        self.max_steps         = max_steps
        self.llm               = react_llm  or AnyOpenAILLM()
        self.reflect_llm       = reflect_llm or AnyOpenAILLM()
        self.trajectory_store  = trajectory_store if trajectory_store is not None \
                                 else TrajectoryStore()
        self.knowledge_store   = knowledge_store if knowledge_store is not None \
                                 else StepKnowledgeStore()
        self.retrieval_k       = retrieval_k
        self.use_reflection    = use_reflection
        self.react_examples    = WEBTHINK_SIMPLE2
        self.reflect_examples  = REFLECTIONS
        self.enc               = tiktoken.encoding_for_model("text-davinci-003")
        # reflections_str shows only reflections from retrieved trajectories
        self.reflections_str   = ''
        self.docstore          = DocstoreExplorer(Wikipedia())
        self.__reset_agent()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, reset: bool = True,
            reflect_strategy: ReflexionStrategy = ReflexionStrategy.STAR) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            if self.use_reflection:
                self._reflect()
        if reset:
            self.__reset_agent()
        while not self.is_halted() and not self.is_finished():
            self.step()

    # ------------------------------------------------------------------
    # Step — one LLM call does everything
    # ------------------------------------------------------------------

    def step(self) -> None:
        # 1. Use pre-fetched step knowledge from previous step's NEXT_INTENT
        knowledge_str = format_step_knowledge(self._prefetched_knowledge)

        # 2. One LLM call
        prompt = self._build_agent_prompt(knowledge_str)
        raw    = self.llm(prompt, REACT_SYSTEM_PROMPT) or ''
        parsed = parse_structured_response(raw)

        thought     = parsed['thought']
        action_str  = parsed['action']
        expected    = parsed['expected']
        next_intent = parsed['next_intent']
        correction  = parsed['correction']

        # Fallback if structured parsing failed entirely
        if not thought and not action_str:
            print('  [STAR] Structured parse failed — falling back to ReAct format')
            for line in raw.split('\n'):
                line = line.strip()
                if not thought and re.match(r'Thought\s*\d*\s*:', line, re.IGNORECASE):
                    thought = line.split(':', 1)[-1].strip()
                if not action_str and any(a in line for a in ['Search[', 'Lookup[', 'Finish[']):
                    action_str = line.strip()

        # 3. Update scratchpad
        self.scratchpad += f'\nThought {self.step_n}: {thought}'
        self.scratchpad += f'\nAction {self.step_n}: {action_str}'
        print(f'Thought {self.step_n}: {thought[:80]}')
        print(f'ACTION=> {action_str}')
        print("EXPECTED:", expected)
        print("NEXT_INTENT:", next_intent)
        print('CORRECTION:', correction)

        action_type, argument = '', ''
        try:
            action_type, argument = parse_action(action_str)
        except Exception:
            print("Invalid Action")

        self.scratchpad += f'\nObservation {self.step_n}: '

        # 4. Execute action
        if action_type == 'Finish':
            self.answer = argument
            observation = 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.scratchpad += observation
            self.finished   = True
            print(observation)
            # Store successful trajectory in RAR store
            self.trajectory_store.add(TrajectoryRecord(
                question    = self.question,
                scratchpad  = self.scratchpad,
                reflection  = '',
                success     = True,
                error_class = 'SUCCESS',
            ))
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                observation = format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                observation = 'Could not find that page, please try again.'
        elif action_type == 'Lookup':
            try:
                observation = format_step(self.docstore.lookup(argument))
            except ValueError:
                observation = ('The last page Searched was not found, '
                               'so you cannot Lookup a keyword in it. '
                               'Please try one of the similar pages given.')
        else:
            observation = ('Invalid Action. '
                           'Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')

        self.scratchpad += observation
        print(f'Observation {self.step_n}: {observation[:100]}')

        # 5. Store step knowledge — no extra LLM call
        if correction and len(correction) > 15:
            self.knowledge_store.add(StepKnowledge(
                action_intent = next_intent or action_str,
                rule          = correction,
                positive      = False,
            ))
            print(f'  [STAR] Correction stored: {correction[:80]}')
        elif expected and observation and self._prediction_matched(expected, observation):
            rule = f"{action_type}[{argument}] returns {expected[:100]}"
            self.knowledge_store.add(StepKnowledge(
                action_intent = next_intent or action_str,
                rule          = rule,
                positive      = True,
            ))

        # Store prev expected/observation for mismatch injection next step
        self._prev_expected    = expected
        self._prev_observation = observation

        # 6. Pre-fetch step knowledge for next step using NEXT_INTENT
        if next_intent:
            self._prefetched_knowledge = self.knowledge_store.retrieve(
                next_intent, k=self.retrieval_k)
            if self._prefetched_knowledge:
                print(f'  [STAR] Pre-fetched {len(self._prefetched_knowledge)} rules '
                      f'for: {next_intent[:60]}')
        else:
            self._prefetched_knowledge = []

        self.step_n += 1

    # ------------------------------------------------------------------
    # Prediction match
    # ------------------------------------------------------------------

    @staticmethod
    def _prediction_matched(expected: str, actual: str) -> bool:
        stopwords = {'the','a','an','is','was','will','to','of','in','and','or','it','this','that'}
        exp_words = set(re.findall(r'\w+', expected.lower())) - stopwords
        act_words = set(re.findall(r'\w+', actual.lower()))   - stopwords
        if not exp_words:
            return False
        return len(exp_words & act_words) / len(exp_words) > 0.4

    # ------------------------------------------------------------------
    # Reflection — RAR style: retrieve trajectories, use their reflections
    # as context to generate new reflection, store trajectory
    # ------------------------------------------------------------------

    def _reflect(self) -> None:
        print('  [STAR] Reflecting...')

        # ── RAR: classify error (same signature as retrieval_agents.py) ───────
        error_class = classify_error(
            self.question, self.scratchpad, self.reflect_llm)
        print(f'  Error class: {error_class}')

        # ── RAR: retrieve similar trajectories ───────────────────────────────
        retrieved = self.trajectory_store.retrieve(
            question    = self.question,
            error_class = error_class,
            k           = self.retrieval_k,
        )
        print(f'  [STAR] Retrieved {len(retrieved)} trajectories '
              f'({sum(1 for r in retrieved if r.success)} successes, '
              f'{sum(1 for r in retrieved if not r.success)} failures)')

        # ── STAR: step knowledge as additional context ────────────────────────
        recent_knowledge = self.knowledge_store.retrieve(self.question, k=3)
        knowledge_ctx    = format_step_knowledge(recent_knowledge)

        # ── Build reflection prompt: same as RAR + step knowledge ─────────────
        retrieved_context  = format_retrieved_trajectories(retrieved)
        current_block = (
            "\n=== CURRENT FAILED TRAJECTORY ===\n"
            f"Question: {self.question}\n"
            f"Error class: {error_class}\n\n"
            f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
        )
        instruction = (
            "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
            "FAILED_STEP: <the step where reasoning went wrong>\n"
            "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
            "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial>\n"
            "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
        )
        reflection_prompt = (
            RETRIEVAL_REFLECTION_HEADER + retrieved_context +
            current_block + instruction
        )
        if knowledge_ctx:
            reflection_prompt = knowledge_ctx + "\n" + reflection_prompt

        reflection = format_step(
            self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
        print(f'  Reflection: {reflection[:120]}...')

        # ── Store trajectory (same as RAR) ────────────────────────────────────
        self.trajectory_store.add(TrajectoryRecord(
            question    = self.question,
            scratchpad  = self.scratchpad,
            reflection  = reflection,
            success     = False,
            error_class = error_class,
        ))

        # ── reflections_str: same format as RAR ───────────────────────────────
        # last attempt scratchpad + current reflection only
        # NOT all accumulated reflections — token efficient
        self.reflections     = [reflection]
        self.reflections_str = (
            format_last_attempt(self.question, self.scratchpad) +
            format_reflections(self.reflections,
                               header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        )
        print(self.reflections_str)

    # ------------------------------------------------------------------
    # Agent prompt
    # ------------------------------------------------------------------

    def _build_agent_prompt(self, knowledge_str: str = '') -> str:
        parts = []
        # Step knowledge (STAR) — injected before examples
        if knowledge_str:
            parts.append(knowledge_str)
        parts.append(self.react_examples)
        parts.append('(END OF EXAMPLES)')
        # Reflections from retrieved trajectories (RAR) — not all reflections
        if self.reflections_str:
            parts.append(self.reflections_str)
        parts.append(f"Question: {self.question}")
        parts.append(f"Scratchpad:\n{self.scratchpad}")
        # Mismatch injection — tells LLM to write CORRECTION
        if (self._prev_expected and self._prev_observation
                and not self._prediction_matched(self._prev_expected,
                                                  self._prev_observation)):
            parts.append(
                f"NOTE: Your previous prediction was WRONG.\n"
                f"You expected: {self._prev_expected[:100]}\n"
                f"You got:      {self._prev_observation[:100]}\n"
                f"You MUST include a CORRECTION rule in your response."
            )
        parts.append(STAR_STEP_INSTRUCTION)
        return '\n\n'.join(parts)

    # ------------------------------------------------------------------
    # Standard interface
    # ------------------------------------------------------------------

    def is_finished(self) -> bool: return self.finished
    def is_correct(self)  -> bool: return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps) or
            (len(self.enc.encode(self._build_agent_prompt())) > 3896)
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n                = 1
        self.finished              = False
        self.answer                = ''
        self.scratchpad            = ''
        self._prefetched_knowledge = []
        self._prev_expected        = ''
        self._prev_observation     = ''
        # reflections_str persists across resets — updated in _reflect()

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key      = key


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string: str):
    for action_type in ['Finish', 'Search', 'Lookup']:
        match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
        if match:
            return action_type, match.group(1)
    return None, None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') if step else ''

def format_reflections(reflections: List[str],
                       header: str = REFLECTION_HEADER) -> str:
    if not reflections:
        return ''
    return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str, scratchpad: str,
                        header: str = LAST_TRIAL_HEADER) -> str:
    return (header + f'Question: {question}\n' +
            truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() +
            '\n(END PREVIOUS TRIAL)\n')

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600,
                        tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = list(filter(lambda x: x.startswith('Observation'), lines))
    observations_by_tokens = sorted(observations,
                                    key=lambda x: len(tokenizer.encode(x)))
    while (len(gpt2_enc.encode('\n'.join(lines))) > n_tokens
           and observations_by_tokens):
        largest = observations_by_tokens.pop(-1)
        ind     = lines.index(largest)
        lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):  return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer: str, key: str) -> bool:
    return normalize_answer(answer) == normalize_answer(key)