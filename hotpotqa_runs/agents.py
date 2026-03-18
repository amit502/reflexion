import re, string, os
from typing import List, Union, Literal, Tuple
from enum import Enum
import numpy as np
import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import COT_REFLECTION_SYSTEM_PROMPT, COT_SYSTEM_PROMPT, REACT_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT, reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER,CLASSIFY_ERROR_SYSTEM_PROMPT
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION, SUMMARIZE_REFLECTION_INSTRUCTION
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from typing import Optional
from sentence_transformers import SentenceTransformer

class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context 
    REFLEXION: Apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace 
    LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION: Summarize all reflections into one compressed belief state
    RETRIEVED_TRAJECTORY_REFLEXION: Retrieve top-k similar past trajectories (by question similarity
                                          + error class match) and use them as contrastive context when
                                          generating the reflection for the current failed episode.
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'
    LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION = 'last_trial_and_summarized_reflexion'
    LAST_ATTEMPT_AND_RETRIEVAL_REFLEXION = 'last_trial_and_retrieval_reflexion'
    RETRIEVED_TRAJECTORY_REFLEXION = 'retrieved_trajectory_reflexion'


# ---------------------------------------------------------------------------
# Trajectory store — a lightweight episodic memory shared across trials
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    """Stores a single episode for later retrieval."""
    def __init__(self,
                 question: str,
                 scratchpad: str,
                 reflection: str,
                 success: bool,
                 error_class: str = "UNKNOWN"):
        self.question    = question
        self.scratchpad  = scratchpad
        self.reflection  = reflection
        self.success     = success
        self.error_class = error_class          # e.g. WRONG_BRIDGE_ENTITY, EARLY_FINISH …
        self._embedding: Optional[np.ndarray] = None

    # Lazy embedding — computed once on first access
    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.question)
        return self._embedding


class TrajectoryStore:
    """
    Episodic memory of past trajectories.

    Retrieval uses a composite score:
        score = λ1 * cos_sim(q_curr, q_i)          # question semantics
              + λ2 * (error_class_i == error_class_curr)  # same failure mode
    
    Then Maximum Marginal Relevance (MMR) is applied to ensure the selected
    k trajectories are diverse (not near-duplicates of each other).
    Finally the k slots are split between failures and successes so the
    reflection always has a contrastive anchor.
    """

    def __init__(self,
                 embed_fn=None,
                 lambda_q: float = 0.6,
                 lambda_err: float = 0.4,
                 mmr_lambda: float = 0.5):
        """
        Args:
            embed_fn:    callable(text: str) -> np.ndarray  (unit vector)
                         If None, falls back to simple token-overlap similarity.
            lambda_q:    weight for question semantic similarity
            lambda_err:  weight for error-class match
            mmr_lambda:  trade-off between relevance and diversity in MMR
        """
        self.records: List[TrajectoryRecord] = []
        # self.embed_fn = embed_fn if embed_fn is not None else self._token_overlap_embed
        self.embed_fn = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.lambda_q   = lambda_q
        self.lambda_err = lambda_err
        self.mmr_lambda = mmr_lambda

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self,
                 question: str,
                 error_class: str,
                 k: int = 5,
                 max_failures: int = 3,
                 max_successes: int = 2) -> List[TrajectoryRecord]:
        """
        Return up to k trajectories most relevant to (question, error_class).
        The result contains at most max_failures failed episodes and
        max_successes successful episodes, selected via MMR for diversity.
        """
        if not self.records:
            return []

        q_emb = self.embed_fn(question)

        # 1. Score every record
        scored: List[Tuple[float, TrajectoryRecord]] = []
        for rec in self.records:
            sim_q   = float(np.dot(q_emb, rec.embedding(self.embed_fn)))
            sim_err = float(rec.error_class == error_class)
            score   = self.lambda_q * sim_q + self.lambda_err * sim_err
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 2. Split into failure / success pools
        failures  = [(s, r) for s, r in scored if not r.success]
        successes = [(s, r) for s, r in scored if r.success]

        # 3. MMR selection within each pool
        selected_failures  = self._mmr_select(failures,  max_failures)
        selected_successes = self._mmr_select(successes, max_successes)

        return selected_failures + selected_successes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mmr_select(self,
                    scored: List[Tuple[float, TrajectoryRecord]],
                    k: int) -> List[TrajectoryRecord]:
        """
        Greedy MMR:
            next = argmax [ λ * relevance(i) - (1-λ) * max_j∈selected sim(i,j) ]
        """
        if not scored:
            return []

        selected: List[TrajectoryRecord] = []
        candidates = list(scored)   # (score, record)

        while len(selected) < k and candidates:
            if not selected:
                # First pick: highest relevance
                _, best = max(candidates, key=lambda x: x[0])
            else:
                best_score = -1e9
                best = None
                sel_embs = [r.embedding(self.embed_fn) for r in selected]
                for rel_score, rec in candidates:
                    max_sim_to_selected = max(
                        float(np.dot(rec.embedding(self.embed_fn), se))
                        for se in sel_embs
                    )
                    mmr_score = (self.mmr_lambda * rel_score
                                 - (1 - self.mmr_lambda) * max_sim_to_selected)
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = rec

            selected.append(best)
            candidates = [(s, r) for s, r in candidates if r is not best]

        return selected

    # @staticmethod
    # def _token_overlap_embed(text: str) -> np.ndarray:
    #     """
    #     Fallback embedding: normalised bag-of-words token vector.
    #     Works without any external model; lower quality than dense embeddings.
    #     """
    #     tokens = re.findall(r'\w+', text.lower())
    #     vocab  = sorted(set(tokens))
    #     vec    = np.array([tokens.count(w) for w in vocab], dtype=float)
    #     norm   = np.linalg.norm(vec)
    #     return vec / norm if norm > 0 else vec

    _st_model = None  # class-level cache, loaded once

    @staticmethod
    def _get_st_model():
        if TrajectoryStore._st_model is None:
            from sentence_transformers import SentenceTransformer
            TrajectoryStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return TrajectoryStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        """
        Dense embedding using sentence-transformers all-MiniLM-L6-v2.
        384-dim, normalised. Loaded once and cached at class level.
        """
        model = TrajectoryStore._get_st_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float64)


# ---------------------------------------------------------------------------
# Prompt helpers for retrieval-augmented reflection
# ---------------------------------------------------------------------------

RETRIEVAL_REFLECTION_HEADER = """You are an expert at reasoning about what went wrong in a question-answering attempt.
Below you will find:
  1. Several PAST TRAJECTORIES (with their outcome) that are similar to the current attempt.
  2. The CURRENT FAILED TRAJECTORY.

Study the similarities and differences between past trajectories and the current one.
Use the successful trajectories as a reference for what a correct approach looks like.
Use the failed trajectories to identify recurring mistakes.
Then write a concise, actionable reflection for the CURRENT trajectory only.

"""

def format_retrieved_trajectories(records: List[TrajectoryRecord]) -> str:
    """
    Render retrieved trajectories into a readable block for the reflection prompt.
    Failures and successes are grouped and labelled clearly.
    """
    if not records:
        return ''

    lines = []
    failures  = [r for r in records if not r.success]
    successes = [r for r in records if r.success]

    if successes:
        lines.append("=== SIMILAR SUCCESSFUL TRAJECTORIES ===")
        for i, r in enumerate(successes, 1):
            lines.append(f"\n[Success {i}] Question: {r.question}")
            lines.append(truncate_scratchpad(r.scratchpad).strip())
            if r.reflection:
                lines.append(f"Reflection: {r.reflection}")

    if failures:
        lines.append("\n=== SIMILAR FAILED TRAJECTORIES ===")
        for i, r in enumerate(failures, 1):
            lines.append(f"\n[Failure {i}] Question: {r.question}  |  Error class: {r.error_class}")
            lines.append(truncate_scratchpad(r.scratchpad).strip())
            if r.reflection:
                lines.append(f"Reflection: {r.reflection}")

    return '\n'.join(lines)


def classify_error(question: str, scratchpad: str, llm: AnyOpenAILLM) -> str:
    """
    Ask the LLM to classify the failure into one of the known error types.
    Returns a string label. Falls back to 'UNKNOWN' on parse errors.

    Error taxonomy (HotpotQA-specific):
        WRONG_BRIDGE_ENTITY  – found the wrong intermediate entity
        SEARCH_TOO_BROAD     – query returned irrelevant passages
        EARLY_FINISH         – committed to answer before completing both hops
        ENTITY_CONFUSION     – conflated two entities with similar names
        MISSING_HOP          – answered only a 1-hop sub-question
        UNKNOWN              – does not fit any category
    """
    TAXONOMY = [
        "WRONG_BRIDGE_ENTITY",
        "SEARCH_TOO_BROAD",
        "EARLY_FINISH",
        "ENTITY_CONFUSION",
        "MISSING_HOP",
        "UNKNOWN",
    ]
    prompt = (
        "Classify the following failed question-answering trajectory into exactly ONE of these error types:\n"
        f"{', '.join(TAXONOMY)}\n\n"
        f"Question: {question}\n\n"
        f"Trajectory:\n{truncate_scratchpad(scratchpad)}\n\n"
        "Reply with only the error type label, nothing else."
    )
    raw = format_step(llm(prompt,CLASSIFY_ERROR_SYSTEM_PROMPT))
    # Accept the label even if the model adds surrounding text
    for label in TAXONOMY:
        if label in raw.upper():
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# CoT Agent (unchanged except minor formatting)
# ---------------------------------------------------------------------------

class CoTAgent:
    def __init__(self,
                    question: str,
                    context: str,
                    key: str,
                    agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
                    reflect_prompt: PromptTemplate = cot_reflect_prompt,
                    cot_examples: str = COT,
                    reflect_examples: str = COT_REFLECT,
                    self_reflect_llm: AnyOpenAILLM = AnyOpenAILLM(),
                    action_llm: AnyOpenAILLM = AnyOpenAILLM(),
                    ) -> None:
        self.question = question
        self.context = context
        self.key = key
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples 
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 0
        self.reset()

    def run(self, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def step(self) -> None:
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type=''
        argument=''
        try:
            action_type, argument = parse_action(action)
        except Exception as e:
            print("Invalid Action")
        print(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')
    
    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n' + format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        return format_step(self.self_reflect_llm(self._build_reflection_prompt(),COT_REFLECTION_SYSTEM_PROMPT))

    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished = False

    def prompt_agent(self) -> str:
        return format_step(self.action_llm(self._build_agent_prompt(),COT_SYSTEM_PROMPT))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples=self.cot_examples,
                            reflections=self.reflections_str,
                            context=self.context,
                            question=self.question,
                            scratchpad=self.scratchpad)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples=self.reflect_examples,
                            context=self.context,
                            question=self.question,
                            scratchpad=self.scratchpad)
 
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)   


# ---------------------------------------------------------------------------
# ReactAgent (unchanged)
# ---------------------------------------------------------------------------

class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(),
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.docstore = DocstoreExplorer(docstore)
        self.llm = react_llm
        
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print("ACTION=>", action)
        action_type=''
        argument=''
        try:
            action_type, argument = parse_action(action)
        except Exception as e:
            print("Invalid Action")
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])
        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.llm(self._build_agent_prompt(),REACT_SYSTEM_PROMPT))
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            question=self.question,
                            scratchpad=self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


# ---------------------------------------------------------------------------
# ReactReflectAgent — with RETRIEVED_TRAJECTORY_REFLEXION added
# ---------------------------------------------------------------------------

class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(),
                 trajectory_store: Optional[TrajectoryStore] = None,
                 retrieval_k: int = 5,
                 retrieval_max_failures: int = 3,
                 retrieval_max_successes: int = 2,
                 ) -> None:
        
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm    = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''

        # --- Retrieval reflexion state ---
        # A single TrajectoryStore can be shared across many questions so that
        # episodes from different questions all feed the retrieval memory.
        self.trajectory_store      = trajectory_store if trajectory_store is not None else TrajectoryStore()
        self.retrieval_k           = retrieval_k
        self.retrieval_max_failures  = retrieval_max_failures
        self.retrieval_max_successes = retrieval_max_successes
        self._current_error_class: str = "UNKNOWN"   # filled during reflect()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, reset=True,
            reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)
        ReactAgent.run(self, reset)

        if reflect_strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION \
            and self.is_correct():
            self.record_success()

    # ------------------------------------------------------------------
    # Reflect dispatcher
    # ------------------------------------------------------------------

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])

        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)

        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)

        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            current_reflection = self.prompt_reflection()
            if len(self.reflections) == 0:
                summarized = current_reflection
            else:
                self.reflections.append(current_reflection)
                summarized = self.prompt_summarized_reflection()
            self.reflections = [summarized]
            self.reflections_str += format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )

        elif strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION:
            self._reflect_with_retrieval()

        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')

        print(self.reflections_str)

    # ------------------------------------------------------------------
    # Core: retrieval-augmented reflection
    # ------------------------------------------------------------------

    def _reflect_with_retrieval(self) -> None:
        """
        RETRIEVED_TRAJECTORY_REFLEXION strategy:

        1. Classify the current failure into an error taxonomy class.
        2. Retrieve top-k similar past trajectories from the store
           (scored by question similarity + error class match, diversified by MMR,
            balanced between failures and successes).
        3. Build a rich reflection prompt that includes:
               - The retrieved trajectories as contrastive context
               - The current failed trajectory
        4. Generate the structured reflection.
        5. Store the current trajectory + new reflection in the store for
           future retrievals.
        """

        # Step 1 — Classify error (1 LLM call, cheap)
        self._current_error_class = classify_error(
            self.question, self.scratchpad, self.reflect_llm
        )
        print(f'  Error class: {self._current_error_class}')

        # Step 2 — Retrieve similar trajectories (0 LLM calls)
        retrieved = self.trajectory_store.retrieve(
            question=self.question,
            error_class=self._current_error_class,
            k=self.retrieval_k,
            max_failures=self.retrieval_max_failures,
            max_successes=self.retrieval_max_successes,
        )
        print(f'  Retrieved {len(retrieved)} trajectories '
              f'({sum(1 for r in retrieved if r.success)} successes, '
              f'{sum(1 for r in retrieved if not r.success)} failures)')

        # Step 3 — Build reflection prompt with retrieved context
        reflection_prompt = self._build_retrieval_reflection_prompt(retrieved)

        # Step 4 — Generate reflection (1 LLM call)
        current_reflection = format_step(self.reflect_llm(reflection_prompt,REFLECTION_SYSTEM_PROMPT))
        self.reflections += [current_reflection]

        # Step 5 — Assemble reflections_str exactly like LAST_ATTEMPT_AND_REFLEXION
        self.reflections_str = format_last_attempt(self.question, self.scratchpad)
        self.reflections_str += format_reflections(
            self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
        )

        # Step 6 — Persist current episode to the store for future episodes
        self.trajectory_store.add(TrajectoryRecord(
            question=self.question,
            scratchpad=self.scratchpad,
            reflection=current_reflection,
            success=False,                          # we only call reflect() on failure
            error_class=self._current_error_class,
        ))

    def _build_retrieval_reflection_prompt(self,
                                           retrieved: List[TrajectoryRecord]) -> str:
        """
        Compose the full prompt for retrieval-augmented reflection.

        Structure:
            [Header with instructions]
            [Retrieved trajectories — successes first, then failures]
            [Current failed trajectory]
            [Instruction to generate structured reflection]
        """
        retrieved_context = format_retrieved_trajectories(retrieved)

        current_trial_block = (
            "\n=== CURRENT FAILED TRAJECTORY ===\n"
            f"Question: {self.question}\n"
            f"Error class: {self._current_error_class}\n\n"
            f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
        )

        # instruction = (
        #     "\nBased on the trajectories above, write a concise reflection for the CURRENT "
        #     "FAILED TRAJECTORY. Your reflection must:\n"
        #     "  1. Identify the specific step where the reasoning went wrong.\n"
        #     "  2. Explain why that step was wrong, referencing the successful trajectories if available.\n"
        #     "  3. Give a concrete, actionable instruction for what to do differently next time.\n"
        #     "  4. Be specific enough to generalise beyond this exact question.\n"
        # )
        instruction = (
            "\nBased on the trajectories above, write a reflection for the CURRENT "
            "FAILED TRAJECTORY in EXACTLY this format, no other text:\n\n"
            "FAILED_STEP: <the step number where reasoning went wrong>\n"
            "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
            "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
            "e.g. 'Search[Inception director] instead of Search[Inception]'>\n"
            "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
        )

        return (
            RETRIEVAL_REFLECTION_HEADER
            + retrieved_context
            + current_trial_block
            + instruction
        )

    # ------------------------------------------------------------------
    # Helper: register a successful episode (call from your eval loop)
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """
        Call this after a successful run() to store the winning trajectory.
        This ensures the retrieval store has positive examples to contrast against.
        """
        self.trajectory_store.add(TrajectoryRecord(
            question=self.question,
            scratchpad=self.scratchpad,
            reflection='',          # no reflection needed for successes
            success=True,
            error_class='SUCCESS',
        ))

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------

    def prompt_summarized_reflection(self) -> str:
        prompt = SUMMARIZE_REFLECTION_INSTRUCTION.format(
            reflections='\n- '.join(self.reflections)
        )
        return format_step(self.reflect_llm(prompt,REFLECTION_SYSTEM_PROMPT))

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt(),REFLECTION_SYSTEM_PROMPT))

    def _build_reflection_prompt(self) -> str:
        base= self.reflect_prompt.format(
                            examples=self.reflect_examples,
                            question=self.question,
                            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))
        # structured_suffix = (
        #     "\nWrite your reflection in EXACTLY this format, no other text:\n\n"
        #     "FAILED_STEP: <step number>\n"
        #     "WHAT_WENT_WRONG: <one sentence>\n"
        #     "WHAT_TO_DO_DIFFERENTLY: <exact first action for next trial>\n"
        #     "GENERALISATION: <one sentence>\n"
        # )
        return base 
 
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            reflections=self.reflections_str,
                            question=self.question,
                            scratchpad=self.scratchpad)

    # def _build_agent_prompt(self) -> str:
    #     if self.reflections_str:
    #         emphasis = (
    #             "!! IMPORTANT: You have attempted this question before and FAILED. "
    #             "Read the reflection below carefully. "
    #             "You MUST change your approach — do NOT repeat the same actions as your previous attempt. "
    #             "Your very first action should directly follow WHAT_TO_DO_DIFFERENTLY below !!\n\n"
    #         )
    #         reflections_with_emphasis = emphasis + self.reflections_str
    #     else:
    #         reflections_with_emphasis = self.reflections_str

    #     return self.agent_prompt.format(
    #                         examples=self.react_examples,
    #                         reflections=reflections_with_emphasis,
    #                         question=self.question,
    #                         scratchpad=self.scratchpad)


# ---------------------------------------------------------------------------
# String utilities (unchanged)
# ---------------------------------------------------------------------------

gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string):
    # Try to find Finish, Search, or Lookup specifically
    for action_type in ['Finish', 'Search', 'Lookup']:
        pattern = rf'{action_type}\[([^\]]+)\]'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return action_type, match.group(1)
    return None, None

# def parse_action(string):
#     pattern = r'(\w+)\[([^\]]+)\]' #r'^(\w+)\[(.+)\]$'
#     match = re.search(pattern, string)
    
#     if match:
#         action_type = match.group(1)
#         argument = match.group(2)
#         return action_type, argument
    
#     else:
#         return None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') if step else ''

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens and observations_by_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)