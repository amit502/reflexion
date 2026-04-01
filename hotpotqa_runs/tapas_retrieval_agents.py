import re, string, os
from typing import List, Union, Literal, Tuple
from enum import Enum
import numpy as np
import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import (COT_REFLECTION_SYSTEM_PROMPT, COT_SYSTEM_PROMPT, REACT_SYSTEM_PROMPT,
                     REFLECTION_SYSTEM_PROMPT, reflect_prompt, react_agent_prompt,
                     react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER,
                     REFLECTION_AFTER_LAST_TRIAL_HEADER, CLASSIFY_ERROR_SYSTEM_PROMPT,
                     cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt,
                     COT_INSTRUCTION, COT_REFLECT_INSTRUCTION, SUMMARIZE_REFLECTION_INSTRUCTION)
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT
from typing import Optional
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('../..')
from policy_store import PolicyStore


class ReflexionStrategy(Enum):
    NONE                              = 'base'
    LAST_ATTEMPT                      = 'last_trial'
    REFLEXION                         = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION        = 'last_trial_and_reflexion'
    LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION = 'last_trial_and_summarized_reflexion'
    LAST_ATTEMPT_AND_RETRIEVAL_REFLEXION  = 'last_trial_and_retrieval_reflexion'
    RETRIEVED_TRAJECTORY_REFLEXION    = 'retrieved_trajectory_reflexion'
    TAPAS                             = 'tapas'   # ← new


# ---------------------------------------------------------------------------
# TrajectoryRecord + TrajectoryStore (unchanged)
# ---------------------------------------------------------------------------

class TrajectoryRecord:
    def __init__(self, question, scratchpad, reflection, success, error_class="UNKNOWN"):
        self.question    = question
        self.scratchpad  = scratchpad
        self.reflection  = reflection
        self.success     = success
        self.error_class = error_class
        self._embedding: Optional[np.ndarray] = None

    def embedding(self, embed_fn) -> np.ndarray:
        if self._embedding is None:
            self._embedding = embed_fn(self.question)
        return self._embedding


class TrajectoryStore:
    _st_model = None

    def __init__(self, embed_fn=None, tau=0.1, adaptive_tau=True,
                 error_bonus=0.5, mmr_lambda=0.5):
        self.records      = []
        self.embed_fn     = embed_fn if embed_fn is not None else self._sentence_transformer_embed
        self.tau          = tau
        self.adaptive_tau = adaptive_tau
        self.error_bonus  = error_bonus
        self.mmr_lambda   = mmr_lambda

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)

    def retrieve(self, question, error_class, k=5,
                 max_failures=3, max_successes=2):
        if not self.records:
            return []
        q_emb = self.embed_fn(question)
        d     = q_emb.shape[0]
        tau   = (0.05 + 0.25 * min(len(self.records) / 100.0, 1.0)
                 if self.adaptive_tau else self.tau)

        logits = []
        for rec in self.records:
            dot   = float(np.dot(q_emb, rec.embedding(self.embed_fn))) / np.sqrt(d)
            bonus = self.error_bonus if rec.error_class == error_class else 0.0
            logits.append((dot + bonus) / tau)

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
            TrajectoryStore._st_model = SentenceTransformer('all-MiniLM-L6-v2')
        return TrajectoryStore._st_model

    @staticmethod
    def _sentence_transformer_embed(text: str) -> np.ndarray:
        return TrajectoryStore._get_st_model().encode(text, normalize_embeddings=True).astype(np.float64)


# ---------------------------------------------------------------------------
# Prompt helpers (unchanged)
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

def format_retrieved_trajectories(records):
    if not records: return ''
    lines     = []
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


def classify_error(question, scratchpad, llm):
    TAXONOMY = ["WRONG_BRIDGE_ENTITY","SEARCH_TOO_BROAD","EARLY_FINISH",
                "ENTITY_CONFUSION","MISSING_HOP","UNKNOWN"]
    prompt = (
        "Classify the following failed question-answering trajectory into exactly ONE of these error types:\n"
        f"{', '.join(TAXONOMY)}\n\n"
        f"Question: {question}\n\nTrajectory:\n{truncate_scratchpad(scratchpad)}\n\n"
        "Reply with only the error type label, nothing else."
    )
    raw = format_step(llm(prompt, CLASSIFY_ERROR_SYSTEM_PROMPT))
    for label in TAXONOMY:
        if label in raw.upper():
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# CoTAgent (unchanged)
# ---------------------------------------------------------------------------

class CoTAgent:
    def __init__(self, question, context, key,
                 agent_prompt=cot_reflect_agent_prompt, reflect_prompt=cot_reflect_prompt,
                 cot_examples=COT, reflect_examples=COT_REFLECT,
                 self_reflect_llm=None, action_llm=None):
        self.question=question; self.context=context; self.key=key
        self.agent_prompt=agent_prompt; self.reflect_prompt=reflect_prompt
        self.cot_examples=cot_examples; self.reflect_examples=reflect_examples
        self.self_reflect_llm=self_reflect_llm or AnyOpenAILLM()
        self.action_llm=action_llm or AnyOpenAILLM()
        self.reflections=[]; self.reflections_str=''; self.answer=''; self.step_n=0
        self.reset()

    def run(self, reflexion_strategy=ReflexionStrategy.REFLEXION):
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset(); self.step(); self.step_n += 1

    def step(self):
        self.scratchpad += '\nThought:'; self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])
        self.scratchpad += '\nAction:'; action = self.prompt_agent(); self.scratchpad += ' ' + action
        try: action_type, argument = parse_action(action)
        except: print("Invalid Action"); action_type, argument = '', ''
        print(self.scratchpad.split('\n')[-1])
        self.scratchpad += '\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            self.scratchpad += 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.finished = True
        else:
            print('Invalid action type, please try again.')

    def reflect(self, strategy):
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections=[self.scratchpad]; self.reflections_str=format_last_attempt(self.question,self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections+=[self.prompt_reflection()]; self.reflections_str=format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str=format_last_attempt(self.question,self.scratchpad)
            self.reflections=[self.prompt_reflection()]
            self.reflections_str+='\n'+format_reflections(self.reflections,header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')

    def prompt_reflection(self):
        return format_step(self.self_reflect_llm(self._build_reflection_prompt(), COT_REFLECTION_SYSTEM_PROMPT))
    def reset(self): self.scratchpad=''; self.finished=False
    def prompt_agent(self): return format_step(self.action_llm(self._build_agent_prompt(), COT_SYSTEM_PROMPT))
    def _build_agent_prompt(self):
        return self.agent_prompt.format(examples=self.cot_examples,reflections=self.reflections_str,
                                        context=self.context,question=self.question,scratchpad=self.scratchpad)
    def _build_reflection_prompt(self):
        return self.reflect_prompt.format(examples=self.reflect_examples,context=self.context,
                                          question=self.question,scratchpad=self.scratchpad)
    def is_finished(self): return self.finished
    def is_correct(self): return EM(self.answer, self.key)


# ---------------------------------------------------------------------------
# ReactAgent (unchanged)
# ---------------------------------------------------------------------------

class ReactAgent:
    def __init__(self, question, key, max_steps=6, agent_prompt=react_agent_prompt,
                 docstore=None, react_llm=None):
        self.question=question; self.answer=''; self.key=key; self.max_steps=max_steps
        self.agent_prompt=agent_prompt; self.react_examples=WEBTHINK_SIMPLE6
        self.docstore=DocstoreExplorer(docstore or Wikipedia())
        self.llm=react_llm or AnyOpenAILLM()
        self.enc=tiktoken.encoding_for_model("text-davinci-003")
        self.__reset_agent()

    def run(self, reset=True):
        if reset: self.__reset_agent()
        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self):
        self.scratchpad += f'\nThought {self.step_n}:'; self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])
        self.scratchpad += f'\nAction {self.step_n}:'; action = self.prompt_agent()
        self.scratchpad += ' ' + action; print("ACTION=>", action)
        try: action_type, argument = parse_action(action)
        except: print("Invalid Action"); action_type, argument = '', ''
        print(self.scratchpad.split('\n')[-1])
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action_type == 'Finish':
            self.answer=argument
            self.scratchpad+='Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.finished=True; self.step_n+=1; return
        if action_type == 'Search':
            try: self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e: print(e); self.scratchpad += 'Could not find that page, please try again.'
        elif action_type == 'Lookup':
            try: self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError: self.scratchpad += 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
        print(self.scratchpad.split('\n')[-1]); self.step_n += 1

    def prompt_agent(self): return format_step(self.llm(self._build_agent_prompt(), REACT_SYSTEM_PROMPT))
    def _build_agent_prompt(self):
        return self.agent_prompt.format(examples=self.react_examples,question=self.question,scratchpad=self.scratchpad)
    def is_finished(self): return self.finished
    def is_correct(self): return EM(self.answer, self.key)
    def is_halted(self):
        return ((self.step_n > self.max_steps) or
                (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
    def __reset_agent(self): self.step_n=1; self.finished=False; self.scratchpad=''
    def set_qa(self, question, key): self.question=question; self.key=key


# ---------------------------------------------------------------------------
# ReactReflectAgent — RAR + TAPAS
# ---------------------------------------------------------------------------

class ReactReflectAgent(ReactAgent):
    def __init__(self, question, key, max_steps=6,
                 agent_prompt=react_reflect_agent_prompt,
                 reflect_prompt=reflect_prompt,
                 docstore=None, react_llm=None, reflect_llm=None,
                 trajectory_store=None,
                 retrieval_k=5, retrieval_max_failures=3, retrieval_max_successes=2,
                 policy_store=None):   # ← new
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm             = reflect_llm or AnyOpenAILLM()
        self.reflect_prompt          = reflect_prompt
        self.reflect_examples        = REFLECTIONS
        self.reflections: List[str]  = []
        self.reflections_str: str    = ''
        self.trajectory_store        = trajectory_store if trajectory_store is not None else TrajectoryStore()
        self.retrieval_k             = retrieval_k
        self.retrieval_max_failures  = retrieval_max_failures
        self.retrieval_max_successes = retrieval_max_successes
        self._current_error_class: str = "UNKNOWN"
        self.policy_store            = policy_store   # ← new: shared PolicyStore

    def run(self, reset=True, reflect_strategy=ReflexionStrategy.REFLEXION):
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)
        ReactAgent.run(self, reset)
        if reflect_strategy in (ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION,
                                 ReflexionStrategy.TAPAS) and self.is_correct():
            self.record_success()

    def reflect(self, strategy):
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections=[self.scratchpad]
            self.reflections_str=format_last_attempt(self.question,self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections+=[self.prompt_reflection()]
            self.reflections_str=format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str=format_last_attempt(self.question,self.scratchpad)
            self.reflections=[self.prompt_reflection()]
            self.reflections_str+=format_reflections(self.reflections,header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION:
            self.reflections_str=format_last_attempt(self.question,self.scratchpad)
            current=self.prompt_reflection()
            if not self.reflections: summarized=current
            else: self.reflections.append(current); summarized=self.prompt_summarized_reflection()
            self.reflections=[summarized]
            self.reflections_str+=format_reflections(self.reflections,header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        elif strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION:
            self._reflect_with_retrieval()
        elif strategy == ReflexionStrategy.TAPAS:
            self._reflect_with_tapas()   # ← new
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    # ------------------------------------------------------------------
    # RAR reflection (unchanged)
    # ------------------------------------------------------------------

    def _reflect_with_retrieval(self):
        self._current_error_class = classify_error(self.question, self.scratchpad, self.reflect_llm)
        print(f'  Error class: {self._current_error_class}')
        retrieved = self.trajectory_store.retrieve(
            question=self.question, error_class=self._current_error_class,
            k=self.retrieval_k, max_failures=self.retrieval_max_failures,
            max_successes=self.retrieval_max_successes,
        )
        print(f'  Retrieved {len(retrieved)} trajectories')
        reflection_prompt  = self._build_retrieval_reflection_prompt(retrieved)
        current_reflection = format_step(self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
        self.reflections  += [current_reflection]
        self.reflections_str  = format_last_attempt(self.question, self.scratchpad)
        self.reflections_str += format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        self.trajectory_store.add(TrajectoryRecord(
            question=self.question, scratchpad=self.scratchpad,
            reflection=current_reflection, success=False,
            error_class=self._current_error_class,
        ))

    # ------------------------------------------------------------------
    # TAPAS reflection — RAR + policy injection + policy update
    # ------------------------------------------------------------------

    def _reflect_with_tapas(self):
        # Step 1 — classify error (same as RAR)
        self._current_error_class = classify_error(self.question, self.scratchpad, self.reflect_llm)
        print(f'  Error class: {self._current_error_class}')

        # Step 2 — retrieve trajectories (same as RAR)
        retrieved = self.trajectory_store.retrieve(
            question=self.question, error_class=self._current_error_class,
            k=self.retrieval_k, max_failures=self.retrieval_max_failures,
            max_successes=self.retrieval_max_successes,
        )
        print(f'  Retrieved {len(retrieved)} trajectories')

        # Step 3 — get current policy for this error class
        policy_str = ""
        if self.policy_store is not None:
            policy = self.policy_store.get(self._current_error_class)
            policy_str = policy.to_prompt_str()
            if policy_str:
                print(f'  Policy v{policy.version} injected for {self._current_error_class}')

        # Step 4 — build reflection prompt with policy + trajectories
        reflection_prompt  = self._build_tapas_reflection_prompt(retrieved, policy_str)

        # Step 5 — generate reflection
        current_reflection = format_step(self.reflect_llm(reflection_prompt, REFLECTION_SYSTEM_PROMPT))
        self.reflections  += [current_reflection]
        self.reflections_str  = format_last_attempt(self.question, self.scratchpad)
        self.reflections_str += format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)

        # Step 6 — store trajectory (same as RAR)
        self.trajectory_store.add(TrajectoryRecord(
            question=self.question, scratchpad=self.scratchpad,
            reflection=current_reflection, success=False,
            error_class=self._current_error_class,
        ))

        # Step 7 — update policy from this failure + reflection (NEW)
        if self.policy_store is not None:
            llm_fn = lambda p: self.reflect_llm(p, REFLECTION_SYSTEM_PROMPT)
            self.policy_store.update(
                key=self._current_error_class,
                trajectory=truncate_scratchpad(self.scratchpad, tokenizer=self.enc),
                reflection=current_reflection,
                llm_fn=llm_fn,
            )

    def _build_tapas_reflection_prompt(self, retrieved, policy_str: str) -> str:
        retrieved_context   = format_retrieved_trajectories(retrieved)
        current_trial_block = (
            "\n=== CURRENT FAILED TRAJECTORY ===\n"
            f"Question: {self.question}\n"
            f"Error class: {self._current_error_class}\n\n"
            f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
        )
        instruction = (
            "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
            "FAILED_STEP: <step number where reasoning went wrong>\n"
            "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
            "WHAT_TO_DO_DIFFERENTLY: <exact first action for next trial>\n"
            "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
        )
        # Policy block prepended before trajectories
        policy_block = (policy_str + "\n") if policy_str else ""
        return RETRIEVAL_REFLECTION_HEADER + policy_block + retrieved_context + current_trial_block + instruction

    # ------------------------------------------------------------------
    # Agent prompt — inject policy at inference time for TAPAS
    # ------------------------------------------------------------------

    def _build_agent_prompt(self) -> str:
        base = self.agent_prompt.format(
            examples=self.react_examples, reflections=self.reflections_str,
            question=self.question, scratchpad=self.scratchpad)

        # Inject current policy into agent prompt if available
        if self.policy_store is not None and self._current_error_class != "UNKNOWN":
            policy = self.policy_store.get(self._current_error_class)
            policy_str = policy.to_prompt_str()
            if policy_str:
                return policy_str + "\n" + base
        return base

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def record_success(self):
        self.trajectory_store.add(TrajectoryRecord(
            question=self.question, scratchpad=self.scratchpad,
            reflection='', success=True, error_class='SUCCESS',
        ))

    def _build_retrieval_reflection_prompt(self, retrieved):
        retrieved_context   = format_retrieved_trajectories(retrieved)
        current_trial_block = (
            "\n=== CURRENT FAILED TRAJECTORY ===\n"
            f"Question: {self.question}\n"
            f"Error class: {self._current_error_class}\n\n"
            f"{truncate_scratchpad(self.scratchpad, tokenizer=self.enc).strip()}\n"
        )
        instruction = (
            "\nWrite a reflection for the CURRENT FAILED TRAJECTORY in EXACTLY this format:\n\n"
            "FAILED_STEP: <the step number where reasoning went wrong>\n"
            "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
            "WHAT_TO_DO_DIFFERENTLY: <exact first action to take in next trial, "
            "e.g. 'Search[Inception director] instead of Search[Inception]'>\n"
            "GENERALISATION: <one sentence on when this fix applies beyond this question>\n"
        )
        return RETRIEVAL_REFLECTION_HEADER + retrieved_context + current_trial_block + instruction

    def prompt_summarized_reflection(self):
        prompt = SUMMARIZE_REFLECTION_INSTRUCTION.format(reflections='\n- '.join(self.reflections))
        return format_step(self.reflect_llm(prompt, REFLECTION_SYSTEM_PROMPT))

    def prompt_reflection(self):
        return format_step(self.reflect_llm(self._build_reflection_prompt(), REFLECTION_SYSTEM_PROMPT))

    def _build_reflection_prompt(self):
        return self.reflect_prompt.format(
            examples=self.reflect_examples, question=self.question,
            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))


# ---------------------------------------------------------------------------
# String utilities (unchanged)
# ---------------------------------------------------------------------------

gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string):
    for action_type in ['Finish', 'Search', 'Lookup']:
        match = re.search(rf'{action_type}\[([^\]]+)\]', string, re.IGNORECASE)
        if match: return action_type, match.group(1)
    return None, None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') if step else ''

def format_reflections(reflections, header=REFLECTION_HEADER):
    if not reflections: return ''
    return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question, scratchpad, header=LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad, n_tokens=1600, tokenizer=gpt2_enc):
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens and observations_by_tokens:
        largest = observations_by_tokens.pop(-1)
        ind = lines.index(largest)
        lines[ind] = largest.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):  return " ".join(text.split())
    def remove_punc(text):      return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):            return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)