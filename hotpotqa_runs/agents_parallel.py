import re, string, os
from typing import List, Union, Literal, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION, SUMMARIZE_REFLECTION_INSTRUCTION
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT


# ---------------------------------------------------------------------------
# Prompt addition for parallel ReAct
# ---------------------------------------------------------------------------
PARALLEL_REACT_ADDENDUM = """
You may issue MULTIPLE actions at once when they are independent of each other.
Format them as:
  Action {n}a: Search[topic1]
  Action {n}b: Search[topic2]
  Action {n}c: Lookup[keyword]

All actions labelled with the same step number will be executed in parallel and
their observations returned together before you think again.
Only issue sequential (single) actions when a later action depends on the result
of an earlier one.
""".strip()


# ---------------------------------------------------------------------------
# Helpers (kept identical to originals so nothing downstream breaks)
# ---------------------------------------------------------------------------
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string: str):
    pattern = r'(\w+)\[([^\]]+)\]'
    match = re.search(pattern, string)
    if match:
        return match.group(1), match.group(2)
    return None


def parse_parallel_actions(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse one or more actions from a model response.

    Handles both:
      - Classic single action:   "Action 2: Search[Alan Turing]"
      - Parallel sub-actions:    "Action 2a: Search[Alan Turing]"
                                 "Action 2b: Search[Enigma machine]"

    Returns a list of (sub_label, action_type, argument) tuples, e.g.:
        [("a", "Search", "Alan Turing"), ("b", "Search", "Enigma machine")]
    For a single classic action the sub_label is "".
    """
    # Match lines like: Action 3a: Search[foo]  OR  Action 3: Search[foo]
    pattern = r'Action\s+\d+([a-z]?)\s*:\s*(\w+)\[([^\]]+)\]'
    matches = re.findall(pattern, text)

    if matches:
        return [(sub_label, action_type, argument)
                for sub_label, action_type, argument in matches]
    return []


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') if step else ''


def format_reflections(reflections: List[str],
                       header: str = REFLECTION_HEADER) -> str:
    if not reflections:
        return ''
    return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])


def format_last_attempt(question: str,
                        scratchpad: str,
                        header: str = LAST_TRIAL_HEADER):
    return (header + f'Question: {question}\n'
            + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip()
            + '\n(END PREVIOUS TRIAL)\n')


def truncate_scratchpad(scratchpad: str,
                        n_tokens: int = 1600,
                        tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split('\n')
    observations = list(filter(lambda x: x.startswith('Observation'), lines))
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens and observations_by_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


# ---------------------------------------------------------------------------
# Original agents (unchanged)
# ---------------------------------------------------------------------------

class ReflexionStrategy(Enum):
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial'
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'
    LAST_ATTEMPT_AND_SUMMARIZED_REFLEXION = 'last_trial_and_summarized_reflexion'
    LAST_ATTEMPT_AND_RETRIEVAL_REFLEXION = 'last_trial_and_retrieval_reflexion'


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
                 action_llm: AnyOpenAILLM = AnyOpenAILLM()) -> None:
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
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            self.scratchpad += 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.finished = True
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
        return format_step(self.self_reflect_llm(self._build_reflection_prompt()))

    def reset(self) -> None:
        self.scratchpad: str = ''
        self.finished = False

    def prompt_agent(self) -> str:
        return format_step(self.action_llm(self._build_agent_prompt()))

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


class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM()) -> None:
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
        action_type = ''
        try:
            action_type, argument = parse_action(action)
        except Exception:
            print("Invalid Action")
        print(self.scratchpad.split('\n')[-1])

        self.scratchpad += f'\nObservation {self.step_n}: '
        if action_type == 'Finish':
            self.answer = argument
            self.scratchpad += 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += 'Could not find that page, please try again.'
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])
        self.step_n += 1

    def prompt_agent(self) -> str:
        return format_step(self.llm(self._build_agent_prompt()))

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
        return ((self.step_n > self.max_steps) or
                (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


class ReactReflectAgent(ReactAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM()) -> None:
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''

    def prompt_summarized_reflection(self) -> str:
        prompt = SUMMARIZE_REFLECTION_INSTRUCTION.format(
            reflections='\n- '.join(self.reflections))
        return format_step(self.reflect_llm(prompt))

    def run(self, reset=True,
            reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)
        ReactAgent.run(self, reset)

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
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad)


# ---------------------------------------------------------------------------
# NEW: ParallelReactAgent
# ---------------------------------------------------------------------------

class ParallelReactAgent(ReactAgent):
    """
    A ReAct agent that reduces LLM API calls by executing multiple independent
    tool calls per reasoning step in parallel.

    How it works
    ------------
    1.  **Think** – one LLM call produces a Thought AND one or more Actions.
    2.  **Act in parallel** – every Action emitted in that single response is
        dispatched concurrently via a ThreadPoolExecutor.
    3.  **Observe** – all Observations are written back into the scratchpad
        together before the next Think call.

    This means N independent lookups that would cost N+1 LLM calls in classic
    ReAct cost only 2 LLM calls here (1 Think + 1 final answer step).

    Prompt guidance
    ---------------
    PARALLEL_REACT_ADDENDUM is appended to the base prompt so the model knows
    it *can* emit multiple actions.  The model still emits a single action when
    a later step genuinely depends on an earlier result.

    Compatibility
    -------------
    *  Inherits fully from ReactAgent – `run`, `is_halted`, `is_correct`, etc.
       all work unchanged.
    *  max_parallel_actions caps the thread pool to avoid runaway fan-out.
    *  Falls back gracefully to single-action behaviour when the model only
       emits one action (no code changes needed).
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 max_parallel_actions: int = 4,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM()) -> None:
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.max_parallel_actions = max_parallel_actions

    # ------------------------------------------------------------------
    # Core step: one LLM call → many tool calls → many observations
    # ------------------------------------------------------------------

    def step(self) -> None:
        # ── Think + Act ── single LLM call returns both thought and action(s)
        self.scratchpad += f'\nThought {self.step_n}:'
        raw_response = self.prompt_agent()          # ONE LLM call
        self.scratchpad += ' ' + raw_response
        print(self.scratchpad.split('\n')[-1])

        # Parse all actions the model decided to emit
        actions = parse_parallel_actions(raw_response)

        # Fallback: model used old-style single action format
        if not actions:
            parsed = parse_action(raw_response)
            if parsed:
                actions = [('', parsed[0], parsed[1])]
            else:
                self.scratchpad += f'\nObservation {self.step_n}: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
                self.step_n += 1
                return

        # ── Check for Finish among the actions ──
        for sub_label, action_type, argument in actions:
            if action_type == 'Finish':
                self.answer = argument
                suffix = f'\nObservation {self.step_n}: '
                suffix += 'Answer is CORRECT' if self.is_correct() else 'Answer is INCORRECT'
                self.scratchpad += suffix
                print(suffix.strip())
                self.finished = True
                self.step_n += 1
                return

        # ── Execute remaining actions in parallel ──
        results = self._execute_parallel(actions)

        # ── Write all observations into scratchpad together ──
        for sub_label, action_type, argument, observation in results:
            label = f'{self.step_n}{sub_label}' if sub_label else str(self.step_n)
            entry = f'\nObservation {label} ({action_type}[{argument}]): {observation}'
            self.scratchpad += entry
            print(entry.strip())

        self.step_n += 1

    # ------------------------------------------------------------------
    # Parallel execution via thread pool
    # ------------------------------------------------------------------

    def _execute_parallel(
            self,
            actions: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str, str]]:
        """
        Run tool calls concurrently.  Returns list of
        (sub_label, action_type, argument, observation_text).

        Note: Wikipedia's DocstoreExplorer is not thread-safe for Lookup
        (it holds a `last_search` cursor), so Lookup calls are serialised
        while Search calls run freely in parallel.
        """
        # Separate safe-to-parallelise (Search) from serial (Lookup)
        search_actions = [(sl, at, arg) for sl, at, arg in actions if at == 'Search']
        other_actions  = [(sl, at, arg) for sl, at, arg in actions if at != 'Search']

        results: List[Tuple[str, str, str, str]] = []

        # Parallel searches
        with ThreadPoolExecutor(max_workers=min(len(search_actions) or 1,
                                                self.max_parallel_actions)) as pool:
            future_to_meta = {
                pool.submit(self._execute_single, at, arg): (sl, at, arg)
                for sl, at, arg in search_actions
            }
            for future in as_completed(future_to_meta):
                sl, at, arg = future_to_meta[future]
                try:
                    obs = future.result()
                except Exception as e:
                    obs = f'Error during {at}[{arg}]: {e}'
                results.append((sl, at, arg, obs))

        # Serial Lookup / other actions (order-preserved)
        for sl, at, arg in other_actions:
            obs = self._execute_single(at, arg)
            results.append((sl, at, arg, obs))

        # Restore original order so scratchpad matches emitted action order
        action_order = {(sl, at, arg): i for i, (sl, at, arg) in enumerate(actions)}
        results.sort(key=lambda r: action_order.get((r[0], r[1], r[2]), 999))
        return results

    def _execute_single(self, action_type: str, argument: str) -> str:
        """Execute one tool call and return the observation string."""
        if action_type == 'Search':
            try:
                return format_step(self.docstore.search(argument))
            except Exception as e:
                return f'Could not find that page, please try again. ({e})'
        elif action_type == 'Lookup':
            try:
                return format_step(self.docstore.lookup(argument))
            except ValueError:
                return ('The last page Searched was not found, so you cannot '
                        'Lookup a keyword in it. Please try one of the similar pages given.')
        else:
            return f'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

    # ------------------------------------------------------------------
    # Prompt: inject parallel-action instructions into the base prompt
    # ------------------------------------------------------------------

    def _build_agent_prompt(self) -> str:
        base = self.agent_prompt.format(
            examples=self.react_examples,
            question=self.question,
            scratchpad=self.scratchpad)
        return base + '\n\n' + PARALLEL_REACT_ADDENDUM


# ---------------------------------------------------------------------------
# ParallelReactReflectAgent  (parallel execution + Reflexion)
# ---------------------------------------------------------------------------

class ParallelReactReflectAgent(ReactReflectAgent):
    """
    Combines ParallelReactAgent's batched execution with ReactReflectAgent's
    reflection strategies.  All reflection methods are inherited unchanged;
    only the `step` and prompt-building are overridden.
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 max_parallel_actions: int = 4,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM()) -> None:
        super().__init__(question, key, max_steps, agent_prompt, reflect_prompt,
                         docstore, react_llm, reflect_llm)
        self.max_parallel_actions = max_parallel_actions

    # Reuse parallel step logic from ParallelReactAgent
    step             = ParallelReactAgent.step
    _execute_parallel = ParallelReactAgent._execute_parallel
    _execute_single   = ParallelReactAgent._execute_single

    def _build_agent_prompt(self) -> str:
        base = self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad)
        return base + '\n\n' + PARALLEL_REACT_ADDENDUM