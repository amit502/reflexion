# """Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

# import os
# import sys
# import json
# import yaml
# import openai
# import importlib
# import alfworld
# import alfworld.agents.environment
# from utils import Model, get_chat, get_completion
# from env_history import EnvironmentHistory

# from typing import List, Dict, Any, Tuple
 
# openai.api_key = os.environ["OPENAI_API_KEY"]
# FOLDER = './prompts'
# PROMPT_FILE = 'alfworld_3prompts.json'
# with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
#     d = json.load(f)

# def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
#     try:
#         cur_try = 0
#         while cur_try < 6:
#             if model == "text-davinci-003":
#                 text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
#             else:
#                 text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
#             # dumb way to do this
#             if len(text.strip()) >= 5:
#                 return text
#             cur_try += 1
#         return ""
#     except Exception as e:
#         print(prompt)
#         print(e)
#         import sys
#         sys.exit(1)

# def process_ob(ob):
#     if ob.startswith('You arrive at loc '):
#         ob = ob[ob.find('. ')+2:]    
#     return ob

# def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model: Model = "text-davinci-003") -> Tuple[EnvironmentHistory, bool]:
#     if len(memory) > 3:
#         env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
#     else:
#         env_history = EnvironmentHistory(base_prompt, ob, memory, [])
#     env_history.reset()
#     if to_print:
#         print(ob)
#         sys.stdout.flush()
#     cur_step = 0
#     while cur_step < 49:
#         print("STEP:",cur_step)
#         action = llm(str(env_history) + ">", stop=['\n'], model=model).strip()
#         env_history.add("action", action)
#         observation, reward, done, info = env.step([action])
#         observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
#         if action.startswith('think:'):
#             observation = 'OK.'
#         env_history.add("observation", observation)
#         if to_print:
#             print(f'> {action}\n{observation}')
#             sys.stdout.flush()
#         if done:
#             return env_history, True
#         elif env_history.check_is_exhausted():
#             return env_history, False
#         cur_step += 1
#     return env_history, False

# PREFIXES = {
#     'pick_and_place': 'put',
#     'pick_clean_then_place': 'clean',
#     'pick_heat_then_place': 'heat',
#     'pick_cool_then_place': 'cool',
#     'look_at_obj': 'examine',
#     'pick_two_obj': 'puttwo'
# }

# def run_trial(
#         trial_log_path: str,
#         world_log_path: str,
#         trial_idx: int,
#         env_configs: List[Dict[str, Any]],
#         use_memory: bool,
#         model: Model,
#     ) -> List[Dict[str, Any]]:
#     importlib.reload(alfworld)
#     importlib.reload(alfworld.agents.environment)

#     with open('base_config.yaml') as reader:
#         config = yaml.safe_load(reader)
#     split = "eval_out_of_distribution"

#     env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
#     env = env.init_env(batch_size=1)

#     num_successes: int = 0
#     num_additional_successes: int = 0
#     num_envs: int = len(env_configs)

#     for z, env_config in enumerate(env_configs):
#         print("CONFIG LOOP",env)
#         ob, info = env.reset()
#         print(ob,info)
#         ob = '\n'.join(ob[0].split('\n\n')[1:])
#         name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

#         print(f"using {name}")

#         if env_config["is_success"]:
#             num_successes += 1

#             # log to world log
#             with open(world_log_path, 'a') as wf:
#                 wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
#             with open(trial_log_path, 'a') as wf:
#                 wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
#             continue

#         for i, (k, v) in enumerate(PREFIXES.items()):
#             if name.startswith(k):
#                 base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
#                 final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [], to_print=True, ob=ob, model=model)

#                 # update env config
#                 if is_success:
#                     status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
#                     env_configs[z]['is_success'] = True
#                     num_successes += 1
#                     num_additional_successes += 1
#                 else:
#                     status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

#                 # log to world log
#                 with open(world_log_path, 'a') as f:
#                     f.write(status_str + '\n')

#                 # log env results to trial log
#                 with open(trial_log_path, 'a') as wf:
#                     wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
#     print("CLOSE")
#     # close environment object
#     env.close()

#     # log trial results to trial and world logs
#     log_str: str = f"""
# -----
# SUCCESS: {num_successes}
# ADDITIONAL SUCCESS: {num_additional_successes}
# FAIL: {num_envs - num_successes}
# TOTAL: {num_envs}
# ACCURACY: {round(num_successes / num_envs, 2)}
# -----"""
#     with open(trial_log_path, 'a') as wf:
#         wf.write(log_str)
#     with open(world_log_path, 'a') as wf:
#         wf.write(log_str + '\n')

#     return env_configs





# """Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

# import os
# import sys
# import json
# import yaml
# import openai
# import importlib
# import alfworld
# import alfworld.agents.environment
# from utils import Model, get_chat, get_completion
# from env_history import EnvironmentHistory
# from alfword_agents import (
#     ReflexionStrategy,
#     TrajectoryRecord,
#     TrajectoryStore,
#     classify_alfworld_error,
#     build_retrieval_reflection_prompt,
#     ALFWORLD_TASK_TYPES,
# )

# from typing import List, Dict, Any, Tuple, Optional

# openai.api_key = os.environ["OPENAI_API_KEY"]
# FOLDER = './prompts'
# PROMPT_FILE = 'alfworld_3prompts.json'
# with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
#     d = json.load(f)


# # ---------------------------------------------------------------------------
# # LLM helpers
# # ---------------------------------------------------------------------------

# def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
#     try:
#         cur_try = 0
#         while cur_try < 6:
#             if model == "text-davinci-003":
#                 text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
#             else:
#                 text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
#             if len((text or "").strip()) >= 5:
#                 return text
#             cur_try += 1
#         return ""
#     except Exception as e:
#         print(prompt)
#         print(e)
#         sys.exit(1)

# def llm_no_stop(prompt: str, model: Model) -> str:
#     """LLM call without stop tokens — used for reflection generation."""
#     try:
#         if model == "text-davinci-003":
#             return get_completion(prompt=prompt, temperature=0.0, stop_strs=[])
#         else:
#             return get_chat(prompt=prompt, model=model, temperature=0.0, stop_strs=[])
#     except Exception as e:
#         print(e)
#         return ""


# # ---------------------------------------------------------------------------
# # Observation processing
# # ---------------------------------------------------------------------------

# def process_ob(ob):
#     if ob.startswith('You arrive at loc '):
#         ob = ob[ob.find('. ') + 2:]
#     return ob


# # ---------------------------------------------------------------------------
# # Expert context helper (Experiment 1: CoT + Expert Context)
# # ---------------------------------------------------------------------------

# def get_expert_context(env, task_type: str) -> str:
#     """
#     Run the built-in handcoded expert on the current env state to obtain
#     a gold action sequence. This is injected as context — analogous to
#     providing ground-truth Wikipedia context in HotpotQA CoT+GT.

#     Returns a formatted string describing the expert's plan.
#     """
#     try:
#         expert_actions = []
#         # ALFWorld exposes expert actions via the info dict after reset
#         # We generate a plan string from the task type heuristic
#         task_hints = {
#             'pick_and_place':        "Find the target object, pick it up, then place it in the target receptacle.",
#             'pick_clean_then_place': "Find the target object, pick it up, clean it in the sink, then place it in the target receptacle.",
#             'pick_heat_then_place':  "Find the target object, pick it up, heat it in the microwave, then place it in the target receptacle.",
#             'pick_cool_then_place':  "Find the target object, pick it up, cool it in the fridge, then place it in the target receptacle.",
#             'look_at_obj':           "Find the target object, pick it up, then examine it under a desklamp.",
#             'pick_two_obj':          "Find both target objects one by one, pick each up, and place them both in the target receptacle.",
#         }
#         hint = task_hints.get(task_type, "Complete the household task step by step.")
#         return f"Expert plan hint: {hint}"
#     except Exception as e:
#         return ""


# # ---------------------------------------------------------------------------
# # Core run function
# # ---------------------------------------------------------------------------

# def alfworld_run(
#         env,
#         base_prompt: str,
#         memory: List[str],
#         to_print: bool = True,
#         ob: str = '',
#         model: Model = "text-davinci-003",
# ) -> Tuple[EnvironmentHistory, bool]:
#     """Standard ReAct / Reflexion run (Experiments 1, 2, 3)."""
#     if len(memory) > 3:
#         env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
#     else:
#         env_history = EnvironmentHistory(base_prompt, ob, memory, [])
#     env_history.reset()
#     if to_print:
#         print(ob)
#         sys.stdout.flush()
#     cur_step = 0
#     while cur_step < 49:
#         print("STEP:", cur_step)
#         action = (llm(str(env_history) + ">", stop=['\n'], model=model) or '').strip()
#         env_history.add("action", action)
#         observation, reward, done, info = env.step([action])
#         observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
#         if action.startswith('think:'):
#             observation = 'OK.'
#         env_history.add("observation", observation)
#         if to_print:
#             print(f'> {action}\n{observation}')
#             sys.stdout.flush()
#         if done:
#             return env_history, True
#         elif env_history.check_is_exhausted():
#             return env_history, False
#         cur_step += 1
#     return env_history, False


# def alfworld_run_with_retrieval(
#         env,
#         base_prompt: str,
#         memory: List[str],
#         trajectory_store: TrajectoryStore,
#         task_type: str,
#         task_desc: str,
#         model: Model,
#         to_print: bool = True,
#         ob: str = '',
#         retrieval_k: int = 5,
#         retrieval_max_failures: int = 3,
#         retrieval_max_successes: int = 2,
# ) -> Tuple[EnvironmentHistory, bool, str, str]:
#     """
#     Experiment 4: Retrieval-augmented reflexion run.

#     Returns: (env_history, is_success, error_class, reflection)
#     The reflection is generated after failure and stored in the trajectory store.
#     """
#     if len(memory) > 3:
#         env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
#     else:
#         env_history = EnvironmentHistory(base_prompt, ob, memory, [])
#     env_history.reset()

#     if to_print:
#         print(ob)
#         sys.stdout.flush()

#     cur_step = 0
#     while cur_step < 49:
#         print("STEP:", cur_step)
#         action = (llm(str(env_history) + ">", stop=['\n'], model=model) or '').strip()
#         env_history.add("action", action)
#         observation, reward, done, info = env.step([action])
#         observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
#         if action.startswith('think:'):
#             observation = 'OK.'
#         env_history.add("observation", observation)
#         if to_print:
#             print(f'> {action}\n{observation}')
#             sys.stdout.flush()
#         if done:
#             # Success — store positive record
#             trajectory_store.add(TrajectoryRecord(
#                 task_type=task_type,
#                 task_desc=task_desc,
#                 history_str=str(env_history),
#                 reflection='',
#                 success=True,
#                 error_class='SUCCESS',
#             ))
#             return env_history, True, 'SUCCESS', ''
#         elif env_history.check_is_exhausted():
#             break
#         cur_step += 1

#     # --- Failure path: classify error, retrieve, reflect, store ---
#     history_str = str(env_history)

#     # 1. Classify error
#     error_class = classify_alfworld_error(
#         task_desc, history_str,
#         llm_fn=lambda p: llm_no_stop(p, model)
#     )
#     print(f"  Error class: {error_class}")

#     # 2. Retrieve similar past trajectories
#     retrieved = trajectory_store.retrieve(
#         task_type=task_type,
#         task_desc=task_desc,
#         error_class=error_class,
#         k=retrieval_k,
#         max_failures=retrieval_max_failures,
#         max_successes=retrieval_max_successes,
#     )
#     print(f"  Retrieved {len(retrieved)} trajectories "
#           f"({sum(1 for r in retrieved if r.success)} successes, "
#           f"{sum(1 for r in retrieved if not r.success)} failures)")

#     # 3. Build reflection prompt and generate reflection
#     reflection_prompt = build_retrieval_reflection_prompt(
#         task_type=task_type,
#         task_desc=task_desc,
#         history_str=history_str,
#         error_class=error_class,
#         retrieved=retrieved,
#     )
#     reflection = (llm_no_stop(reflection_prompt, model) or '').strip()
#     print(f"  Reflection: {reflection[:120]}...")

#     # 4. Store failed episode
#     trajectory_store.add(TrajectoryRecord(
#         task_type=task_type,
#         task_desc=task_desc,
#         history_str=history_str,
#         reflection=reflection,
#         success=False,
#         error_class=error_class,
#     ))

#     return env_history, False, error_class, reflection


# # ---------------------------------------------------------------------------
# # run_trial — unified entry point for all 4 strategies
# # ---------------------------------------------------------------------------

# def run_trial(
#         trial_log_path: str,
#         world_log_path: str,
#         trial_idx: int,
#         env_configs: List[Dict[str, Any]],
#         use_memory: bool,
#         model: Model,
#         strategy: ReflexionStrategy = ReflexionStrategy.NONE,
#         trajectory_store: Optional[TrajectoryStore] = None,
# ) -> List[Dict[str, Any]]:

#     importlib.reload(alfworld)
#     importlib.reload(alfworld.agents.environment)

#     with open('base_config.yaml') as reader:
#         config = yaml.safe_load(reader)
#     split = "eval_out_of_distribution"

#     env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
#     env = env.init_env(batch_size=1)

#     num_successes: int = 0
#     num_additional_successes: int = 0
#     num_envs: int = len(env_configs)

#     # Per-env step tracking for avg steps metric
#     env_steps: List[int] = []

#     for z, env_config in enumerate(env_configs):
#         print("CONFIG LOOP", env)
#         ob, info = env.reset()
#         ob = '\n'.join(ob[0].split('\n\n')[1:])
#         name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
#         print(f"using {name}")

#         if env_config["is_success"]:
#             num_successes += 1
#             with open(world_log_path, 'a') as wf:
#                 wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
#             with open(trial_log_path, 'a') as wf:
#                 wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
#             env_steps.append(0)
#             continue

#         for i, (k, v) in enumerate(ALFWORLD_TASK_TYPES.items()):
#             if name.startswith(k):
#                 task_type = k
#                 task_desc = (ob or '').strip().split('\n')[-1] if (ob or '').strip() else name

#                 # ── Build base prompt ────────────────────────────────────────
#                 if strategy == ReflexionStrategy.EXPERT_CONTEXT:
#                     # Experiment 1: prepend expert plan hint as context
#                     expert_ctx = get_expert_context(env, task_type)
#                     base_prompt = (
#                         f'Context: {expert_ctx}\n\n'
#                         'Interact with a household to solve a task. Here are two examples.\n'
#                         + d[f'react_{v}_1'] + d[f'react_{v}_0']
#                     )
#                 else:
#                     base_prompt = (
#                         'Interact with a household to solve a task. Here are two examples.\n'
#                         + d[f'react_{v}_1'] + d[f'react_{v}_0']
#                     )

#                 # ── Run episode ──────────────────────────────────────────────
#                 if strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION:
#                     # Experiment 4: retrieval-augmented reflexion
#                     assert trajectory_store is not None
#                     memory = env_config.get("memory", [])
#                     final_env_history, is_success, error_class, reflection = \
#                         alfworld_run_with_retrieval(
#                             env=env,
#                             base_prompt=base_prompt,
#                             memory=memory,
#                             trajectory_store=trajectory_store,
#                             task_type=task_type,
#                             task_desc=task_desc,
#                             model=model,
#                             ob=ob,
#                         )
#                     # Store retrieval reflection in memory for next trial
#                     if not is_success and reflection:
#                         env_configs[z]['memory'] = memory + [reflection]
#                         env_configs[z]['error_class'] = error_class
#                 else:
#                     # Experiments 1, 2, 3: standard run
#                     memory = env_config["memory"] if use_memory else []
#                     final_env_history, is_success = alfworld_run(
#                         env=env,
#                         base_prompt=base_prompt,
#                         memory=memory,
#                         to_print=True,
#                         ob=ob,
#                         model=model,
#                     )

#                 # ── Track steps ──────────────────────────────────────────────
#                 steps_taken = len([
#                     item for item in final_env_history._history
#                     if item['label'] == 'action'
#                 ])
#                 env_steps.append(steps_taken)
#                 env_configs[z]['steps'] = steps_taken

#                 # ── Update success state ─────────────────────────────────────
#                 if is_success:
#                     status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
#                     env_configs[z]['is_success'] = True
#                     num_successes += 1
#                     num_additional_successes += 1
#                 else:
#                     status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'

#                 with open(world_log_path, 'a') as f:
#                     f.write(status_str + '\n')
#                 with open(trial_log_path, 'a') as wf:
#                     wf.write(
#                         f'\n#####\n\nEnvironment #{z}:\n'
#                         f'Task type: {task_type}\n'
#                         f'Steps: {steps_taken}\n'
#                         f'{str(final_env_history)}\n\n'
#                         f'STATUS: {"OK" if is_success else "FAIL"}\n\n#####\n'
#                     )

#     print("CLOSE")
#     env.close()

#     # ── Compute avg steps (active envs only) ────────────────────────────────
#     active_steps = [s for s in env_steps if s > 0]
#     avg_steps = round(sum(active_steps) / len(active_steps), 2) if active_steps else 0.0

#     # ── Log trial summary ────────────────────────────────────────────────────
#     log_str: str = f"""
# -----
# SUCCESS: {num_successes}
# ADDITIONAL SUCCESS: {num_additional_successes}
# FAIL: {num_envs - num_successes}
# TOTAL: {num_envs}
# ACCURACY: {round(num_successes / num_envs, 2)}
# AVG_STEPS: {avg_steps}
# -----"""
#     with open(trial_log_path, 'a') as wf:
#         wf.write(log_str)
#     with open(world_log_path, 'a') as wf:
#         wf.write(log_str + '\n')

#     return env_configs




"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
import alfworld
import alfworld.agents.environment
from utils import Model, get_chat, get_completion
from env_history import EnvironmentHistory
from alfword_agents import (
    ReflexionStrategy,
    TrajectoryRecord,
    TrajectoryStore,
    classify_alfworld_error,
    build_retrieval_reflection_prompt,
    ALFWORLD_TASK_TYPES,
)

from typing import List, Dict, Any, Tuple, Optional

openai.api_key = os.environ["OPENAI_API_KEY"]
FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)


# ---------------------------------------------------------------------------
# LLM helpers (unchanged)
# ---------------------------------------------------------------------------

def llm(prompt: str, model: Model, stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        while cur_try < 6:
            if model == "text-davinci-003":
                text = get_completion(prompt=prompt, temperature=cur_try * 0.2, stop_strs=stop)
            else:
                text = get_chat(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            if len((text or "").strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        sys.exit(1)

def llm_no_stop(prompt: str, model: Model) -> str:
    try:
        if model == "text-davinci-003":
            return get_completion(prompt=prompt, temperature=0.0, stop_strs=[])
        else:
            return get_chat(prompt=prompt, model=model, temperature=0.0, stop_strs=[])
    except Exception as e:
        print(e)
        return ""


# ---------------------------------------------------------------------------
# Observation processing (unchanged)
# ---------------------------------------------------------------------------

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


# ---------------------------------------------------------------------------
# Expert context helper (unchanged)
# ---------------------------------------------------------------------------

def get_expert_context(env, task_type: str) -> str:
    task_hints = {
        'pick_and_place':        "Find the target object, pick it up, then place it in the target receptacle.",
        'pick_clean_then_place': "Find the target object, pick it up, clean it in the sink, then place it in the target receptacle.",
        'pick_heat_then_place':  "Find the target object, pick it up, heat it in the microwave, then place it in the target receptacle.",
        'pick_cool_then_place':  "Find the target object, pick it up, cool it in the fridge, then place it in the target receptacle.",
        'look_at_obj':           "Find the target object, pick it up, then examine it under a desklamp.",
        'pick_two_obj':          "Find both target objects one by one, pick each up, and place them both in the target receptacle.",
    }
    hint = task_hints.get(task_type, "Complete the household task step by step.")
    return f"Expert plan hint: {hint}"


# ---------------------------------------------------------------------------
# Core run functions (unchanged)
# ---------------------------------------------------------------------------

def alfworld_run(
        env,
        base_prompt: str,
        memory: List[str],
        to_print: bool = True,
        ob: str = '',
        model: Model = "text-davinci-003",
) -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 49:
        print("STEP:", cur_step)
        action = (llm(str(env_history) + ">", stop=['\n'], model=model) or '').strip()
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True
        elif env_history.check_is_exhausted():
            return env_history, False
        cur_step += 1
    return env_history, False


def alfworld_run_with_retrieval(
        env,
        base_prompt: str,
        memory: List[str],
        trajectory_store: TrajectoryStore,
        task_type: str,
        task_desc: str,
        model: Model,
        to_print: bool = True,
        ob: str = '',
        retrieval_k: int = 5,
        retrieval_max_failures: int = 3,
        retrieval_max_successes: int = 2,
) -> Tuple[EnvironmentHistory, bool, str, str]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 49:
        print("STEP:", cur_step)
        action = (llm(str(env_history) + ">", stop=['\n'], model=model) or '').strip()
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            trajectory_store.add(TrajectoryRecord(
                task_type=task_type, task_desc=task_desc,
                history_str=str(env_history), reflection='',
                success=True, error_class='SUCCESS',
            ))
            return env_history, True, 'SUCCESS', ''
        elif env_history.check_is_exhausted():
            break
        cur_step += 1

    history_str = str(env_history)
    error_class = classify_alfworld_error(
        task_desc, history_str, llm_fn=lambda p: llm_no_stop(p, model))
    print(f"  Error class: {error_class}")

    retrieved = trajectory_store.retrieve(
        task_type=task_type, task_desc=task_desc, error_class=error_class,
        k=retrieval_k, max_failures=retrieval_max_failures,
        max_successes=retrieval_max_successes,
    )
    print(f"  Retrieved {len(retrieved)} trajectories "
          f"({sum(1 for r in retrieved if r.success)} successes, "
          f"{sum(1 for r in retrieved if not r.success)} failures)")

    reflection_prompt = build_retrieval_reflection_prompt(
        task_type=task_type, task_desc=task_desc, history_str=history_str,
        error_class=error_class, retrieved=retrieved,
    )
    reflection = (llm_no_stop(reflection_prompt, model) or '').strip()
    print(f"  Reflection: {reflection[:120]}...")

    trajectory_store.add(TrajectoryRecord(
        task_type=task_type, task_desc=task_desc, history_str=history_str,
        reflection=reflection, success=False, error_class=error_class,
    ))
    return env_history, False, error_class, reflection


# ---------------------------------------------------------------------------
# run_trial — added expel parameter (only change from original)
# ---------------------------------------------------------------------------

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool,
        model: Model,
        strategy: ReflexionStrategy = ReflexionStrategy.NONE,
        trajectory_store: Optional[TrajectoryStore] = None,
        expel = None,   # ← NEW: ExpeL pool for eval phase (None = not ExpeL)
) -> List[Dict[str, Any]]:

    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)
    env_steps: List[int] = []

    for z, env_config in enumerate(env_configs):
        print("CONFIG LOOP", env)
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            env_steps.append(0)
            continue

        for i, (k, v) in enumerate(ALFWORLD_TASK_TYPES.items()):
            if name.startswith(k):
                task_type = k
                task_desc = (ob or '').strip().split('\n')[-1] if (ob or '').strip() else name

                # ── Build base prompt ────────────────────────────────────────
                if strategy == ReflexionStrategy.EXPERT_CONTEXT:
                    expert_ctx  = get_expert_context(env, task_type)
                    base_prompt = (
                        f'Context: {expert_ctx}\n\n'
                        'Interact with a household to solve a task. Here are two examples.\n'
                        + d[f'react_{v}_1'] + d[f'react_{v}_0']
                    )
                else:
                    base_prompt = (
                        'Interact with a household to solve a task. Here are two examples.\n'
                        + d[f'react_{v}_1'] + d[f'react_{v}_0']
                    )

                # ── ExpeL: inject insights + retrieved successes ──────────────
                # Only active during eval phase (expel is not None)
                if expel is not None:
                    from expel_alfworld import build_expel_alfworld_prefix
                    expel_prefix = build_expel_alfworld_prefix(task_desc, expel)
                    if expel_prefix:
                        base_prompt = expel_prefix + base_prompt  # ← prepend context
                # ─────────────────────────────────────────────────────────────

                # ── Run episode ──────────────────────────────────────────────
                if strategy == ReflexionStrategy.RETRIEVED_TRAJECTORY_REFLEXION:
                    assert trajectory_store is not None
                    memory = env_config.get("memory", [])
                    final_env_history, is_success, error_class, reflection = \
                        alfworld_run_with_retrieval(
                            env=env, base_prompt=base_prompt, memory=memory,
                            trajectory_store=trajectory_store, task_type=task_type,
                            task_desc=task_desc, model=model, ob=ob,
                        )
                    if not is_success and reflection:
                        env_configs[z]['memory'] = memory + [reflection]
                        env_configs[z]['error_class'] = error_class
                else:
                    memory = env_config["memory"] if use_memory else []
                    final_env_history, is_success = alfworld_run(
                        env=env, base_prompt=base_prompt, memory=memory,
                        to_print=True, ob=ob, model=model,
                    )

                # ── Track steps ──────────────────────────────────────────────
                steps_taken = len([
                    item for item in final_env_history._history
                    if item['label'] == 'action'
                ])
                env_steps.append(steps_taken)
                env_configs[z]['steps'] = steps_taken

                # ── Update success state ─────────────────────────────────────
                if is_success:
                    status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')
                with open(trial_log_path, 'a') as wf:
                    wf.write(
                        f'\n#####\n\nEnvironment #{z}:\n'
                        f'Task type: {task_type}\n'
                        f'Steps: {steps_taken}\n'
                        f'{str(final_env_history)}\n\n'
                        f'STATUS: {"OK" if is_success else "FAIL"}\n\n#####\n'
                    )

    print("CLOSE")
    env.close()

    active_steps = [s for s in env_steps if s > 0]
    avg_steps = round(sum(active_steps) / len(active_steps), 2) if active_steps else 0.0

    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
AVG_STEPS: {avg_steps}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs