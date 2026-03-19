"""
Experiment 4: Retrieval-augmented Reflexion for programming tasks.
Mirrors reflexion.py structure but with retrieval-augmented reflection.

Follows the same pattern as HotpotQA and ALFWorld retrieval strategies:
- classify error
- retrieve similar past trajectories (cross-problem memory)
- generate retrieval-augmented reflection
- store trajectory
- improve implementation using reflection
"""

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from generators.model import ModelBase, Message
from programming_agents import (
    TrajectoryRecord,
    TrajectoryStore,
    format_retrieved_trajectories,
    PROGRAMMING_ERROR_TAXONOMY,
)

from typing import List, Optional


# ---------------------------------------------------------------------------
# System prompts (mirrors HotpotQA/ALFWorld system prompt approach)
# ---------------------------------------------------------------------------

CLASSIFY_ERROR_SYSTEM_PROMPT = (
    "You are an error classification agent. You will be given a failed Python "
    "implementation and its test feedback. Classify the failure into exactly ONE "
    "error type from the provided list. Respond with only the error type label, "
    "nothing else. No explanation, no punctuation, just the label."
)

RETRIEVAL_REFLECTION_SYSTEM_PROMPT = (
    "You are a Python programming assistant analyzing why a function implementation failed. "
    "You will be given similar past trajectories as context and the current failed implementation. "
    "Write a concise, actionable reflection explaining what went wrong and what to do differently. "
    "Only provide the reflection text, not the implementation."
)


# ---------------------------------------------------------------------------
# Error classification using Message structure
# ---------------------------------------------------------------------------

def classify_error_with_model(func_sig: str,
                               implementation: str,
                               feedback: str,
                               model: ModelBase) -> str:
    """Classify error using the model's generate_chat interface."""
    prompt = (
        f"Classify the following failed Python implementation into exactly ONE of these error types:\n"
        f"{', '.join(PROGRAMMING_ERROR_TAXONOMY)}\n\n"
        f"Function signature:\n{func_sig[:300]}\n\n"
        f"Failed implementation:\n{implementation[:500]}\n\n"
        f"Test feedback:\n{feedback[:300]}\n\n"
        "Reply with only the error type label, nothing else."
    )
    messages = [
        Message(role="system", content=CLASSIFY_ERROR_SYSTEM_PROMPT),
        Message(role="user",   content=prompt),
    ]
    raw = model.generate_chat(messages=messages) or ""
    raw = raw.strip().upper()
    for label in PROGRAMMING_ERROR_TAXONOMY:
        if label in raw:
            return label
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Retrieval-augmented reflection using Message structure
# ---------------------------------------------------------------------------

def generate_retrieval_reflection(func_sig: str,
                                   implementation: str,
                                   feedback: str,
                                   error_class: str,
                                   retrieved: List[TrajectoryRecord],
                                   model: ModelBase) -> str:
    """
    Generate a retrieval-augmented reflection.

    Mirrors HotpotQA's _reflect_with_retrieval() and ALFWorld's
    alfworld_run_with_retrieval() — same 3-part structure:
        1. Retrieved trajectories as contrastive context
        2. Current failed trajectory
        3. Instruction to generate structured reflection
    """
    retrieved_context = format_retrieved_trajectories(retrieved)

    current_block = (
        f"=== CURRENT FAILED IMPLEMENTATION ===\n"
        f"Error class: {error_class}\n\n"
        f"Function:\n{func_sig[:300].strip()}\n\n"
        f"Implementation:\n{implementation[:500].strip()}\n\n"
        f"Test feedback:\n{feedback[:300].strip()}\n"
    )

    instruction = (
        "\nWrite a reflection for the CURRENT FAILED IMPLEMENTATION in EXACTLY this format:\n\n"
        "FAILED_STEP: <what specifically went wrong>\n"
        "WHAT_WENT_WRONG: <one sentence explaining the root cause>\n"
        "WHAT_TO_DO_DIFFERENTLY: <exact change to make in next implementation>\n"
        "GENERALISATION: <one sentence on when this fix applies beyond this problem>\n"
    )

    user_content = retrieved_context + "\n" + current_block + instruction

    messages = [
        Message(role="system", content=RETRIEVAL_REFLECTION_SYSTEM_PROMPT),
        Message(role="user",   content=user_content),
    ]
    reflection = model.generate_chat(messages=messages) or ""
    return reflection.strip()


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run_retrieval_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    trajectory_store: Optional[TrajectoryStore] = None,
) -> None:

    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    # Shared trajectory store — persists across ALL problems in the dataset
    # so earlier solved problems inform later ones (cross-problem memory)
    if trajectory_store is None:
        trajectory_store = TrajectoryStore()

    num_items = len(dataset)
    num_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        func_sig = item["prompt"]
        skipped = False

        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(func_sig, model, 1)

            # ── First attempt ────────────────────────────────────────────────
            # cur_func_impl = gen.func_impl(func_sig, model, "simple")
            cur_func_impl = gen.func_impl(func_sig, model, "simple")
            if not cur_func_impl:
                print(f"Warning: empty implementation for problem {i}, skipping")
                item["solution"] = ""
                item["is_solved"] = False
                write_jsonl(log_path, [item], append=True)
                skipped = True
                break
            assert isinstance(cur_func_impl, str)
            implementations.append(cur_func_impl)
            # assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                if is_passing:
                    is_solved = True
                    num_success += 1
                    # Store successful trajectory
                    trajectory_store.add(TrajectoryRecord(
                        func_sig=func_sig,
                        implementation=cur_func_impl,
                        feedback=feedback,
                        reflection='',
                        success=True,
                        error_class='SUCCESS',
                    ))
                break

            # ── Iterative improvement with retrieval-augmented reflection ────
            cur_iter = 1
            cur_feedback = feedback

            while cur_iter < max_iters:

                # Step 1 — Classify error (1 LLM call)
                error_class = classify_error_with_model(
                    func_sig=func_sig,
                    implementation=cur_func_impl,
                    feedback=cur_feedback,
                    model=model,
                )
                print_v(f"  Error class: {error_class}")

                # Step 2 — Retrieve similar past trajectories (0 LLM calls)
                retrieved = trajectory_store.retrieve(
                    func_sig=func_sig,
                    error_class=error_class,
                    k=3,
                    max_failures=2,
                    max_successes=1,
                )
                print_v(
                    f"  Retrieved {len(retrieved)} trajectories "
                    f"({sum(1 for r in retrieved if r.success)} successes, "
                    f"{sum(1 for r in retrieved if not r.success)} failures)"
                )

                # Step 3 — Generate retrieval-augmented reflection (1 LLM call)
                reflection = generate_retrieval_reflection(
                    func_sig=func_sig,
                    implementation=cur_func_impl,
                    feedback=cur_feedback,
                    error_class=error_class,
                    retrieved=retrieved,
                    model=model,
                )
                reflections.append(reflection)
                print_v(f"  Reflection: {reflection[:120]}...")

                # Step 4 — Store failed trajectory in store
                trajectory_store.add(TrajectoryRecord(
                    func_sig=func_sig,
                    implementation=cur_func_impl,
                    feedback=cur_feedback,
                    reflection=reflection,
                    success=False,
                    error_class=error_class,
                ))

                # Step 5 — Generate improved implementation using reflection
                # Uses existing gen.func_impl with reflexion strategy —
                # same interface as standard reflexion.py
                cur_func_impl = gen.func_impl(
                    func_sig=func_sig,
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                # Step 6 — Check if passes internal tests
                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                        # Store successful trajectory for future retrieval
                        trajectory_store.add(TrajectoryRecord(
                            func_sig=func_sig,
                            implementation=cur_func_impl,
                            feedback=cur_feedback,
                            reflection='',
                            success=True,
                            error_class='SUCCESS',
                        ))
                    break

                cur_iter += 1
            cur_pass += 1
        if not skipped:
            item["is_solved"] = is_solved
            item["reflections"] = reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["solution"] = cur_func_impl
            write_jsonl(log_path, [item], append=True)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')