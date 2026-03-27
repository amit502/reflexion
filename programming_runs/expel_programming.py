"""
ExpeL baseline for HumanEval / HumanEval-Hard programming tasks.

Mirrors run_reflexion.py structure.

Two phases:
    Phase 1 — Gathering: run_expel_gather() — runs reflexion on dataset,
              stores trajectories, extracts insights
    Phase 2 — Evaluation: run_expel_eval() — single attempt with
              insights + retrieved successes injected into prompt
"""

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from expel_store import ExpeL, ExperienceRecord

from typing import List


# ---------------------------------------------------------------------------
# Phase 1 — Experience Gathering
# ---------------------------------------------------------------------------

def run_expel_gather(
        dataset: List[dict],
        model_name: str,
        language: str,
        max_iters: int,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        expel: ExpeL,
        is_leetcode: bool = False,
) -> None:
    """
    Run standard Reflexion on dataset to populate ExpeL pool.
    Identical to run_reflexion but additionally stores each
    trajectory in the ExpeL pool after completion.

    Call run_expel_extract_insights() after this to extract insights.
    """
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items   = len(dataset)
    num_success = resume_success_count(dataset)

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        cur_pass        = 0
        is_solved       = False
        reflections     = []
        implementations = []
        test_feedback   = []
        cur_func_impl   = ""
        skipped         = False

        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            if not cur_func_impl:
                skipped = True
                break

            implementations.append(cur_func_impl)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                is_solved = is_passing
                num_success += int(is_passing)
                break

            cur_iter    = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model)
                reflections.append(reflection)

                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"], model=model,
                    strategy="reflexion", prev_func_impl=cur_func_impl,
                    feedback=cur_feedback, self_reflection=reflection,
                )
                implementations.append(cur_func_impl)
                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10)
                    if is_passing:
                        is_solved = True
                        num_success += 1
                    break
                cur_iter += 1
            cur_pass += 1

        if not skipped:
            item["is_solved"]      = is_solved
            item["reflections"]    = reflections
            item["implementations"] = implementations
            item["test_feedback"]  = test_feedback
            item["solution"]       = cur_func_impl
            write_jsonl(log_path, [item], append=True)

            # ── Store in ExpeL pool ──────────────────────────────────────────
            # trajectory = final implementation + test feedback
            trajectory = (
                f"Implementation:\n{cur_func_impl[:600]}\n\n"
                f"Final feedback:\n{test_feedback[-1][:300] if test_feedback else ''}"
            )
            expel.add(ExperienceRecord(
                task_id=item.get("task_id", str(i)),
                task_desc=item["prompt"],
                trajectory=trajectory,
                success=is_solved,
            ))

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

    print(f"\nExpeL gathering complete. Pool: "
          f"{len([r for r in expel.pool if r.success])} successes, "
          f"{len([r for r in expel.pool if not r.success])} failures")


def run_expel_extract_insights(expel: ExpeL, model_name: str) -> None:
    """Extract insights from gathered experience. Call after run_expel_gather."""
    model   = model_factory(model_name)
    llm_fn  = lambda p: model.generate_chat([{"role": "user", "content": p}])
    expel.extract_insights(llm_fn)
    print("Extracted insights:")
    for i, ins in enumerate(expel.insights, 1):
        print(f"  {i}. {ins}")


# ---------------------------------------------------------------------------
# Phase 2 — Evaluation
# ---------------------------------------------------------------------------

def run_expel_eval(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        expel: ExpeL,
        is_leetcode: bool = False,
) -> None:
    """
    Single-attempt evaluation with ExpeL context injection.

    For each problem:
        1. Build prompt with ExpeL insights + retrieved successful implementations
        2. Single generation attempt — no reflection
        3. Evaluate against test suite
    """
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items   = len(dataset)
    num_success = 0

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        is_solved = False
        skipped   = False

        # ── Build ExpeL context ──────────────────────────────────────────────
        expel_context = expel.format_inference_context(item["prompt"])

        # Augment the prompt with ExpeL context
        augmented_prompt = item["prompt"]
        if expel_context:
            augmented_prompt = (
                "# Past Experience\n"
                f"{expel_context}\n\n"
                "# Current Problem\n"
                f"{item['prompt']}"
            )

        # ── Single generation attempt ────────────────────────────────────────
        cur_func_impl = gen.func_impl(augmented_prompt, model, "simple")
        if not cur_func_impl:
            skipped = True
        else:
            # Evaluate
            is_passing = exe.evaluate(
                item["entry_point"], cur_func_impl, item["test"], timeout=10)
            is_solved   = is_passing
            num_success += int(is_passing)

        if not skipped:
            item["is_solved"]       = is_solved
            item["reflections"]     = []       # no reflection in ExpeL eval
            item["implementations"] = [cur_func_impl] if cur_func_impl else []
            item["test_feedback"]   = []
            item["solution"]        = cur_func_impl or ""
            write_jsonl(log_path, [item], append=True)

        print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')