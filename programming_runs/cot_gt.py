# """
# Experiment 2: CoT + Ground Truth Context for programming tasks.

# Analogous to HotpotQA CoT+GT where the ground truth Wikipedia passage is given.
# Here the docstring IS the ground truth specification — we extract it and inject
# it as structured context to guide chain-of-thought reasoning before coding.
# """

# from utils import enumerate_resume, make_printv, write_jsonl
# from executors import executor_factory
# from generators import generator_factory, model_factory
# from programming_agents import build_cot_gt_prompt, PY_COT_GT_INSTRUCTION

# from typing import List


# def run_cot_gt(
#     dataset: List[dict],
#     model_name: str,
#     language: str,
#     pass_at_k: int,
#     log_path: str,
#     verbose: bool,
#     is_leetcode: bool = False,
# ) -> None:

#     exe = executor_factory(language, is_leet=is_leetcode)
#     gen = generator_factory(language)
#     model = model_factory(model_name)

#     print_v = make_printv(verbose)

#     num_items = len(dataset)
#     num_success = 0

#     for i, item in enumerate_resume(dataset, log_path):
#         cur_pass = 0
#         is_solved = False
#         cur_func_impl = ""

#         while cur_pass < pass_at_k:
#             # Build CoT+GT prompt — inject docstring as structured context
#             cot_prompt = build_cot_gt_prompt(item["prompt"])

#             # Generate with CoT+GT prompt using simple strategy
#             # (single shot, no reflection — tests pure reasoning with GT context)
#             cur_func_impl = gen.func_impl(
#                 func_sig=cot_prompt,
#                 model=model,
#                 strategy="simple",
#             )
#             assert isinstance(cur_func_impl, str)

#             is_passing = exe.evaluate(
#                 item["entry_point"], cur_func_impl, item["test"],
#                 timeout=20 if is_leetcode else 10
#             )
#             if is_passing:
#                 is_solved = True
#                 num_success += 1
#                 break
#             cur_pass += 1

#         item["solution"] = cur_func_impl
#         item["is_solved"] = is_solved
#         write_jsonl(log_path, [item], append=True)

#         print_v(f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')

from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory
from programming_agents import build_cot_gt_prompt, PY_COT_GT_INSTRUCTION

from typing import List


def run_cot_gt(
    dataset: List[dict],
    model_name: str,
    language: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    max_iters: int = 10,  # ← added
) -> None:

    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0

    for i, item in enumerate_resume(dataset, log_path, resume=False):
        cur_pass = 0
        is_solved = False
        implementations = []
        test_feedback = []
        cur_func_impl = ""

        while cur_pass < pass_at_k and not is_solved:
            # ── Run max_iters attempts (mirrors reflexion loop) ───────────────
            cur_iter = 0
            while cur_iter < max_iters and not is_solved:
                cot_prompt = build_cot_gt_prompt(item["prompt"])
                cur_func_impl = gen.func_impl(
                    func_sig=cot_prompt,
                    model=model,
                    strategy="simple",
                )
                if not cur_func_impl:
                    print(f"Warning: empty implementation for problem {i} iter {cur_iter}, skipping")
                    cur_iter += 1
                    continue
                assert isinstance(cur_func_impl, str)
                implementations.append(cur_func_impl)

                # Run internal tests for feedback
                tests_i = gen.internal_tests(item["prompt"], model, 1)
                is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(feedback)

                # Check real tests
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"],
                    timeout=20 if is_leetcode else 10
                )
                if is_passing:
                    is_solved = True
                    num_success += 1

                cur_iter += 1
            cur_pass += 1

        item["solution"] = cur_func_impl
        item["is_solved"] = is_solved
        item["implementations"] = implementations
        item["reflections"] = []         # empty for cot_gt — needed for metrics
        item["test_feedback"] = test_feedback
        write_jsonl(log_path, [item], append=True)
        print(f'Problem {i+1}/{num_items}: {"✓ SOLVED" if is_solved else "✗ FAILED"} | '
              f'iters = {len(implementations)} | acc = {round(num_success/(i+1), 2)}')