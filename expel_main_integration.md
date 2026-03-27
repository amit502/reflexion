# ExpeL Integration Notes

## How to add ExpeL to main.py (Programming)

### 1. Add to strategy_factory:

```python
elif strategy == "expel":
    return kwargs_wrapper_gen(run_expel_eval, delete_keys=["expansion_factor", "max_iters"])
```

### 2. Add ExpeL gather + extract before eval in main():

```python
if args.strategy == "expel":
    # Phase 1 — gather
    expel = ExpeL(max_insights=10, retrieval_k=3)
    gather_log = log_path.replace("expel", "expel_gather")
    run_expel_gather(dataset, args.model, args.language,
                     max_iters=10, pass_at_k=1,
                     log_path=gather_log, verbose=args.verbose,
                     expel=expel)
    run_expel_extract_insights(expel, args.model)

    # Phase 2 — eval (pass expel to run_strategy)
    run_expel_eval(dataset, args.model, args.language,
                   pass_at_k=1, log_path=log_path,
                   verbose=args.verbose, expel=expel)
```

## How to add ExpeL to ALFWorld main.py

### 1. Add 'expel' to strategy choices in argparse

### 2. In main(), handle expel strategy:

```python
if args.strategy == 'expel':
    expel = ExpeL(max_insights=10, retrieval_k=3)

    # Phase 1 — gathering (first 3 trials with Reflexion)
    for trial_idx in range(3):
        env_configs = run_trial(..., strategy=ReflexionStrategy.REFLEXION, ...)
        expel_store_trial_results(env_configs, trial_log_path, expel)
    expel.extract_insights(llm_fn)

    # Phase 2 — eval (single trial, inject ExpeL context into base_prompt)
    # In alfworld_trial.py run_trial(), before alfworld_run():
    #   expel_prefix = build_expel_alfworld_prefix(task_desc, expel)
    #   base_prompt  = expel_prefix + base_prompt
```
