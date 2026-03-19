#!/bin/bash
# Experiment 2: CoT + GT Context (docstring as ground truth)
python main.py \
  --run_name "cot_gt" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "cot_gt" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 1 \
  --verbose