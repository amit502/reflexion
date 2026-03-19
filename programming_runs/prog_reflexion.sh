#!/bin/bash
# Experiment 3: Standard Reflexion
python main.py \
  --run_name "reflexion" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 10 \
  --verbose