#!/bin/bash
# Experiment 1: Simple generation (baseline — no reflection)
# python main.py \
#   --run_name "simple" \
#   --root_dir "root" \
#   --dataset_path ./benchmarks/humaneval-py.jsonl \
#   --strategy "simple" \
#   --language "py" \
#   --model "gpt-oss" \
#   --pass_at_k 1 \
#   --max_iters 5 \
#   --verbose

python main.py \
  --run_name "simple" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "simple" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 1 \
  --verbose