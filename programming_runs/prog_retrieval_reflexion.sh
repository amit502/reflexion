#!/bin/bash
# Experiment 4: Retrieval-augmented Reflexion (novel method)
python main.py \
  --run_name "retrieval" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "retrieval" \
  --language "py" \
  --model "gpt-oss" \
  --pass_at_k 1 \
  --max_iters 10 \
  --verbose