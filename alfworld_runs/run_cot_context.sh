#!/bin/bash
# Experiment 1: CoT + Expert Context (ALFWorld analog of CoT+Context in HotpotQA)
python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "cot_context" \
        --strategy expert_context \
        --model "gpt-oss"
 