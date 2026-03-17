#!/bin/bash
# Experiment 2: ReAct Only (base — no memory, no reflection)
python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "react" \
        --strategy base \
        --model "gpt-oss"

# python main.py \
#         --num_trials 1 \
#         --num_envs 1 \
#         --run_name "react" \
#         --strategy base \
#         --model "gpt-oss"