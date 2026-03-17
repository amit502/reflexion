#!/bin/bash
# Experiment 4: Retrieval-augmented Reflexion (novel method — your contribution)
python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "retrieval" \
        --strategy retrieved_trajectory_reflexion \
        --model "gpt-oss"

# python main.py \
#         --num_trials 1 \
#         --num_envs 1 \
#         --run_name "retrieval" \
#         --strategy retrieved_trajectory_reflexion \
#         --model "gpt-oss"