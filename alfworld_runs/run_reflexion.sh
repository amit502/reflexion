# python main.py \
#         --num_trials 10 \
#         --num_envs 134 \
#         --run_name "reflexion_run_logs" \
#         --use_memory \
#         --model "gpt-3.5-turbo"

# python main.py \
#         --num_trials 1 \
#         --num_envs 1 \
#         --run_name "reflexion" \
#         --strategy reflexion \
#         --use_memory \
#         --model "gpt-oss"

python main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "reflexion" \
        --strategy reflexion \
        --use_memory \
        --model "gpt-oss"
 