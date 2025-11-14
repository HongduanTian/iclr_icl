#! /bin/bash

# Set environment variable to allow longer sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GPU=0,1
task_modes=(linear_classification circle_classification moon_classification)

# Run for each LLM
for task_mode in ${task_modes[@]}
    do
        echo "Running with model=Qwen2.5-72B, task_mode=$task_mode"
        python implicit_reasoning_pred.py \
            --gpu_id=$GPU \
            --model_name Qwen2.5-72b \
            --seed 11 \
            --prompt_mode standard \
            --task_mode $task_mode \
            --batch_size=25 \
            --data_type 2D
    done