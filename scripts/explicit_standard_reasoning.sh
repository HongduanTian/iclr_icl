#! /bin/bash

# Set environment variable to allow longer sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GPU=0,1
task_modes=(linear_classification circle_classification moon_classification)

# Run for each LLM
for task_mode in ${task_modes[@]}
    do
        echo "Running with model=Qwen2.5-72B, task_mode=$task_mode"
        python explicit_reasoning_pred.py \
            --gpu_id=$GPU \
            --seed 11 \
            --model_name Qwen2.5-72b \
            --task_mode $task_mode \
            --prompt_mode standard \
            --inference_mode explicit \
            --batch_size=2500 \
            --data_type 2D
    done
