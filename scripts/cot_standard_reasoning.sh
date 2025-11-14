#! /bin/bash

# Set environment variable to allow longer sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GPU=0,1
task_modes=(linear_classification circle_classification moon_classification)
num_responses=(5 11)

# Run for each LLM
for task_mode in ${task_modes[@]}
do
    for num_responses in ${num_responses[@]}
    do
        echo "Running CoT-$num_responses with model=Qwen2.5-72b, task_mode=$task_mode"
        python explicit_reasoning_pred.py \
            --gpu_id=$GPU \
            --seed 11 \
            --model_name Qwen2.5-72b \
            --task_mode $task_mode \
            --prompt_mode standard \
            --num_responses $num_responses \
            --inference_mode explicit \
            --batch_size=2500 \
            --data_type 2D
    done
done