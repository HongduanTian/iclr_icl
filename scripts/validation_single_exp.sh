#!/bin/bash

# Set environment variable to allow longer sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GPU=0,1
prompt_modes=(decision_tree knn svm mlp)
task_modes=(linear_classification circle_classification moon_classification)

# Run for each LLM
# for reference_mode in "implicit" "explicit"
# do
for prompt_mode in ${prompt_modes[@]}
do
    for task_mode in ${task_modes[@]}
    do
        echo "Running with model=Qwen2.5-72b, prompt_mode=$prompt_mode, task_mode=$task_mode"
        python validation_utilization_ml.py \
            --gpu_id=$GPU \
            --model_name Qwen2.5-72b \
            --prompt_mode $prompt_mode \
            --inference_mode implicit \
            --task_mode $task_mode \
            --data_type 2D \
            --num_classes 2 \
            --num_samples 128 \
            --num_eval 1000 \
            --batch_size 1000 \
            --seed 11 \
            --num_responses 1
    done
done
#done