#! /bin/bash

# Set environment variable to allow longer sequences
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GPU=0,1
methods=(any decision_tree knn svm mlp linear_regression)
task_modes=(linear_classification circle_classification moon_classification)

for method in ${methods[@]}
    do
        for task_mode in ${task_modes[@]}
        do
            echo "Running with model=Qwen2.5-72b, task_mode=$task_mode, method=$method"
            python explicit_reasoning_pred.py \
                --gpu_id=0 \
                --seed 11 \
                --model_name Qwen2.5-72b \
                --task_mode $task_mode \
                --data_type 2D \
                --inference_mode explicit \
                --prompt_mode $method \
                --batch_size=2500
        done
    done
