#!/bin/bash

# model list
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

# Create the directory
MODEL_DIR="/home/cshdtian/pretrained_models/LargeLanguageModels"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating directory: $MODEL_DIR"
    mkdir -p $MODEL_DIR
else
    echo "Directory already exists: $MODEL_DIR"
fi

# download function, with retry mechanism
download_model() {
    local model=$1
    local max_retries=5
    local retry_count=0
    local wait_time=10

    # 检查模型是否已存在（目录非空）
    if [ -d "$MODEL_DIR/$model" ] && [ "$(ls -A \"$MODEL_DIR/$model\")" ]; then
        echo "Model $model already exists in $MODEL_DIR/$model, skipping download."
        return 0
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Downloading model: $model (Attempt $((retry_count + 1))/$max_retries)"
        
        if huggingface-cli download $model --resume-download --local-dir "$MODEL_DIR/$model" --local-dir-use-symlinks False; then
            echo "Successfully downloaded $model"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Download failed. Waiting $wait_time seconds before retrying..."
                sleep $wait_time
                # increase waiting time to avoid frequent retries
                wait_time=$((wait_time + 5))
            fi
        fi
    done

    echo "Failed to download $model after $max_retries attempts"
    return 1
}

# iterate over the model list and download
failed_models=()
for model in "${MODELS[@]}"; do
    if ! download_model "$model"; then
        failed_models+=("$model")
    fi
done

# 报告下载结果
if [ ${#failed_models[@]} -eq 0 ]; then
    echo "All models have been downloaded successfully!"
else
    echo "The following models failed to download:"
    printf '%s\n' "${failed_models[@]}"
    echo "Please check your network connection and try again."
fi