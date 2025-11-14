import os
import re
import sys

import math
import torch
import numpy as np

from vllm import LLM, SamplingParams
from tqdm import tqdm

from src.model_paths import MODEL_PATHS
from src.args import parse_args
from src.utils import set_seed, load_config_from_yaml
from src.task_generator import generate_linear_task, generate_circle_task, generate_moon_task
from src.datagenerator import generate_grid_data, generate_N_dim_tasks
from src.prompt_v2 import *
from src.dataloader import load_data

"""
Run the command:

    - Qwen2.5-7B standard:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode standard --task_mode linear_classification --batch_size=2500
    
    - Qwen2.5-7B any:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode any --task_mode linear_classification --batch_size=2500
    
    - Qwen2.5-7B decision_tree:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode decision_tree --task_mode linear_classification --batch_size=2500
    
    - Qwen2.5-7B k-NN:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode knn --task_mode linear_classification --batch_size=2500
    
    - Qwen2.5-7B SVM:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode svm --task_mode linear_classification --batch_size=2500
    
    - Qwen2.5-7B MLP:
        python implicit_reasoning_pred.py --gpu_id=0 --model_name Qwen2.5-7b --seed 11 --prompt_mode mlp --task_mode linear_classification --batch_size=2500
"""

ML_METHODS_OPTIONS = ["any", "decision_tree", "knn", "svm", "mlp"]
config = load_config_from_yaml("config.yaml")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # ========= model settings =========
    if args.parallel:
        gpus = args.gpu_id.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        tensor_parallel_size = len(gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        tensor_parallel_size = 1

    if args.model_name == "mistral-8b":
        model = LLM(
            model=MODEL_PATHS[args.model_name],
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            max_model_len=config["general_configs"]["max_token_length"]
        )
    else:
        model = LLM(
            model=MODEL_PATHS[args.model_name],
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            max_model_len=config["general_configs"]["max_token_length"]
        )
    
    tokenizer = model.get_tokenizer()

    # ========= data generation =========
    if args.task_mode == "linear_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_linear_task(num_classes=args.num_classes, mode="train", num_feat=int(args.data_type[0]), 
                                            num_samples=args.num_samples, precision=args.precision, randseed=args.seed)
    elif args.task_mode == "circle_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_circle_task(num_samples=args.num_samples, noise=0.03, 
                                            mode="train", precision=args.precision, randseed=args.seed)
    elif args.task_mode == "moon_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_moon_task(num_samples=args.num_samples, mode="train", precision=args.precision, randseed=args.seed)
    else:
        raise ValueError(f"Task mode {args.task_mode} is not supported")
    
    # ========= query data generation =========
    queries = generate_N_dim_tasks(data, num_coord=args.num_coordinate, 
                                   num_dim=int(args.data_type[0]), num_query=2500, 
                                   random_seed=args.task_seed)
    
    # ========= process data as a batch of prompts =========
    assert args.inference_mode == "implicit", f"In this file (implicit_reasoning_pred.py), only implicit reasoning is performed."
    
    if args.prompt_mode == "standard":
        batch_prompts = implicit_standard_batch_prompt_generation(in_context_data=data, in_context_labels=labels, 
                                                                  queries=queries, boolInstruct=True)
    elif args.prompt_mode in ML_METHODS_OPTIONS:
        batch_prompts = implicit_ml_batch_prompt_generation(in_context_data=data, in_context_labels=labels, queries=queries, 
                                                            method=args.prompt_mode, boolInstruct=True)
    else:
        raise ValueError("Invalid prompt mode.")
        
    dataloader, dataDict = load_data(promptList=batch_prompts, tokenizer=tokenizer, batch_size=args.batch_size)
    
    # ========= run prediction =========
    predictions = []
    original_probs = []
    perplexities = []
    with torch.no_grad():
        with tqdm(dataloader) as pbar:
            for data_item in pbar:
                outputs = model.generate(
                    data_item,
                    SamplingParams(
                        n=args.num_responses,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        stop_token_ids=[tokenizer.eos_token_id],
                        logprobs=1,
                    )
                )
                
                for idx, output in enumerate(outputs):
                    ans = output.outputs[0].text

                    # obtain the logprobs
                    token_logprobs = output.outputs[0].logprobs
                    
                    probs = []
                    for token_info in token_logprobs:
                        for key, value in token_info.items():
                            if value.rank == 1:
                                probs.append(value.logprob)
                                break
                    
                    if probs:
                        valid_probs = [prob for prob in probs if prob != float("-inf") and prob is not None]
                        if valid_probs:
                            avg_logprob = sum(valid_probs) / len(valid_probs)
                            perplexity = math.exp(-avg_logprob)
                        else:
                            perplexity = float(-1)
                    else:
                        valid_probs = probs
                        perplexity = float(-1)
                    
                    predictions.append(ans)
                    original_probs.append(valid_probs)
                    perplexities.append(perplexity)

    # ========= save results =========
    res_dict = {
        "train_data": data,
        "train_labels": labels,
        "query_data": queries,
        "predictions": predictions,
        "original_probs": original_probs,
        "perplexities": perplexities
    }
    
    save_root = os.path.join(config["general_configs"]["save_path"], "decision_boundary", "npy_results", 
                             f"{args.model_name}_{args.prompt_mode}_implicit_{args.num_classes}-classification_{args.task_mode}_{args.data_type}_tasks")
        
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        
    filename = f"randomseed-{args.seed}"
    if int(args.data_type[0]) > 2:
        filename += f"_taskseed-{args.task_seed}"
    filename += ".npy"
    np.save(os.path.join(save_root, filename), res_dict)
    print(f"Results successfully saved to {os.path.join(save_root, filename)}")
    

if __name__ == "__main__":
    main()
