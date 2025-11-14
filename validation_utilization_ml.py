import os
import sys
import argparse
import math
import torch
import random
import numpy as np

from vllm import LLM, SamplingParams
from tqdm import tqdm
from sklearn.datasets import make_classification, make_circles, make_moons

from macros import macros
from src.model_paths import MODEL_PATHS
from src.args import parse_args
from src.utils import set_seed
from src.prompt_v2 import *
from src.dataloader import load_data


# Macros
ROOT_SAVE = macros["SAVE_ROOT"]
MAX_TOKEN_LENGTH = 24064
NUM_SAMPLES_PER_CLASS = 10000


def generate_linear_classification_data(randseed:int, num_classes:int, num_samples_per_class:int, num_feat:int, precision:int):
    
    random.seed(randseed)
    class_sep = random.uniform(1., 1.4)
    
    assert num_samples_per_class < NUM_SAMPLES_PER_CLASS, f"The number of samples per class should be less than {NUM_SAMPLES_PER_CLASS}."
    
    data, labels = make_classification(
        n_samples=num_classes*num_samples_per_class,
        n_features=num_feat,
        n_informative=num_feat,
        n_redundant=0,
        n_repeated=0,
        flip_y=0,
        n_classes=num_classes,
        class_sep=class_sep,
        n_clusters_per_class=1,
        shuffle=True,
        random_state=randseed
    )
    
    if precision is not None:
        data = np.around(data, precision)
    
    return data, labels


def generate_circle_classification_data(randseed:int, num_samples_per_class:int, noise:float=0.03, precision:int=None):
    """Only generate binary classification data."""

    random.seed(randseed)
    
    factor = random.uniform(0.5, 0.9)
    data, labels = make_circles(n_samples=num_samples_per_class*2, factor=factor, noise=noise, random_state=randseed)
    
    if precision is not None:
        data = np.around(data, precision)
    
    return data, labels


def generate_moon_classification_data(randseed:int, num_samples_per_class:int, precision:int=None):

    random.seed(randseed)
    
    noise = random.uniform(0.1, 0.2)
    data, labels = make_moons(n_samples=num_samples_per_class*2, noise=noise, random_state=randseed)
    
    if precision is not None:
        data = np.around(data, precision)
    
    return data, labels


def data_sampling(randseed:int, data:np.ndarray, labels:np.ndarray, num_context_per_cls:int, num_query_per_cls:int):

    unique_labels = np.unique(labels)
    assert num_context_per_cls + num_query_per_cls < NUM_SAMPLES_PER_CLASS, f"The number of context and query per class should be less than {NUM_SAMPLES_PER_CLASS}."
    
    indices_list = []
    context_sample_indices = []
    query_sample_indices = []
    for target_label in unique_labels:
        res = np.where(labels == target_label)[0]   # indices of the samples
        rng = np.random.RandomState(randseed)
        rng.shuffle(res)
        selected_indices = list(res[:num_context_per_cls+num_query_per_cls])
        indices_list.append(len(selected_indices))
        context_sample_indices += list(res[:num_context_per_cls])
        query_sample_indices += list(res[num_context_per_cls:num_context_per_cls+num_query_per_cls])
    
    assert len(list(np.unique(np.array(indices_list)))) == 1, f"The classes are imbalanced!"
    
    cur_random_state = random.getstate()
    random.seed(randseed)
    random.shuffle(context_sample_indices)
    random.shuffle(query_sample_indices)
    random.setstate(cur_random_state)
    
    context_data = data[context_sample_indices]
    query_data = data[query_sample_indices]
    context_labels = labels[context_sample_indices]
    query_labels = labels[query_sample_indices]
    
    return context_data, context_labels, query_data, query_labels
        

def main():
    
    # argument parser
    args = parse_args()
    set_seed(args.seed)

    # model settings
    if args.parallel:
        num_gpus = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = [str(gpu_idx) for gpu_idx in range(num_gpus)]
    else:
        num_gpus = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.model_name == "mistral-8b":
        model = LLM(
            model=MODEL_PATHS[args.model_name],
            tensor_parallel_size=num_gpus,
            dtype=torch.bfloat16,
            max_model_len=MAX_TOKEN_LENGTH
        )
    else:
        model = LLM(
            model=MODEL_PATHS[args.model_name],
            tensor_parallel_size=num_gpus,
            dtype=torch.bfloat16,
        )
    
    tokenizer = model.get_tokenizer()

    # ========= data generation =========
    if args.task_mode == "linear_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_linear_classification_data(randseed=args.seed, num_classes=args.num_classes, num_samples_per_class=NUM_SAMPLES_PER_CLASS, num_feat=int(args.data_type[0]), precision=args.precision)
    elif args.task_mode == "circle_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_circle_classification_data(randseed=args.seed, num_samples_per_class=NUM_SAMPLES_PER_CLASS, noise=0.03, precision=args.precision)
    elif args.task_mode == "moon_classification":
        assert int(args.data_type[0]) > 1, "Classification task only supports 2D or higher dimensional data."
        data, labels = generate_moon_classification_data(randseed=args.seed, num_samples_per_class=NUM_SAMPLES_PER_CLASS, precision=args.precision)
    else:
        raise ValueError(f"Unrecognized task mode: {args.task_mode}.")
    
    # ========= ten trials =========
    for trial in range(41, 46):
        # sample different in-context data and query data from the same distribution for each trial
        context_data, context_labels, query_data, query_labels = data_sampling(randseed=trial, 
                                                                               data=data, 
                                                                               labels=labels, 
                                                                               num_context_per_cls=int(args.num_samples/args.num_classes), 
                                                                               num_query_per_cls=int(args.num_eval/args.num_classes))
        
        if args.inference_mode == "implicit":
            prompt_fn = implicit_ml_batch_prompt_generation
        elif args.inference_mode == "explicit":
            prompt_fn = explicit_ml_batch_prompt_generation
        else:
            raise ValueError(f"Unrecognized inference mode: {args.inference_mode}.")
        
        prompt_batch = prompt_fn(in_context_data=context_data, in_context_labels=context_labels, queries=query_data, method=args.prompt_mode)
        
        dataloader, dataDict = load_data(promptList=prompt_batch, tokenizer=tokenizer, batch_size=args.batch_size)

        # ========= run prediction =========
        predictions = []
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
                        )
                    )
                    
                    for idx, output in enumerate(outputs):
                        if args.num_responses == 1:
                            ans = output.outputs[0].text
                            predictions.append(ans)
                        else:
                            cur_rsp_list = []
                            for gen_idx, gen_output in enumerate(output.outputs):
                                ans = gen_output.text
                                cur_rsp_list.append(ans)
                            predictions.append(cur_rsp_list)
        
        # ========= save results =========
        res_dict = {
            "train_data": context_data,
            "train_labels": context_labels,
            "query_data": query_data,
            "predictions": predictions
        }
        
        save_root = os.path.join(ROOT_SAVE, "decision_boundary", "npy_results", 
                                 f"validation_util_ml_seed{trial}_{args.model_name}_{args.prompt_mode}_{args.inference_mode}_{args.num_classes}-classification_{args.task_mode}_{args.data_type}_tasks")
        
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
        filename = f"trial-{trial}.npy"
        np.save(os.path.join(save_root, filename), res_dict)
        print(f"Results successfully saved to {os.path.join(save_root, filename)}")