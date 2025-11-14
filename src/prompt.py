"""
This file is designed for prompt generation.
"""
from typing import List

def naive_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, algorithm:str=None):
    
    prompt = ""
    
    ### general info
    prompt += f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}:\n\n"
    
    ### Add context data
    for idx, item in enumerate(in_context_data):
        prompt += f"Input: {item[0]} {item[1]}\nLabel: {in_context_labels[idx]}\n"
    
    ### Query data
    prompt += f"\nWhat is the label for this input? \nInput: {query_example[0]} {query_example[1]}\nLabel:"
    prompt += "\nPlease directly provide the answer. Do not give any analysis."
    
    return prompt


def analysis_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, algorithm:str=None):
    
    prompt = ""
    
    ### general info
    prompt += f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}:\n\n"
    
    ### Add context data
    for idx, item in enumerate(in_context_data):
        prompt += f"Input: {item[0]} {item[1]}\nLabel: {in_context_labels[idx]}\n"
    
    ### Query data
    prompt += f"\nWhat is the label for this input? \nInput: {query_example[0]} {query_example[1]}\nLabel:"
    prompt += "\nPlease directly provide the answer. Do not give any analysis."
    
    return prompt


def ml_model_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, algorithm:str=None):
    
    mlalgList = ["DecisionTree", "KNN", "SVM", "MLP", "LinearRegression"]
    assert algorithm in mlalgList, print(f"Unrecognized machine learning methods! Your algorithm should be selected from {mlalgList}!")
    
    if algorithm == "DecisionTree":
        alg = "Decision Tree"
    elif algorithm == "KNN":
        alg = "K-NN"
    elif algorithm == "LinearRegression":
        alg = "Linear Regression"
    else:
        alg = algorithm
    
    prompt = ""
    
    ### general info
    prompt += f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}:\n\n"
    
    ### Add context data
    for idx, item in enumerate(in_context_data):
        prompt += f"Input: {item[0]} {item[1]}\nLabel: {in_context_labels[idx]}\n"
    
    ### Query data
    prompt += f"\nWhat is the label for this input? \nInput: {query_example[0]} {query_example[1]}\nLabel:"
    
    if algorithm is None:
        prompt += f"\nYour answer must be based on machine learning algorithms or models!"
    else:
        prompt += f"\nYour answer must be based on the running result of {alg} algorithm/model!" 
        
    prompt += "\nPlease directly provide the answer. Do not give any analysis."
    
    return prompt


def nonml_model_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, algorithm:str=None):
    
    prompt = ""
    
    ### general info
    prompt += f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}:\n\n"
    
    ### Add context data
    for idx, item in enumerate(in_context_data):
        prompt += f"Input: {item[0]} {item[1]}\nLabel: {in_context_labels[idx]}\n"
    
    ### Query data
    prompt += f"\nWhat is the label for this input? \nInput: {query_example[0]} {query_example[1]}\nLabel:"
    prompt += f"\nMachine learning algorithms or models are not allowed in this task!"
    prompt += "\nPlease directly provide the answer. Do not give any analysis."
    
    return prompt


def batch_prompt_generation(in_context_data:List[List], in_context_labels:List[List], queries:List, prompt_mode:str, algorithm:str=None):
    
    prompt_batch = []
    
    if prompt_mode == "standard":
        prompt_fn = naive_classification_task_prompt
    elif prompt_mode == "ml":
        prompt_fn = ml_model_classification_task_prompt
    elif prompt_mode == "non_ml":
        prompt_fn = nonml_model_classification_task_prompt
    elif prompt_mode == "analysis":
        prompt_fn = analysis_classification_task_prompt
    else:
        raise TypeError("Unrecognized prompt mode.")
    
    if algorithm is not None:
        assert prompt_mode == "ml"
    
    for query in queries:
        prompt_batch.append(prompt_fn(
            in_context_data=in_context_data,
            in_context_labels=in_context_labels,
            query_example=query,
            algorithm=algorithm
        ))
    return prompt_batch