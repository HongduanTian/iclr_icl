from typing import List, Dict

__all__ = ["implicit_standard_batch_prompt_generation",
           "implicit_ml_batch_prompt_generation",
           "explicit_standard_batch_prompt_generation",
           "explicit_ml_batch_prompt_generation"]

ML_METHODS_OPTIONS = ["any", "decision_tree", "knn", "svm", "mlp", "linear_regression"]


def implicit_standard_batch_prompt_generation(in_context_data:List[List], in_context_labels:List[List], queries:List, boolInstruct:bool=True) -> List:
    
    prompt_batch = [implicit_standard_classification_task_prompt(
            in_context_data=in_context_data,
            in_context_labels=in_context_labels,
            query_example=query,
            boolInstruct=boolInstruct
        ) for query in queries]

    return prompt_batch


def implicit_ml_batch_prompt_generation(in_context_data:List[List], in_context_labels:List[List], queries:List, method:str="decision_tree", boolInstruct:bool=True) -> List:
    
    prompt_batch = []

    assert method in ML_METHODS_OPTIONS, f"Please choose machine learning algorithms from {ML_METHODS_OPTIONS}."
    
    prompt_batch = [implicit_ml_classification_task_prompt(
            in_context_data=in_context_data,
            in_context_labels=in_context_labels,
            query_example=query,
            method=method,
            boolInstruct=boolInstruct
        ) for query in queries]
    
    return prompt_batch


def explicit_standard_batch_prompt_generation(in_context_data:List[List], in_context_labels:List[List], queries:List, boolInstruct:bool=True) -> List:
    
    prompt_batch = [explicit_standard_classification_task_prompt(
            in_context_data=in_context_data,
            in_context_labels=in_context_labels,
            query_example=query,
            boolInstruct=boolInstruct
        ) for query in queries] 
    
    return prompt_batch


def explicit_ml_batch_prompt_generation(in_context_data:List[List], in_context_labels:List[List], queries:List, method:str="decision_tree", boolInstruct:bool=True) -> List:        
    
    prompt_batch = []

    assert method in ML_METHODS_OPTIONS, f"Please choose machine learning algorithms from {ML_METHODS_OPTIONS}."
    
    prompt_batch = [explicit_ml_classification_task_prompt(
            in_context_data=in_context_data,
            in_context_labels=in_context_labels,
            query_example=query,
            method=method,
            boolInstruct=boolInstruct
        ) for query in queries]
    
    return prompt_batch


def implicit_standard_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, boolInstruct:bool=True) -> str:
    
    system_message = f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}, and evaluate your confidence for the answer with a float number between {0.0} and {1.0}, where {0.0} means you are absolutely inconfident to your answer while {1.0} means you are absolutely confident to your answer. Please do not provide any reasoning process in your response."
    ic_data = ""
    for idx, item in enumerate(in_context_data):
        sub_items = ""
        for sub_item in item:
            sub_items += f"{sub_item} "
        ic_data += f"Input: {sub_items}\nLabel: {in_context_labels[idx]}\n"
    query_prompt = "What is the label for this input? And how confident you are to your answer?"
    
    query_items = ""
    for sub_item in query_example:
        query_items += f"{sub_item} "

    if boolInstruct:
        prompt = f"### Instructions:\n{system_message}\n### Input:\n{ic_data}\n{query_prompt}\nInput: {query_items}\n### Response:\nLabel: \nConfidence: "
    else:
        prompt = f"{system_message}\n{ic_data}\n{query_prompt}\nInput: {query_items}\nLabel: \nConfidence: "
    
    return prompt


def implicit_ml_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, method:str="decision_tree", boolInstruct:bool=True):
    
    method_names = {
        "decision_tree": "Decision Tree",
        "knn": "k-NN",
        "svm": "SVM",
        "mlp": "MLP",
        "linear_regression": "Linear Regression",
        "any": "machine learning",
    }
    
    if method == "any":
        method_key = "machine learning"
    elif method in ML_METHODS_OPTIONS:
        method_key = method_names[method]
    
    system_message = f"Given pairs of numbers and their labels, please apply {method_names[method]} method(s) to predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}, and evaluate your confidence for the answer with a float number between {0.0} and {1.0}, where {0.0} means you are absolutely inconfident to your answer while {1.0} means you are absolutely confident to your answer. Please do not provide any reasoning process in your response."
    ic_data = ""
    for idx, item in enumerate(in_context_data):
        sub_items = ""
        for sub_item in item:
            sub_items += f"{sub_item} "
        ic_data += f"Input: {sub_items}\nLabel: {in_context_labels[idx]}\n"
    query_prompt = "What is the label for this input? And how confident you are to your answer?"
    
    query_items = ""
    for sub_item in query_example:
        query_items += f"{sub_item} "
    
    if boolInstruct:
        prompt = f"### Instructions:\n{system_message}\n### Input:\n{ic_data}\n{query_prompt}\nInput: {query_items}\n### Response:\nLabel: \nConfidence: "
    else:
        prompt = f"{system_message}\n{ic_data}\n{query_prompt}\nInput: {query_items}\nLabel: \nConfidence: "
    
    return prompt


def explicit_standard_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, boolInstruct:bool=True) -> str:
    
    system_message = f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}, and evaluate your confidence for the answer with a float number between {0.0} and {1.0}, where {0.0} means you are absolutely inconfident to your answer while {1.0} means you are absolutely confident to your answer. Please provide detailed reasoning process in your response. Let's think step by step."
    ic_data = ""
    for idx, item in enumerate(in_context_data):
        sub_items = ""
        for sub_item in item:
            sub_items += f"{sub_item} "
        ic_data += f"Input: {sub_items}\nLabel: {in_context_labels[idx]}\n"
    query_prompt = "What is the label for this input? And how confident you are to your answer?"
    
    query_items = ""
    for sub_item in query_example:
        query_items += f"{sub_item} "

    if boolInstruct:
        prompt = f"### Instructions:\n{system_message}\n### Input:\n{ic_data}\n{query_prompt}\nInput: {query_items}\n### Response:\nLabel: \nConfidence: \nReasoning Process:"
    else:
        prompt = f"{system_message}\n{ic_data}\n{query_prompt}\nInput: {query_items}\nLabel: \nConfidence: \nReasoning Process: "
    
    return prompt


def explicit_ml_classification_task_prompt(in_context_data:List[List], in_context_labels:List, query_example:List, method:str="decision_tree", boolInstruct:bool=True):
    
    method_names = {
        "decision_tree": "Decision Tree",
        "knn": "k-NN",
        "svm": "SVM",
        "mlp": "MLP",
        "linear_regression": "Linear Regression",
        "any": "machine learning",
    }
    
    if method == "any":
        method_key = "machine learning"
    elif method in ML_METHODS_OPTIONS:
        method_key = method_names[method]
    
    system_message = f"Given pairs of numbers and their labels, please apply {method_names[method]} method(s) to predict the label for a new input pair of numbers based on the provided data.\nAnswer with only one of the labels {0} and {1}, and evaluate your confidence for the answer with a float number between {0.0} and {1.0}, where {0.0} means you are absolutely inconfident to your answer while {1.0} means you are absolutely confident to your answer. Please provide detailed reasoning process in your response. Let's think step by step."
    ic_data = ""
    for idx, item in enumerate(in_context_data):
        sub_items = ""
        for sub_item in item:
            sub_items += f"{sub_item} "
        ic_data += f"Input: {sub_items}\nLabel: {in_context_labels[idx]}\n"
    query_prompt = "What is the label for this input? And how confident you are to your answer?"
    
    query_items = ""
    for sub_item in query_example:
        query_items += f"{sub_item} "

    if boolInstruct:
        prompt = f"### Instructions:\n{system_message}\n### Input:\n{ic_data}\n{query_prompt}\nInput: {query_items}\n### Response:\nLabel: \nConfidence: \nReasoning Process:"
    else:
        prompt = f"{system_message}\n{ic_data}\n{query_prompt}\nInput: {query_items}\nLabel: \nConfidence: \nReasoning Process: "
    
    return prompt