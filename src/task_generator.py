"""
This file is designed for generating tasks.
The types of tasks mainly include: linear, circle and moon.  
"""
import os
import random

import numpy as np

from sklearn.datasets import make_classification, make_circles, make_moons

NUM_SAMPLES = 10000

NUM_SAMPLES_PER_CLASS = 2000
# TODO: More extra tasks can be added in the future. 

def generate_linear_task(num_classes:int=2, num_samples:int=128,
                         num_feat:int=2, mode:str="train", precision:int=None, randseed:int=42):
    """
    Generate a linear classification task.
    
    Args:
        num_classes (int): The number of classes included in the task. Default:2 (Binary classification).
        num_samples (int): The number of in-context samples in the task. Default: 128.
        num_feat (int): The feature dimension. Default: 2.
        mode (str): The mode of the task. Options=['train', 'test']. Default: 'train'.
        precision (int): To preserve the precision of the data via `np.around`., Default to None.
        randseed (int): The random seed.
    
    Returns:
        data (np.narray): A set of data
        labels (np.narray): The corresponding labels of the data.
    """
    random.seed(randseed)
    
    if mode == "train":
        class_sep_range = [1.5, 2.]
    elif mode == "test":
        class_sep_range = [1., 1.4]
    else:
        raise ValueError("We only consider 'train' and 'test' modes.")
    
    class_sep = random.uniform(class_sep_range[0], class_sep_range[1])
    data, labels = make_classification(n_samples=num_classes*NUM_SAMPLES_PER_CLASS,   # sample a large batch of data
                                       n_features=num_feat, 
                                       n_informative=num_feat, 
                                       n_redundant=0, 
                                       n_repeated=0,
                                       flip_y=0,
                                       n_classes=num_classes,
                                       class_sep=class_sep,
                                       n_clusters_per_class=1,
                                       shuffle=True, 
                                       random_state=randseed)
    
    if precision is not None:
        data = np.around(data, precision)
    
    context_data, context_labels = task_sampling(data=data, labels=labels, num_context=num_samples)
    
    return context_data, context_labels


def generate_circle_task(num_samples:int=100, noise:float=None, mode:str="train", precision:int=None, randseed:int=42):
    """
    Generate a circle task.
    
    Args:
        num_samples (int, optional): The total number of samples. Defaults to 100.
        noise (float): Noise. Default to None.
        mode (str, optional): The mode of the task. Options=['train', 'test']. Defaults to "train".
        precision (int): To preserve the precision of the data via `np.around`., Default to None.
        randseed (int, optional): The random seed. Defaults to 42.
    
    Returns:
        data (np.narray): A set of data
        labels (np.narray): The corresponding labels of the data.
    """
    random.seed(randseed)
    
    if mode == "train":
        factor_range = [0.1, 0.4]
    elif mode == "test":
        factor_range = [0.5, 0.9]
    else:
        raise ValueError("We only consider 'train' and 'test' modes.")
    
    factor =  random.uniform(factor_range[0], factor_range[1])
    data, labels = make_circles(n_samples=NUM_SAMPLES, factor=factor, noise=noise, random_state=randseed)
    
    if precision is not None:
        data = np.around(data, precision)
    
    context_data, context_labels = task_sampling(data=data, labels=labels, num_context=num_samples)
    
    return context_data, context_labels
    

def generate_moon_task(num_samples:int=100, mode:str="train", precision:int=None, randseed:int=42):
    """
    Generate a moon task.

    Args:
        num_samples (int, optional): The total number of samples. Defaults to 100.
        mode (str, optional): The mode of the task. Options=['train', 'test']. Defaults to "train".
        precision (int): To preserve the precision of the data via `np.around`., Default to None.
        randseed (int, optional): The random seed. Defaults to 42.
    
    Returns:
        data (np.narray): A set of data
        labels (np.narray): The corresponding labels of the data.
    """
    random.seed(randseed)
    
    if mode == "train":
        noise_range = [0.05, 0.1]
    elif mode == "test":
        noise_range = [0.1, 0.2]
    else:
        raise ValueError("We only consider 'train' and 'test' modes.")
    
    noise = random.uniform(noise_range[0], noise_range[1])
    data, labels = make_moons(n_samples=NUM_SAMPLES, noise=noise, random_state=randseed)
    
    if precision is not None:
        data = np.around(data, precision)
    
    context_data, context_labels = task_sampling(data=data, labels=labels, num_context=num_samples)
    
    return context_data, context_labels


### =============================== Utils ===============================
def task_sampling(data:np.ndarray, labels:np.ndarray, num_context:int) -> np.ndarray:
    """
    Sampling in-context data from a set of data.

    Args:
        data (np.ndarray): Data.
        labels (np.ndarray): Labels.
        num_context (int): The number of in-context samples in a single task.

    Returns:
        np.ndarray: sampled data and labels.
    """
    unique_labels = np.unique(labels)
    assert num_context % len(unique_labels) == 0, f"{num_context} % {len(unique_labels)} should be zero."
    
    num_samples_per_cls = int(num_context / len(unique_labels))
    
    indices_list = []
    
    sample_indices = []
    for target_label in unique_labels:
        res = np.where(labels == target_label)[0]
        np.random.shuffle(res)
        indices_list.append(len(res[:num_samples_per_cls]))
        sample_indices += list(res[:num_samples_per_cls])
    
    assert len(list(np.unique(np.array(indices_list)))) == 1, f"The classes are imbalanced!"
    
    random.shuffle(sample_indices)
    
    context_data = data[sample_indices]
    context_labels = labels[sample_indices]
    
    
    return context_data, context_labels



if __name__ == "__main__":
    from args import parse_args
    from utils import set_seed
    import matplotlib.pyplot as plt
    
    #set_seed(42)
    args = parse_args()
    set_seed(args.seed)
    data, labels = generate_linear_task(num_classes=args.num_classes, mode="train", num_feat=int(args.data_type[0]), num_samples=args.num_samples, precision=args.precision, randseed=args.seed)
    
    plt.figure()
    
    plt.scatter(data[:, 0], data[:, 1])
    
    plt.savefig("test_data_distribution.png")