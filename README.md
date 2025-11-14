# Exploring the discriminative capability of LLMs in in-context learning

This is an initial repository for the paper "Exploring the discriminative capability of LLMs in in-context learning" submiited to ICLR 2025.

## Preparation

Install the dependencies via
`pip install -r requirements.txt`.

## Reproduction

1. Run linear/circle/moon classification tasks via **conventional machine leanring algorithms**:
- python run_conv_methods.py --gpu_id=0 --task_mode=linear_classification
- python run_conv_methods.py --gpu_id=0 --task_mode=circle_classification
- python run_conv_methods.py --gpu_id=0 --task_mode=moon_classification

2. Run linear/circle/moon classification tasks via **Llama-3-8B** with **standard prompts**:
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=linear_classification --prompt_mode=standard
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=circle_classification --prompt_mode=standard
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=moon_classification --prompt_mode=standard

3. Run linear/circle/moon classification tasks via **Llama-3-8B** with **ML-only prompts**:
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=linear_classification --prompt_mode=ml --ml_alg=None
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=circle_classification --prompt_mode=ml --ml_alg=None
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=moon_classification --prompt_mode=ml --ml_alg=None

3. Run linear/circle/moon classification tasks via **Llama-3-8B** with **specialized machine leanring algorithms** (e.g.,  decision tree):
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=linear_classification --prompt_mode=ml --ml_alg=DecisionTree
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=circle_classification --prompt_mode=ml --ml_alg=DecisionTree
- python run_conv_methods.py --gpu_id=0 --model_name=llama-3 --task_mode=moon_classification --prompt_mode=ml --ml_alg=DecisionTree

4. Simulate the behavior of **Llama-3-8B** with **standard prompts**:
- python run_hybrid_convs.py --gpu_id=0 --model_name=llama-3 --task_mode=linear_classification --prob_mode=standard