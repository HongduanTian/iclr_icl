import os

from typing import List
from huggingface_hub import snapshot_download

MODELINFO = {
    "Qwen-2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
    "mistral-8b": "mistralai/Ministral-8B-Instruct-2410",
    "mistral-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

def downloadllms(models:List[str], dest_root_path:str):
    """
    Download necessary LLMs from huggingface.co.
    
    Args:
        models (List[str]): A list of model names required for this repo.
        dest_root_path (str): The destination path for the downloaded LLMs.
    """
    for model in models:
        model_dir = os.path.join(dest_root_path, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        if not os.listdir(model_dir):
            print(f"Downloading {model}...")
            snapshot_download(repo_id=MODELINFO[model], cache_dir=model_dir, resume_download=True)
            print(f"Successfully download {model}!")
        else:
            print(f"The model {model} already exists.")
            continue


if __name__ == "__main__":
    model_list = ["llama-3", "mistral-8b", "Qwen-2.5-7b", "llama-3.2-3b", "phi-4-3.8B", "Qwen-2.5-14B", "mistral-24b"]
    #cache_dir = "./cache"
    cache_dir = "/data1/models"
    downloadllms(models=model_list, dest_root_path=cache_dir)