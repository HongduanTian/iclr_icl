import os
import random
import torch
import yaml

import numpy as np
from fastchat.model import get_conversation_template
import yaml

def set_seed(seed_id:int=42):
    """
    Set random seed for the experiment for reproduction. Default: 42.
    """
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_id)
    
def load_config_from_yaml(config_path:str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config