import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Setzt den Zufalls-Seed für vollständige Reproduzierbarkeit
    über alle relevanten Bibliotheken hinweg.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Seed gesetzt auf: {seed}")
