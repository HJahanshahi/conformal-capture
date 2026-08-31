"""
Tiny helper to set process-wide determinism for tests and benchmarks.

Call `set_deterministic(seed)` at the top of any script BEFORE importing
numpy, torch, or any cap_control module that touches either.
"""
import os
import random


def set_deterministic(seed: int = 0):
    """
    Make numpy, python-random, and torch (GPU + CPU) deterministic.

    Note: torch\'s deterministic mode is set inside cap_control.prediction.
    upn_predictor as well -- this helper is for callers that need to lock
    down additional entropy sources (numpy\'s global RNG, random.random()).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
