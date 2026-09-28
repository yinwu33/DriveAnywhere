"""Seed all random generators used by scene optimization and frame sampling."""
import random

import numpy as np
import torch


def seed_scene_optimization(seed: int) -> None:
    """Fix Python frame sampling, NumPy and Torch; CUDA kernels may still vary."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
