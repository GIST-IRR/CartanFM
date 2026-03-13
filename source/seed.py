import random
from datetime import datetime

import torch
import numpy as np


"""Seed management utilities for reproducible experiments."""


random.seed(datetime.now())


def set_seed(args):
    """Set Python, NumPy, and PyTorch seeds from parsed arguments."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def manual_seed(seed):
    """Set Python, NumPy, and PyTorch seeds from a direct integer value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
