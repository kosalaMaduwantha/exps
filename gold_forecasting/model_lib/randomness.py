"""Small reproducibility helper; importing it never imports model-training code."""

import random
import numpy as np


def seed_everything(seed: int) -> None:
    """Reset Python/NumPy random state for an explicitly invoked notebook fit."""
    random.seed(seed)
    np.random.seed(seed)
