from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)
