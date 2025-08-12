from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set global random seeds (stub).

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # TODO: Add numpy/torch seeds when deps available
