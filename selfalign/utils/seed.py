from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for reproducibility.

    - Always seeds Python's 'random' and sets PYTHONHASHSEED.
    - Attempts to seed NumPy if available.
    - Attempts to seed PyTorch if available, but Phase 2 does NOT require torch.
      Import torch inside this function to avoid import-time hard dependency.
    """
    # Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy (optional)
    try:
        import numpy as np  # type: ignore

        try:
            np.random.seed(seed)
        except Exception:
            pass
    except Exception:
        pass

    # PyTorch (optional)
    try:
        import torch  # type: ignore

        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Make CUDA/CuDNN deterministic where applicable
            if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            # Do not fail if torch seeding fails
            pass
    except Exception:
        # Torch not installed; ignore
        pass
