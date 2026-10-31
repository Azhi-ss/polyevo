import hashlib

import numpy as np

from polyevo.core.registry import register


@register("predictor", "Mock")
class Mock:
    """Deterministic synthetic values, not a physical-property model."""

    def __init__(self, property="mock", unit=None, star_mode="keep", batch_size=None):
        if star_mode != "keep":
            raise ValueError("M1 only supports star_mode: keep")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        self.name, self.unit = property, unit
        self.star_mode, self.batch_size = star_mode, batch_size

    def predict(self, smiles):
        return np.array([
            int.from_bytes(hashlib.sha256(f"{self.name}:{s}".encode()).digest()[:8], "big")
            / (2**64 - 1) for s in smiles
        ])
