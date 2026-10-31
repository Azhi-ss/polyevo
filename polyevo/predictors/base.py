from typing import Protocol

import numpy as np


class Predictor(Protocol):
    name: str
    unit: str | None
    star_mode: str
    batch_size: int | None

    def predict(self, smiles: list[str]) -> np.ndarray: ...
