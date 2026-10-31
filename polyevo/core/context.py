from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .individual import Archive


@dataclass(frozen=True)
class Context:
    cfg: Mapping[str, Any]
    rng: np.random.Generator
    generation: int
    spec: Any
    encodings: Mapping[str, Any]
    library: Any
    budget: Any
    archive: Archive

    @property
    def progress(self) -> float:
        generations = self.cfg["max_generations"]
        fraction = self.generation / generations if generations else 1.0
        budget = self.budget.limit
        spent = self.budget.calls / budget if budget else (1.0 if budget == 0 else 0.0)
        return min(1.0, max(fraction, spent))
