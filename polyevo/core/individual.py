from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    id: str
    encoding: str
    genotype: Any
    smiles: str
    props: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    violation: float = 0.0
    fitness: float | None = None
    rank: int | None = None
    crowding: float | None = None
    parents: tuple[str, ...] = ()
    ops: list[str] = field(default_factory=list)
    gen_born: int = 0
    age: int = 0


@dataclass
class Archive:
    kind: str = "best_k"
    individuals: list[Individual] = field(default_factory=list)

    def update(self, population: list[Individual], k: int) -> None:
        # Compare only this generation's scores; archive snapshots contain no scores.
        from copy import deepcopy

        self.individuals = deepcopy(population[:k])
        for individual in self.individuals:
            individual.scores.clear()
            individual.penalties.clear()
            individual.violation = 0.0
            individual.fitness = individual.rank = individual.crowding = None
