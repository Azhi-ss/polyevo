from typing import Any, Callable, Protocol


class Encoding(Protocol):
    name: str
    crossover_ops: dict[str, Callable]
    mutation_ops: dict[str, Callable]

    def random(self, ctx) -> Any: ...
    def decode(self, genotype: Any) -> str | None: ...
    def distance(self, a: Any, b: Any) -> float: ...
