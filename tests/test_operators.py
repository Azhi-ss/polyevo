from types import SimpleNamespace

import pytest

from polyevo.core.individual import Archive, Individual
from polyevo.operators.builtin import generational_elitist


@pytest.mark.parametrize("mode,expected", [("replace", 4), ("append", 5)])
def test_survival_size_elites_and_archive(mode, expected):
    def make(name, fitness):
        return Individual(name, "test", (), name, fitness=fitness, scores={"x": fitness})

    parents = [make(f"parent{i}", i) for i in range(4)]
    children = [make(f"child{i}", i / 10) for i in range(8)]
    ctx = SimpleNamespace(cfg={"population_size": 4}, archive=Archive())
    survivors, archive = generational_elitist(parents + children, parents, ctx, ratio=0.25, mode=mode)
    assert len(survivors) == expected
    assert survivors[0].id == "parent3"
    assert archive.kind == "best_k"
    assert len(archive.individuals) == 1
    assert archive.individuals[0].fitness is None
    assert survivors[0].fitness == 3


def test_survival_fills_shortfall_without_duplicates():
    parents = [Individual(str(i), "test", (), str(i), fitness=i) for i in range(4)]
    ctx = SimpleNamespace(cfg={"population_size": 4}, archive=Archive())
    survivors, _ = generational_elitist(parents, parents, ctx)
    assert len(survivors) == len({i.id for i in survivors}) == 4
