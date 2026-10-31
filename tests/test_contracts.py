from types import SimpleNamespace

import numpy as np
import pytest

from polyevo.core.individual import Individual
from polyevo.domain.connectivity import ConnectivitySpec, canonicalize
from polyevo.fitness.objectives import score_population, weighted_sum
from polyevo.predictors.builtin import Mock
from polyevo.predictors.cache import BudgetLedger, CachedOracle


def individual(smiles, value=0):
    return Individual(smiles, "fragment_seq", (), smiles, props={"p": value})


def test_pool_rescores_elites_and_fixed_fitness_is_hand_calculated():
    ctx = SimpleNamespace(cfg={"objectives": [
        {"property": "p", "direction": "maximize", "weight": 0.7, "normalize": "minmax_pool"},
    ]})
    elite = individual("*CC*", 10)
    score_population([individual("*C*", 0), elite], ctx, weighted_sum)
    assert elite.fitness == pytest.approx(0.7)
    score_population([elite, individual("*CCC*", 20)], ctx, weighted_sum)
    assert elite.fitness == 0
    assert elite.props == {"p": 10}
    obj = ctx.cfg["objectives"][0]
    obj.update(direction="minimize", normalize={"minmax_fixed": [0, 20]})
    score_population([elite], ctx, weighted_sum)
    assert elite.fitness == pytest.approx(0.35)
    assert elite.rank is None


def test_cache_survives_reopen_and_failures_are_cached(tmp_path):
    class Counting(Mock):
        count = 0

        def predict(self, smiles):
            self.count += len(smiles)
            return np.full(len(smiles), np.nan)

    predictor = Counting(property="p")
    path = tmp_path / "cache.sqlite"
    oracle = CachedOracle(path, [predictor], BudgetLedger(10))
    a = individual("*CC*")
    oracle.evaluate([a, a], 0)
    oracle.evaluate([individual("*CC*")], 1)
    assert predictor.count == oracle.budget.calls == 1
    assert np.isnan(a.props["p"])
    oracle.close()
    oracle = CachedOracle(path, [predictor], BudgetLedger(10))
    oracle.evaluate([individual("*CC*")], 2)
    assert predictor.count == oracle.budget.calls == 1
    oracle.close()


def test_multi_predictor_partial_budget_drops_incomplete_individual(tmp_path):
    oracle = CachedOracle(tmp_path / "cache.sqlite", [Mock(property="a"), Mock(property="b")], BudgetLedger(3))
    result = oracle.evaluate([individual("*C*"), individual("*CC*")], 0)
    assert [i.id for i in result] == ["*C*"]
    assert oracle.budget.calls == len(oracle.ledger_rows()) == 3
    assert set(result[0].props) >= {"a", "b"}
    oracle.close()


@pytest.mark.parametrize("smi", ["CC", "*CC", "**", "*C(*)*", "*CC*.[Na+]", "*=CC*"])
def test_invalid_connectivity(smi):
    assert canonicalize(smi, ConnectivitySpec()) is None


def test_canonical_identity_and_molecule_domain():
    spec = ConnectivitySpec()
    assert canonicalize("[16*]CC[16*]", spec) == canonicalize("*CC*", spec)
    assert canonicalize("[CH2:7](*)C*", spec) == "*CC*"
    assert canonicalize("CC", ConnectivitySpec("molecule", 0)) == "CC"


def test_nan_is_worst_and_constant_pool_is_finite():
    ctx = SimpleNamespace(cfg={"objectives": [
        {"property": "p", "direction": "maximize", "weight": 1, "normalize": "minmax_pool"}]})
    a, b = individual("*CC*", 2), individual("*CCC*", float("nan"))
    score_population([a, b], ctx, weighted_sum)
    assert a.fitness == 0.5
    assert b.fitness == float("-inf")
