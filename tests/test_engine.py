import csv
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq
import pytest
import yaml

from polyevo import register, run_ga
from polyevo.cli import main
from polyevo.config import validate


def test_run_is_reproducible_and_archive_does_not_cache_scores(config, tmp_path):
    a = run_ga(config, tmp_path / "a")
    b = run_ga(config, tmp_path / "b")
    assert a.generation == b.generation == 2
    assert len(a.population) == 8
    assert a.oracle_calls == b.oracle_calls
    for relative in ("summary.csv", "lineage.jsonl", "oracle_ledger.csv", "archive.parquet",
                     "generations/gen_002.parquet"):
        assert (a.out / relative).read_bytes() == (b.out / relative).read_bytes()
    assert all(not i.scores and i.fitness is None for i in a.archive.individuals)
    assert all(i.age == a.generation - i.gen_born for i in a.population)
    with pytest.raises(ValueError, match="not empty"):
        run_ga(config, a.out)


@pytest.mark.parametrize("budget", [0, 3, 19])
def test_budget_ends_cleanly_and_does_not_overspend(config, tmp_path, budget):
    config["budget"] = budget
    state = run_ga(config, tmp_path / "run")
    assert state.stop_reason == "budget_exhausted"
    assert state.oracle_calls == budget
    with (state.out / "oracle_ledger.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == budget
    assert (state.out / "run.json").exists()
    assert (state.out / "archive.parquet").exists()


def test_structural_rejection_never_calls_oracle(config, tmp_path):
    config["structural_constraints"] = [{"property": "MolWt", "range": [1, 2]}]
    config["max_attempts"] = 1
    with pytest.raises(ValueError, match="Initialization produced only 0"):
        run_ga(config, tmp_path / "invalid")
    with sqlite3.connect(tmp_path / "invalid" / "predictor_cache.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM ledger").fetchone()[0] == 0


def test_invalid_offspring_is_filtered_before_oracle(config, tmp_path):
    from polyevo.encoding.fragment_seq import FragmentSequence

    @register("encoding", "test_invalid_mutation")
    class InvalidMutation(FragmentSequence):
        def __init__(self, library, spec):
            super().__init__(library, spec)
            self.mutation_ops["invalid"] = lambda g, ctx: ("CC",)

    config["encoding"] = "test_invalid_mutation"
    config["reproduce"].update(crossover_probability=0, mutations=[{"name": "invalid", "probability": 1}])
    config["max_attempts"] = 2
    state = run_ga(config, tmp_path / "invalid_children")
    assert state.stop_reason == "search_exhausted"
    assert state.oracle_calls == config["population_size"] * 2


def test_registered_hook_and_callback(config, tmp_path):
    @register("select", "test_first")
    def first(population, ctx):
        return population[0]

    class Stop:
        def on_oracle_end(self, ctx, population):
            assert population
            assert all("mock_a" in i.props and "mock_b" in i.props for i in population)

        def on_generation_end(self, ctx, population):
            with pytest.raises(TypeError):
                ctx.cfg["reproduce"]["children"] = 999
            if ctx.generation == 1:
                return "stop"

    config["select"] = "test_first"
    state = run_ga(config, tmp_path / "custom", callbacks=[Stop()])
    assert state.stop_reason == "callback"
    assert state.generation == 1


@pytest.mark.parametrize("patch,match", [
    ({"aggregate": "pareto"}, "Incompatible"),
    ({"select": {"name": "tournament", "typo": 2}}, "unexpected keyword"),
    ({"survive": {"name": "generational_elitist", "ratio": 1.5}}, "Invalid elite"),
    ({"population_size": 0}, "population_size"),
    ({"property_constraints": []}, "Unsupported config"),
    ({"predictors": [{"name": "Mock", "star_mode": "cap_H"}]}, "star_mode"),
])
def test_config_rejects_unimplemented_or_invalid_options(config, patch, match):
    config.update(patch)
    with pytest.raises(ValueError, match=match):
        validate(config)


def test_cli(config, tmp_path, capsys):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    assert main(["validate", str(path)]) == 0
    assert "max_attempts" in capsys.readouterr().out
    assert main(["list-ops", "--stage", "mutation"]) == 0
    assert "substitute" in capsys.readouterr().out
    assert main(["run", "--config", str(path), "--out", str(tmp_path / "cli")]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == 2


def test_twenty_generation_demo_is_monotone(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / "presets/mock_demo.yaml").read_text())
    state = run_ga(config, tmp_path / "demo")
    assert state.generation == 20
    assert state.stop_reason == "max_generations"
    with (state.out / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 21  # gen 0 initialization + 20 reproduction steps
    best = [float(row["fitness_best"]) for row in rows]
    assert all(b >= a for a, b in zip(best, best[1:]))
    assert all(int(row["pop_size"]) == config["population_size"] for row in rows)
    table = pq.read_table(state.out / "generations/gen_020.parquet")
    assert table.num_rows == config["population_size"]
