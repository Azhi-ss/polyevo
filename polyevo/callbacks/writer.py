import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


_SCHEMA = pa.schema([
    ("id", pa.string()), ("encoding", pa.string()), ("genotype", pa.string()),
    ("smiles", pa.string()), ("props", pa.string()), ("scores", pa.string()),
    ("penalties", pa.string()), ("violation", pa.float64()), ("fitness", pa.float64()),
    ("rank", pa.int64()), ("crowding", pa.float64()),
    ("parents", pa.list_(pa.string())), ("ops", pa.list_(pa.string())),
    ("gen_born", pa.int64()), ("age", pa.int64()),
])


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _rows(population):
    rows = []
    for individual in population:
        row = asdict(individual)
        for key in ("genotype", "props", "scores", "penalties"):
            row[key] = json.dumps(_json_value(row[key]), sort_keys=True, default=repr, allow_nan=False)
        rows.append(row)
    return rows


def write_population(path, population):
    pq.write_table(pa.Table.from_pylist(_rows(population), schema=_SCHEMA), path)


class RunWriter:
    def __init__(self, directory, config, library):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise ValueError(f"Output directory is not empty: {self.path}; use a new directory")
        (self.path / "generations").mkdir()
        (self.path / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        pq.write_table(pa.table({"smiles": library.smiles, "weight": library.weights}),
                       self.path / "fragment_library.parquet")
        self.summary = []
        self.seen = set()

    def generation(self, ctx, population, oracle):
        write_population(self.path / "generations" / f"gen_{ctx.generation:03d}.parquet", population)
        row = {"generation": ctx.generation, "pop_size": len(population),
               "oracle_calls": ctx.budget.calls}
        fitness = np.array([i.fitness for i in population], dtype=float)
        finite = fitness[np.isfinite(fitness)]
        row.update(fitness_best=float(np.max(fitness)) if len(fitness) else float("nan"),
                   fitness_mean=float(np.mean(finite)) if len(finite) else float("nan"),
                   fitness_std=float(np.std(finite)) if len(finite) else float("nan"))
        for objective in ctx.cfg["objectives"]:
            prop = objective["property"]
            values = np.array([i.props[prop] for i in population])
            values = values[np.isfinite(values)]
            best = np.max if objective["direction"] == "maximize" else np.min
            for name, fn in (("mean", np.mean), ("best", best), ("std", np.std)):
                row[f"{prop}_{name}"] = float(fn(values)) if len(values) else float("nan")
        self.summary.append(row)
        with (self.path / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.summary)
        write_population(self.path / "archive.parquet", ctx.archive.individuals)
        with (self.path / "oracle_ledger.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["call_index", "predictor", "predictor_call_index", "smiles", "value", "generation"])
            writer.writerows(oracle.ledger_rows())

    def lineage(self, individuals):
        with (self.path / "lineage.jsonl").open("a") as stream:
            for individual in individuals:
                if individual.id in self.seen:
                    continue
                self.seen.add(individual.id)
                stream.write(json.dumps({"id": individual.id, "parents": individual.parents,
                                         "ops": individual.ops, "gen_born": individual.gen_born},
                                        sort_keys=True) + "\n")

    def stop(self, reason, ctx):
        (self.path / "run.json").write_text(json.dumps({
            "stop_reason": reason, "generation": ctx.generation,
            "oracle_calls": ctx.budget.calls,
        }, indent=2) + "\n")
