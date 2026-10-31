# PolyEvo

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

PolyEvo is a pluggable evolutionary optimization framework for linear polymer
repeat units and small molecules. The engine knows **stages and data contracts,
not algorithms** — every stage (initialization, selection, reproduction,
scoring, survival, diversity) is a callable resolved from a registry, and a
YAML config only names which callables to wire together.

Two rules hold everywhere:

1. **Properties are absolute; scores are relative.** A predicted property is
   cached forever by canonical SMILES and computed at most once per structure.
   Fitness, ranks and normalized scores are recomputed against the whole
   candidate pool every generation and are never reused across generations.
2. **Cheap checks come before expensive ones.** Structural validity and
   RDKit-computable constraints are enforced the moment an operator produces a
   candidate — invalid offspring never consume an oracle call.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
polyevo validate presets/mock_demo.yaml
polyevo run --config presets/mock_demo.yaml --out runs/mock_demo
polyevo list-ops --stage mutation
```

The demo evolves polymer repeat units against deterministic **Mock**
predictors and writes generation snapshots, lineage, an oracle ledger and a
summary CSV to `runs/mock_demo/`. Mock values are synthetic — they validate
the machinery, not materials performance.

## How it works

```
init → score(P₀) → stop? → select + reproduce + structural filter
     → oracle (new SMILES only) → score(pool) → survive → diversity → ↺
```

Everything behind a stage name lives in a registry — `tournament`, `family`,
`generational_elitist`, `weighted_sum`, `Mock`, the mutation and crossover
operators, stop conditions. Configs combine registered names; users can
`@register("select", "mine")` their own callables and reference them from
YAML (including `python: module:callable` entries).

## Encodings & fragment library

- **`fragment_seq`** — sequences of two-port BRICS fragments joined by
  compatibility rules. Fast, simple, used by `mock_demo`.
- **`fragment_graph`** — a graph of multi-port BRICS fragments. Original
  polymer chain ends (`*`) are preserved, remaining BRICS ports either connect
  pairwise by compatibility or become new chain ends, and a valid candidate
  always ends up with exactly two ends. Supports branched/side-chain fragment
  topologies.

Fragment libraries are built from polymer repeat-unit datasets:

```bash
polyevo build-library --input polymers.csv --out lib.parquet \
  --min-heavy-atoms 2 --allowed-elements C H N O S F Cl Br
```

This performs a full BRICS decomposition (original `*` ends kept distinct from
labeled BRICS ports), filters, deduplicates, and writes a Parquet library with
per-fragment port metadata and occurrence counts.

## Presets

| Preset | Purpose |
|---|---|
| `presets/mock_demo.yaml` | End-to-end smoke run, fixed normalization, monotone elite fitness |
| `presets/ga_mmpolymer.yaml` | Paper-shaped 3-objective GA (Tg ↑, εr ↓, SA ↓) over a 2,450-fragment library built from real data, with **Mock predictors as placeholders** |

The predictor layer is an interface (`Predictor` protocol: `name`,
`star_mode`, `predict(smiles) -> array`, NaN = failure). Swapping a real model
in means registering a callable and naming it in the config — no engine
changes.

## Data

`data/` and `数据/` contain polymer Tg and dielectric constant datasets
(raw, cleaned, and public supplements) plus the built fragment library, each
with local READMEs documenting provenance and cleaning steps. Use of the data
is subject to the terms of its original sources.

## Status & roadmap

Done: core engine and registries, both encodings, library builder, persistent
oracle cache + ledger with budgets, structural constraints, weighted-sum
scalar GA loop, Parquet/CSV outputs, 65 tests.

Not done: real predictors, `StarMode` transformations, property constraints
and penalty modes, mutation schedules, `--resume`, NSGA-II / Pareto,
MAP-Elites, analysis/reporting. Unimplemented config fields fail loudly
rather than being silently ignored.

## FAQ

**Why are properties cached forever but scores recomputed every generation?**
A predicted property depends only on the structure, so one SMILES should cost
one oracle call. A score depends on what it is compared against — elite
survivors must be re-measured against each new pool, or parents and children
would be ranked with different rulers.

**Why Mock predictors instead of a bundled model?**
The framework's job is the search loop, not training. Any callable matching
the `Predictor` protocol can be registered and referenced from YAML; the
included `Mock` exists so every pipeline feature is testable without a model.

**`fragment_seq` or `fragment_graph`?**
Use `fragment_seq` when your fragments are all two-port and a sequence is
enough. Use `fragment_graph` for real BRICS libraries, where fragments have
original chain ends, single ports, or three-plus ports and connection
topology must be tracked explicitly.

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
```

Design notes and PRD live in `docs/`.

## License

Code is released under the [MIT License](LICENSE). Datasets retain the terms
of their original sources.
