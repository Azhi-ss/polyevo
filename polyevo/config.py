from copy import deepcopy
import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
import yaml

from polyevo.core.registry import resolve
from polyevo.domain.connectivity import ConnectivitySpec
from polyevo.encoding.fragment_seq import FragmentLibrary
# Import built-ins once so @register also remains available to user plugins.
from polyevo.fitness import objectives as _objectives
from polyevo.operators import builtin as _operators
from polyevo.predictors import builtin as _predictors
from polyevo.encoding import fragment_graph as _fragment_graph


DEFAULTS = {
    "seed": 42, "population_size": 24, "max_generations": 20,
    "max_attempts": 30, "budget": None,
    "domain": {"name": "polymer_linear", "n_attachment": 2},
    "encoding": {"name": "fragment_seq", "n_fragments": [2, 4]},
    "init": "random_assemble", "score": "objectives",
    "select": {"name": "tournament", "k": 3},
    "reproduce": {"name": "family", "families": 10, "children": 4,
                  "crossover": "brics_single_point", "crossover_probability": 1.0,
                  "mutations": [{"name": "substitute", "probability": 0.25}]},
    "aggregate": "weighted_sum",
    "survive": {"name": "generational_elitist", "ratio": 0.1, "mode": "replace"},
    "diversity": "none", "stop": ["max_generations", "budget_exhausted"],
    "structural_constraints": [],
}


def _name(spec):
    return spec.get("name") if isinstance(spec, dict) else spec


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def _bounds(value, label):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must be [lower, upper]")
    for item in value:
        _number(item, label)
    if value[0] >= value[1]:
        raise ValueError(f"{label}: lower must be less than upper")


def validate(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    unknown = set(config) - (set(DEFAULTS) | {"library", "predictors", "objectives"})
    if unknown:
        raise ValueError(f"Unsupported config fields: {', '.join(sorted(unknown))}")
    cfg = deepcopy(DEFAULTS)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            if "name" not in value or value.get("name") == cfg[key].get("name"):
                cfg[key].update(deepcopy(value))
            else:
                cfg[key] = deepcopy(value)
        else:
            cfg[key] = deepcopy(value)
    for key in ("population_size", "max_attempts"):
        _integer(cfg[key], key)
    for key in ("seed", "max_generations"):
        _integer(cfg[key], key, 0)
    if cfg["budget"] is not None:
        _integer(cfg["budget"], "budget", 0)
    try:
        spec = ConnectivitySpec(**cfg["domain"])
    except TypeError as exc:
        raise ValueError(f"Invalid domain: {exc}") from exc
    library_cfg = cfg.get("library")
    if isinstance(library_cfg, list):
        library = FragmentLibrary.from_entries(library_cfg)
    elif isinstance(library_cfg, dict) and "path" in library_cfg:
        import pyarrow.parquet as pq
        table = pq.read_table(library_cfg["path"])
        rows = table.to_pylist()
        entries = [{"smiles": r["smiles"], "weight": r.get("weight", 1.0)} for r in rows]
        library = FragmentLibrary.from_entries(entries, require_two_brics=False)
    else:
        raise ValueError("library must be a list of fragments or {path: file.parquet}")
    encoding = resolve("encoding", cfg["encoding"])(library=library, spec=spec)
    if not isinstance(cfg.get("predictors"), list) or not cfg["predictors"]:
        raise ValueError("At least one predictor is required")
    predictors = [resolve("predictor", p)() for p in cfg["predictors"]]
    names = [p.name for p in predictors]
    if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
        raise ValueError("Predictor property names must be nonempty and unique")
    for predictor in predictors:
        if predictor.star_mode != "keep":
            raise ValueError("M1 only supports star_mode: keep")
    if not isinstance(cfg.get("objectives"), list) or not cfg["objectives"]:
        raise ValueError("At least one objective is required")
    properties = set()
    for obj in cfg["objectives"]:
        if not isinstance(obj, dict) or set(obj) - {"property", "direction", "weight", "normalize"}:
            raise ValueError("Invalid objective fields")
        if obj.get("property") not in names or obj["property"] in properties:
            raise ValueError("Objective properties must be unique and have a predictor")
        properties.add(obj["property"])
        obj.setdefault("direction", "maximize")
        obj.setdefault("weight", 1.0)
        obj.setdefault("normalize", "minmax_pool")
        if obj["direction"] not in ("maximize", "minimize"):
            raise ValueError("M1 supports maximize/minimize directions only")
        _number(obj["weight"], "objective weight")
        if obj["weight"] < 0:
            raise ValueError("Objective weight must be nonnegative")
        norm = obj["normalize"]
        if isinstance(norm, dict) and set(norm) == {"minmax_fixed"}:
            _bounds(norm["minmax_fixed"], "minmax_fixed")
        elif not isinstance(norm, str) or norm not in ("minmax_pool", "none"):
            raise ValueError("M1 normalization: minmax_pool, {minmax_fixed: [lo, hi]}, none")
    if not any(o["weight"] > 0 for o in cfg["objectives"]):
        raise ValueError("At least one objective weight must be positive")
    if _name(cfg["aggregate"]) == "pareto" and _name(cfg["select"]) == "tournament":
        raise ValueError("Incompatible configuration: pareto + tournament")
    for stage in ("init", "score", "select", "reproduce", "aggregate", "survive", "diversity"):
        resolve(stage, cfg[stage])
    if not isinstance(cfg["stop"], list) or not cfg["stop"]:
        raise ValueError("stop must be a nonempty list")
    for stop in cfg["stop"]:
        resolve("stop", stop)
    selection = cfg["select"]
    if isinstance(selection, dict) and selection["name"] == "tournament":
        _integer(selection.get("k", 3), "tournament.k")
    reproduction = cfg["reproduce"]
    if isinstance(reproduction, dict) and reproduction["name"] == "family":
        for key in ("families", "children"):
            _integer(reproduction[key], f"family.{key}")
        if reproduction["crossover"] not in encoding.crossover_ops:
            raise ValueError("Unknown crossover for configured encoding")
        p = reproduction["crossover_probability"]
        _number(p, "crossover_probability")
        if not 0 <= p <= 1:
            raise ValueError("crossover_probability must be in [0, 1]")
        for mutation in reproduction["mutations"]:
            if set(mutation) != {"name", "probability"} or mutation["name"] not in encoding.mutation_ops:
                raise ValueError("Invalid mutation; expected {name, probability}")
            _number(mutation["probability"], "mutation probability")
            if not 0 <= mutation["probability"] <= 1:
                raise ValueError("Mutation probability must be in [0, 1]")
    survival = cfg["survive"]
    if isinstance(survival, dict) and survival["name"] == "generational_elitist":
        _number(survival["ratio"], "elite ratio")
        if not 0 <= survival["ratio"] <= 1 or survival["mode"] not in ("replace", "append"):
            raise ValueError("Invalid elite ratio or mode")
    for rule in cfg["structural_constraints"]:
        if rule.get("on_fail", "resample") != "resample":
            raise ValueError("M1 supports structural on_fail: resample only")
        if "smarts_forbidden" in rule:
            if set(rule) - {"smarts_forbidden", "on_fail"}:
                raise ValueError("Invalid SMARTS constraint fields")
            if not isinstance(rule["smarts_forbidden"], list) or any(
                    Chem.MolFromSmarts(s) is None for s in rule["smarts_forbidden"]):
                raise ValueError("Invalid forbidden SMARTS")
        else:
            if set(rule) - {"property", "range", "on_fail"} or rule.get("property") not in dict(Descriptors.descList):
                raise ValueError("Unknown structural descriptor")
            _bounds(rule.get("range"), "structural range")
    return cfg


def load_config(path):
    with Path(path).open() as stream:
        return validate(yaml.safe_load(stream))
