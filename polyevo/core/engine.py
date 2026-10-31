from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from polyevo.callbacks.writer import RunWriter
from polyevo.config import validate
from polyevo.core.context import Context
from polyevo.core.individual import Archive, Individual
from polyevo.core.registry import resolve
from polyevo.domain.connectivity import ConnectivitySpec, canonicalize
from polyevo.encoding.fragment_seq import FragmentLibrary
from polyevo.predictors.cache import BudgetLedger, CachedOracle


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass
class RunState:
    population: list[Individual]
    archive: Archive
    generation: int
    oracle_calls: int
    stop_reason: str
    out: Path


def run_ga(config, out, *, callbacks=()):
    """Run the configured stages. Callback objects may implement on_* methods."""
    cfg = validate(config)
    spec = ConnectivitySpec(**cfg["domain"])
    library = FragmentLibrary.from_entries(cfg["library"])
    encoding = resolve("encoding", cfg["encoding"])(library=library, spec=spec)
    predictors = [resolve("predictor", p)() for p in cfg["predictors"]]
    hooks = {stage: resolve(stage, cfg[stage]) for stage in
             ("init", "score", "select", "reproduce", "aggregate", "survive", "diversity")}
    stops = [resolve("stop", s) for s in cfg["stop"]]
    ctx = Context(_freeze(cfg), np.random.default_rng(cfg["seed"]), 0, spec,
                  MappingProxyType({encoding.name: encoding}), library,
                  BudgetLedger(cfg["budget"]), Archive())
    writer = RunWriter(out, cfg, library)
    oracle = CachedOracle(writer.path / "predictor_cache.sqlite", predictors, ctx.budget)
    requested_stop = False
    population = []

    def notify(event, current=None):
        nonlocal requested_stop
        for callback in callbacks:
            handler = getattr(callback, event, None)
            if handler is not None and handler(ctx, population if current is None else current) == "stop":
                requested_stop = True

    def make_candidate(name, genotype, parents, ops):
        smiles = canonicalize(ctx.encodings[name].decode(genotype), spec,
                              ctx.cfg["structural_constraints"])
        if smiles is None:
            return None
        return Individual(smiles, name, genotype, smiles,
                          parents=parents, ops=ops, gen_born=ctx.generation)

    def evaluate(candidates):
        evaluated = oracle.evaluate(candidates, ctx.generation)
        writer.lineage(evaluated)
        notify("on_oracle_end", evaluated)
        return evaluated

    reason = ""
    try:
        notify("on_start")
        if not requested_stop:
            notify("on_generation_begin")
        if not requested_stop:
            population = evaluate(hooks["init"](ctx, make_candidate))
            population = hooks["score"](population, ctx, hooks["aggregate"])
            population, archive = hooks["survive"](population, population, ctx)
            ctx = replace(ctx, archive=archive)
            population = hooks["diversity"](population, ctx)
        while True:
            writer.generation(ctx, population, oracle)
            notify("on_generation_end")
            if requested_stop:
                reason = "callback"
            elif ctx.budget.exhausted:
                reason = "budget_exhausted"
            elif not population:
                reason = "empty_population"
            elif ctx.generation >= cfg["max_generations"]:
                reason = "max_generations"
            elif any(stop(ctx, population) for stop in stops):
                reason = "stop_hook"
            if reason:
                break
            ctx = replace(ctx, generation=ctx.generation + 1)
            notify("on_generation_begin")
            if requested_stop:
                ctx = replace(ctx, generation=ctx.generation - 1)
                reason = "callback"
                break
            offspring = hooks["reproduce"](population, ctx, hooks["select"], make_candidate)
            if not offspring:
                ctx = replace(ctx, generation=ctx.generation - 1)
                reason = "search_exhausted"
                break
            offspring = evaluate(offspring)
            candidates = hooks["score"](population + offspring, ctx, hooks["aggregate"])
            for individual in candidates:
                individual.age = ctx.generation - individual.gen_born
            population, archive = hooks["survive"](candidates, population, ctx)
            ctx = replace(ctx, archive=archive)
            population = hooks["diversity"](population, ctx)
        notify("on_stop")
        writer.stop(reason, ctx)
    except Exception:
        writer.stop("error", ctx)
        raise
    finally:
        oracle.close()
    return RunState(population, ctx.archive, ctx.generation, ctx.budget.calls, reason, writer.path)
