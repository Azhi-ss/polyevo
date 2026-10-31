import numpy as np

from polyevo.core.registry import register


@register("aggregate", "weighted_sum")
def weighted_sum(scores, objectives):
    return sum(scores[obj["property"]] * obj["weight"] for obj in objectives)


@register("score", "objectives")
def score_population(population, ctx, aggregate):
    objectives = ctx.cfg["objectives"]
    for individual in population:
        individual.scores.clear()
        individual.penalties.clear()
        individual.violation = 0.0
        individual.fitness = individual.rank = individual.crowding = None
    for obj in objectives:
        prop = obj["property"]
        raw = np.array([i.props[prop] for i in population], dtype=float)
        valid = np.isfinite(raw)
        scores = np.zeros(len(raw))
        values = raw[valid]
        if len(values):
            norm = obj["normalize"]
            if norm == "none":
                normalized = values.copy()
                if obj["direction"] == "minimize":
                    normalized = -normalized
            else:
                lo, hi = ((values.min(), values.max()) if norm == "minmax_pool"
                          else norm["minmax_fixed"])
                normalized = ((values - lo) / (hi - lo) if hi > lo
                              else np.full(len(values), 0.5))
                if obj["direction"] == "minimize":
                    normalized = 1.0 - normalized
            scores[valid] = normalized
        for individual, value in zip(population, scores):
            individual.scores[prop] = float(value)
    for individual in population:
        if all(np.isfinite(individual.props[obj["property"]]) for obj in objectives):
            individual.fitness = float(aggregate(individual.scores, objectives))
        else:
            individual.fitness = float("-inf")
    return population
