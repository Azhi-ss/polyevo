import math

from polyevo.core.registry import register


def quality(individual):
    return -individual.violation, individual.fitness


@register("select", "tournament")
def tournament(population, ctx, k=3):
    if not population:
        raise ValueError("Cannot select from an empty population")
    choices = ctx.rng.integers(0, len(population), size=k)
    return max((population[int(i)] for i in choices), key=quality)


@register("init", "random_assemble")
def random_assemble(ctx, make_candidate):
    population, seen = [], set()
    encoding = next(iter(ctx.encodings.values()))
    for _ in range(ctx.cfg["population_size"] * ctx.cfg["max_attempts"]):
        individual = make_candidate(encoding.name, encoding.random(ctx), (), ["random"])
        if individual is not None and individual.id not in seen:
            population.append(individual)
            seen.add(individual.id)
        if len(population) == ctx.cfg["population_size"]:
            return population
    raise ValueError(f"Initialization produced only {len(population)} unique valid structures; "
                     "enlarge the library/search space or lower population_size")


@register("reproduce", "family")
def family(population, ctx, select, make_candidate, families=10, children=4,
           crossover="brics_single_point", crossover_probability=1.0,
           mutations=None):
    mutations = mutations or []
    offspring = []
    seen = {i.id for i in population}
    for _ in range(families):
        a = select(population, ctx)
        compatible = [i for i in population if i.encoding == a.encoding]
        b = select(compatible, ctx)
        encoding = ctx.encodings[a.encoding]
        for _ in range(children):
            for _ in range(ctx.cfg["max_attempts"]):
                genotype, ops = a.genotype, []
                candidate = None
                if ctx.rng.random() < crossover_probability:
                    genotype = encoding.crossover_ops[crossover](genotype, b.genotype, ctx)
                    ops.append(crossover)
                    candidate = make_candidate(a.encoding, genotype, (a.id, b.id), ops.copy())
                    if candidate is None:
                        continue
                valid = True
                for mutation in mutations:
                    if ctx.rng.random() < mutation["probability"]:
                        genotype = encoding.mutation_ops[mutation["name"]](genotype, ctx)
                        ops.append(mutation["name"])
                        candidate = make_candidate(a.encoding, genotype, (a.id, b.id), ops.copy())
                        if candidate is None:
                            valid = False
                            break
                if not valid:
                    continue
                if candidate is None:
                    candidate = make_candidate(a.encoding, genotype, (a.id, b.id), ops.copy())
                if candidate is not None and candidate.id not in seen:
                    offspring.append(candidate)
                    seen.add(candidate.id)
                    break
    return offspring


@register("survive", "generational_elitist")
def generational_elitist(candidates, parents, ctx, ratio=0.1, mode="replace"):
    size = ctx.cfg["population_size"]
    parent_ids = {i.id for i in parents}
    ordered = sorted(candidates, key=quality, reverse=True)
    elites = [i for i in ordered if i.id in parent_ids][:math.ceil(size * ratio)]
    children = [i for i in ordered if i.id not in parent_ids]
    target = size + len(elites) if mode == "append" else size
    chosen = elites + children[:target - len(elites)]
    selected = {i.id for i in chosen}
    # Bounded search or a partial budget may yield fewer children: fill with parents.
    chosen.extend(i for i in ordered if i.id not in selected)
    chosen = sorted(chosen[:target], key=quality, reverse=True)
    ctx.archive.update(chosen, max(1, math.ceil(size * ratio)))
    return chosen, ctx.archive


@register("diversity", "none")
def no_diversity(population, ctx):
    return population


@register("stop", "max_generations")
def max_generations(ctx, population):
    return ctx.generation >= ctx.cfg["max_generations"]


@register("stop", "budget_exhausted")
def budget_exhausted(ctx, population):
    return ctx.budget.exhausted
