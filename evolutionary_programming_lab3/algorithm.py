import random
from model import random_genes

WEIGHTS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.4, 0.4))


def dominates(left, right):
    a, b = left['objectives'], right['objectives']
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fronts(population):
    dominated = [[] for _ in population]
    counts = [0] * len(population)
    first = []
    for i, individual in enumerate(population):
        for j in range(i + 1, len(population)):
            if dominates(individual, population[j]):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(population[j], individual):
                dominated[j].append(i)
                counts[i] += 1
        if counts[i] == 0:
            first.append(i)
    layers = []
    current = first
    while current:
        layers.append(current)
        following = []
        for i in current:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        current = following
    return layers


def crowding(population, layer):
    distances = {i: 0.0 for i in layer}
    if len(layer) <= 2:
        return {i: float('inf') for i in layer}
    for objective in range(3):
        ordered = sorted(layer, key=lambda i: population[i]['objectives'][objective])
        low = population[ordered[0]]['objectives'][objective]
        high = population[ordered[-1]]['objectives'][objective]
        if high == low:
            continue
        distances[ordered[0]] = distances[ordered[-1]] = float('inf')
        for index in range(1, len(ordered) - 1):
            before = population[ordered[index - 1]]['objectives'][objective]
            after = population[ordered[index + 1]]['objectives'][objective]
            distances[ordered[index]] += (after - before) / (high - low)
    return distances


def ranks_and_distances(population):
    ranks = {}
    distances = {}
    for rank, layer in enumerate(fronts(population)):
        for index in layer:
            ranks[index] = rank
        distances.update(crowding(population, layer))
    return ranks, distances


def tournament(population, rng, mode, weights=None):
    a = rng.randrange(len(population))
    b = rng.randrange(len(population))
    if mode == 'weighted':
        scores = [sum(w * f for w, f in zip(weights, item['objectives'])) for item in (population[a], population[b])]
        return population[a if scores[0] < scores[1] else b if scores[1] < scores[0] else rng.choice((a, b))]
    ranks, distances = mode
    return population[min((a, b), key=lambda i: (ranks[i], -distances[i], rng.random()))]


def child(first, second, rng, mutation_rate):
    first = list(first)
    second = list(second)
    if rng.random() < 0.9:
        for i in range(6):
            if rng.random() < 0.5:
                first[i] = second[i]
    for i in range(6):
        if rng.random() >= mutation_rate:
            continue
        if i == 0:
            first[i] = rng.choice([depth for depth in (1, 2, 3) if depth != first[i]])
        else:
            limit = 0.5 if i == 5 else 1.0
            sigma = 0.05 if i == 5 else 0.1
            first[i] = min(limit, max(0.0, first[i] + rng.gauss(0, sigma)))
    return tuple(first)


def update_archive(archive, candidate):
    if any(item['genes'] == candidate['genes'] or dominates(item, candidate) for item in archive):
        return archive
    return [item for item in archive if not dominates(candidate, item)] + [candidate]


def optimize(mode, seed, budget, population_size, mutation_rate, evaluate, checkpoint):
    rng = random.Random(seed)
    population = []
    archive = []
    history = []
    evaluated = []
    evaluations = 0

    def score(genes):
        nonlocal evaluations, archive
        candidate = evaluate(genes)
        candidate['evaluation'] = evaluations + 1
        evaluations += 1
        archive = update_archive(archive, candidate)
        evaluated.append(candidate)
        if evaluations % checkpoint == 0 or evaluations == budget:
            history.append({'evaluations': evaluations, 'front': [item['objectives'] for item in archive]})
        return candidate

    while len(population) < min(population_size, budget):
        population.append(score(random_genes(rng)))
    while evaluations < budget:
        ranking = ranks_and_distances(population) if mode == 'nsga2' else None
        offspring = []
        for _ in range(min(population_size, budget - evaluations)):
            first = tournament(population, rng, ranking if ranking is not None else 'weighted', mode if ranking is None else None)
            second = tournament(population, rng, ranking if ranking is not None else 'weighted', mode if ranking is None else None)
            offspring.append(score(child(first['genes'], second['genes'], rng, mutation_rate)))
        combined = population + offspring
        if mode == 'nsga2':
            next_population = []
            for layer in fronts(combined):
                available = population_size - len(next_population)
                if len(layer) <= available:
                    next_population.extend(combined[i] for i in layer)
                else:
                    distances = crowding(combined, layer)
                    chosen = sorted(layer, key=lambda i: (-distances[i], i))[:available]
                    next_population.extend(combined[i] for i in chosen)
                    break
            population = next_population
        else:
            population = sorted(combined, key=lambda item: sum(w * f for w, f in zip(mode, item['objectives'])))[:population_size]
    return {'evaluations': evaluations, 'archive': archive, 'history': history, 'evaluated': evaluated}
