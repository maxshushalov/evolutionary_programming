import random
import time

from problem import decode, encode, evaluate, fitness_key


def copy_solution(solution):
    return {"permutation": solution["permutation"][:], "route_lengths": solution["route_lengths"][:]}


def random_solution(n, vehicles, rng):
    permutation = list(range(1, n + 1))
    rng.shuffle(permutation)
    lengths = [0] * vehicles
    for _ in range(n):
        lengths[rng.randrange(vehicles)] += 1
    return {"permutation": permutation, "route_lengths": lengths}


def order_crossover(first, second, rng):
    n = len(first["permutation"])
    if n < 2:
        return copy_solution(first)
    left, right = sorted(rng.sample(range(n + 1), 2))
    child = [None] * n
    child[left:right] = first["permutation"][left:right]
    used = set(child[left:right])
    position = right % n
    for i in range(n):
        gene = second["permutation"][(right + i) % n]
        if gene not in used:
            child[position] = gene
            position = (position + 1) % n
    return {"permutation": child, "route_lengths": first["route_lengths"][:]}


def mutate(solution, rng):
    child = copy_solution(solution)
    n = len(child["permutation"])
    move = rng.randrange(3)
    if move < 2 and n >= 2:
        left, right = sorted(rng.sample(range(n), 2))
        if move == 0:
            child["permutation"][left], child["permutation"][right] = child["permutation"][right], child["permutation"][left]
        else:
            child["permutation"][left:right + 1] = reversed(child["permutation"][left:right + 1])
    elif len(child["route_lengths"]) > 1:
        source = rng.choice([i for i, size in enumerate(child["route_lengths"]) if size])
        target = rng.choice([i for i in range(len(child["route_lengths"])) if i != source])
        child["route_lengths"][source] -= 1
        child["route_lengths"][target] += 1
    return child


def local_neighbor(solution, rng):
    routes = decode(solution)
    reversible = [i for i, route in enumerate(routes) if len(route) >= 2]
    nonempty = [i for i, route in enumerate(routes) if route]
    moves = []
    if reversible:
        moves.append("reverse")
    if len(routes) >= 2 and nonempty:
        moves.append("relocate")
    if not moves:
        return None
    if rng.choice(moves) == "reverse":
        route = routes[rng.choice(reversible)]
        left, right = sorted(rng.sample(range(len(route)), 2))
        route[left:right + 1] = reversed(route[left:right + 1])
    else:
        source = rng.choice(nonempty)
        target = rng.choice([i for i in range(len(routes)) if i != source])
        client = routes[source].pop(rng.randrange(len(routes[source])))
        routes[target].insert(rng.randrange(len(routes[target]) + 1), client)
    return encode(routes)


def tournament(population, size, rng):
    return min(rng.choices(population, k=size), key=lambda item: fitness_key(item["score"]))


def validate_parameters(params):
    if not 1 <= params["elite_size"] < params["population_size"]:
        raise ValueError("Нужно 1 <= elite_size < population_size")
    if params["evaluation_budget"] < params["population_size"]:
        raise ValueError("Бюджет меньше начальной популяции")
    if params["history_interval"] < 1 or params["tournament_size"] < 1 or params["local_search_steps"] < 0:
        raise ValueError("Некорректные целочисленные параметры")
    for name in ["crossover_rate", "mutation_rate", "local_search_rate"]:
        if not 0 <= params[name] <= 1:
            raise ValueError(f"Некорректная вероятность {name}")


def run_algorithm(instance, method, seed, params):
    validate_parameters(params)
    if method not in ["ga", "memetic"]:
        raise ValueError("Неизвестный метод")
    start = time.perf_counter()
    initial_rng = random.Random(f"{seed}:initial")
    variation_rng = random.Random(f"{seed}:variation")
    local_rng = random.Random(f"{seed}:local")
    state = {"evaluations": 0, "local_evaluations": 0, "accepted_local": 0,
             "generations": 0, "best": None, "first_feasible": None, "history": []}

    def record():
        score = state["best"]["score"]
        state["history"].append({"evaluations": state["evaluations"],
                                 "distance": score["distance"] if score["feasible"] else None,
                                 "violation": score["violation"], "feasible": score["feasible"],
                                 "generations": state["generations"],
                                 "local_evaluations": state["local_evaluations"]})

    def assess(solution, local=False):
        if state["evaluations"] >= params["evaluation_budget"]:
            raise RuntimeError("Превышение бюджета оценок")
        score = evaluate(instance, solution)
        state["evaluations"] += 1
        state["local_evaluations"] += int(local)
        item = {"solution": solution, "score": score}
        if state["best"] is None or fitness_key(score) < fitness_key(state["best"]["score"]):
            state["best"] = item
        if score["feasible"] and state["first_feasible"] is None:
            state["first_feasible"] = state["evaluations"]
        if state["evaluations"] % params["history_interval"] == 0:
            record()
        return item

    population = []
    n = len(instance["clients"]) - 1
    for _ in range(params["population_size"]):
        population.append(assess(random_solution(n, instance["vehicles"], initial_rng)))

    while state["evaluations"] < params["evaluation_budget"]:
        population.sort(key=lambda item: fitness_key(item["score"]))
        next_population = population[:params["elite_size"]]
        while len(next_population) < params["population_size"] and state["evaluations"] < params["evaluation_budget"]:
            first = tournament(population, params["tournament_size"], variation_rng)
            second = tournament(population, params["tournament_size"], variation_rng)
            if variation_rng.random() < params["crossover_rate"]:
                child = order_crossover(first["solution"], second["solution"], variation_rng)
            else:
                child = copy_solution(first["solution"])
            if variation_rng.random() < params["mutation_rate"]:
                child = mutate(child, variation_rng)
            item = assess(child)

            if method == "memetic" and local_rng.random() < params["local_search_rate"]:
                for _ in range(params["local_search_steps"]):
                    if state["evaluations"] >= params["evaluation_budget"]:
                        break
                    neighbor = local_neighbor(item["solution"], local_rng)
                    if neighbor is None:
                        break
                    candidate = assess(neighbor, local=True)
                    if fitness_key(candidate["score"]) < fitness_key(item["score"]):
                        item = candidate
                        state["accepted_local"] += 1
            next_population.append(item)
        population = next_population
        state["generations"] += 1

    if not state["history"] or state["history"][-1]["evaluations"] != state["evaluations"]:
        record()
    return {"method": method, "seed": seed, "params": dict(params), "best": state["best"],
            "evaluations": state["evaluations"], "local_evaluations": state["local_evaluations"],
            "accepted_local": state["accepted_local"], "generations": state["generations"],
            "first_feasible": state["first_feasible"], "history": state["history"],
            "seconds": time.perf_counter() - start}
