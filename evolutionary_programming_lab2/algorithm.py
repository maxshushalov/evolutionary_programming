"""Генетический алгоритм, ремонт и случайный поиск для недельного меню."""

import time
import numpy as np
from model import evaluate_menu, rank_key

METHODS = {
    "penalty_bit": {"repair": False, "mutation": "bit"},
    "penalty_swap": {"repair": False, "mutation": "swap"},
    "combined_bit": {"repair": True, "mutation": "bit"},
    "combined_swap": {"repair": True, "mutation": "swap"},
    "random_repair": {"repair": True, "mutation": None},
}


def random_menu(rng, catalog, config):
    """Создаёт независимые биты; ожидаемая калорийность дня — середина диапазона."""
    target = (config["min_calories"] + config["max_calories"]) / 2
    probability = min(1.0, target / float(catalog["calories"].sum()))
    shape = (config["days"], len(catalog["dishes"]))
    return (rng.random(shape) < probability).astype(np.int8)


def new_state(config):
    """Создаёт словарь бюджета, архива допустимых решений и журнала оценок."""
    return {"limit": config["evaluation_budget"], "evaluations": 0,
            "candidate_evaluations": 0, "repair_evaluations": 0,
            "repair_accepted": 0, "repair_calls": 0, "repair_seconds": 0.0,
            "feasible_evaluations": 0, "first_feasible_evaluation": None,
            "best_menu": None, "best_result": None, "history": []}


def record_history(state):
    """Сохраняет рекорд на общей шкале числа оценок, включая ремонт."""
    best_cost = None
    if state["best_result"] is not None:
        best_cost = state["best_result"]["total_cost"]
    state["history"].append({"evaluations": state["evaluations"],
                             "best_cost": best_cost,
                             "feasible_evaluations": state["feasible_evaluations"]})


def evaluate_counted(menu, catalog, config, state, source):
    """Единственный вход к оценке во время поиска; одна проверка стоит единицу."""
    if state["evaluations"] >= state["limit"]:
        raise RuntimeError("Вычислительный бюджет исчерпан")
    result = evaluate_menu(menu, catalog, config)
    state["evaluations"] += 1
    state[source + "_evaluations"] += 1
    if result["feasible"]:
        state["feasible_evaluations"] += 1
        if state["first_feasible_evaluation"] is None:
            state["first_feasible_evaluation"] = state["evaluations"]
        if state["best_result"] is None or result["total_cost"] < state["best_result"]["total_cost"]:
            state["best_menu"] = menu.copy()
            state["best_result"] = result
    if state["evaluations"] % config["history_interval"] == 0:
        record_history(state)
    return result


def individual_key(individual):
    """Возвращает ключ сортировки особи с уже известной оценкой."""
    return rank_key(individual["result"])


def select_parent(population, rng, tournament_size):
    """Выбирает лучшую особь из случайного турнира без повторов участников."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner = population[int(indices[0])]
    for index in indices[1:]:
        candidate = population[int(index)]
        if individual_key(candidate) < individual_key(winner):
            winner = candidate
    return winner["menu"]


def crossover(first, second, rng, probability):
    """Создаёт одного потомка, наследуя каждый день целиком от одного родителя."""
    child = first.copy()
    if rng.random() < probability:
        for day in range(len(child)):
            if rng.random() < 0.5:
                child[day] = second[day]
    return child


def mutate(menu, rng, config, mutation):
    """Возвращает новую матрицу после битовой мутации или замены блюда."""
    child = menu.copy()
    if mutation == "bit":
        mask = rng.random(child.shape) < config["bit_mutation_rate"]
        child[mask] = 1 - child[mask]
    elif mutation == "swap":
        if rng.random() < config["swap_probability"]:
            day = int(rng.integers(len(child)))
            present = np.flatnonzero(child[day] == 1)
            absent = np.flatnonzero(child[day] == 0)
            if len(present) > 0 and len(absent) > 0:
                child[day, int(rng.choice(present))] = 0
                child[day, int(rng.choice(absent))] = 1
    else:
        raise ValueError("Неизвестная мутация")
    return child


def repair_proposal(menu, result, rng, catalog, config):
    """Предлагает одну правку по первому нарушению; качество здесь не вычисляет."""
    trial = menu.copy()
    # Решение о принятии будет после полноценной оплаченной проверки.
    overused = np.flatnonzero(result["frequency_excess"] > 0)
    low_days = np.flatnonzero(result["calorie_shortage"] > 0)
    high_days = np.flatnonzero(result["calorie_excess"] > 0)
    if len(overused) > 0:
        dish = int(rng.choice(overused))
        day = int(rng.choice(np.flatnonzero(menu[:, dish])))
        trial[day, dish] = 0
        replacements = np.flatnonzero((menu[day] == 0) & (result["frequency"] < config["max_frequency"]))
        if len(replacements) > 0:
            trial[day, int(rng.choice(replacements))] = 1
    elif len(low_days) > 0:
        day = int(rng.choice(low_days))
        absent = np.flatnonzero((menu[day] == 0) & (result["frequency"] < config["max_frequency"]))
        if len(absent) > 0:
            trial[day, int(rng.choice(absent))] = 1
    elif len(high_days) > 0:
        day = int(rng.choice(high_days))
        present = np.flatnonzero(menu[day])
        trial[day, int(rng.choice(present))] = 0
    elif result["diversity_shortage"] > 0:
        absent = np.flatnonzero(result["frequency"] == 0)
        if len(absent) > 0:
            day = int(rng.integers(len(menu)))
            present = np.flatnonzero(menu[day] & (result["frequency"] > 1))
            if len(present) > 0:
                trial[day, int(rng.choice(present))] = 0
            trial[day, int(rng.choice(absent))] = 1
    elif result["budget_excess"] > 0:
        day = int(rng.integers(len(menu)))
        present = np.flatnonzero(menu[day])
        if len(present) > 0:
            dish = int(rng.choice(present))
            cheaper = np.flatnonzero((menu[day] == 0) & (catalog["prices"] < catalog["prices"][dish]))
            if len(cheaper) > 0:
                trial[day, dish] = 0
                trial[day, int(rng.choice(cheaper))] = 1
    return trial


def repair_menu(menu, result, rng, catalog, config, state):
    """Пытается уменьшить нарушения; все пробы, даже отклонённые, тратят бюджет."""
    start = time.perf_counter()
    state["repair_calls"] += 1
    for attempt in range(config["repair_attempts"]):
        if result["feasible"] or state["evaluations"] >= state["limit"]:
            break
        trial = repair_proposal(menu, result, rng, catalog, config)
        trial_result = evaluate_counted(trial, catalog, config, state, "repair")
        # Ремонт уменьшает нарушение; при равенстве уменьшает стоимость.
        old_key = (result["violation"], result["total_cost"])
        new_key = (trial_result["violation"], trial_result["total_cost"])
        if new_key < old_key:
            menu = trial
            result = trial_result
            state["repair_accepted"] += 1
    state["repair_seconds"] += time.perf_counter() - start
    return menu, result


def population_record(population, state, generation):
    """Описывает популяцию по сохранённым оценкам, не расходуя новые оценки."""
    feasible = 0
    for individual in population:
        if individual["result"]["feasible"]:
            feasible += 1
    return {"generation": generation, "evaluations": state["evaluations"],
            "population_size": len(population), "feasible_fraction": feasible / len(population),
            "best_score": min(population, key=individual_key)["result"]["score"]}


def run_search(method, seed, catalog, config):
    """Выполняет один запуск с точным общим бюджетом; возвращает результат и историю."""
    settings = METHODS[method]
    start = time.perf_counter()
    initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    repair_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    state = new_state(config)
    population_history = []
    generation = 0

    if method == "random_repair":
        # Каждая попытка независима; между попытками нет наследования.
        while state["evaluations"] < state["limit"]:
            menu = random_menu(initial_rng, catalog, config)
            result = evaluate_counted(menu, catalog, config, state, "candidate")
            repair_menu(menu, result, repair_rng, catalog, config, state)
    else:
        # 1. Создание и оценка общей для всех вариантов ГА начальной популяции.
        # Здесь ремонта нет: A–D начинают с одних и тех же матриц и оценок.
        population = []
        for index in range(config["population_size"]):
            menu = random_menu(initial_rng, catalog, config)
            result = evaluate_counted(menu, catalog, config, state, "candidate")
            population.append({"menu": menu, "result": result})
        population_history.append(population_record(population, state, generation))

        while state["evaluations"] < state["limit"]:
            # 2. Элита переносится вместе с оценками, без повторного вычисления.
            ordered = sorted(population, key=individual_key)
            next_population = ordered[:config["elite_count"]]
            while len(next_population) < config["population_size"]:
                if state["evaluations"] >= state["limit"]:
                    break
                # 3. Выбор родителей → кроссовер → мутация.
                first = select_parent(population, rng, config["tournament_size"])
                second = select_parent(population, rng, config["tournament_size"])
                child = crossover(first, second, rng, config["crossover_probability"])
                child = mutate(child, rng, config, settings["mutation"])
                # 4. Оценка → необязательный ремонт → добавление потомка.
                result = evaluate_counted(child, catalog, config, state, "candidate")
                if settings["repair"]:
                    child, result = repair_menu(child, result, repair_rng, catalog, config, state)
                next_population.append({"menu": child, "result": result})
            # 5. В неполном последнем поколении свободные места занимают старые особи.
            missing = config["population_size"] - len(next_population)
            first_old = config["elite_count"]
            next_population.extend(ordered[first_old:first_old + missing])
            population = next_population
            generation += 1
            population_history.append(population_record(population, state, generation))

    if not state["history"] or state["history"][-1]["evaluations"] != state["evaluations"]:
        record_history(state)
    state["method"] = method
    state["seed"] = seed
    state["generations"] = generation
    state["population_history"] = population_history
    state["seconds"] = time.perf_counter() - start
    return state
