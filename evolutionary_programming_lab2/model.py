"""Данные задачи, единственная проверка ограничений и декодирование меню."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_config(path=ROOT / "config.json"):
    """Читает JSON и проверяет диапазоны параметров; возвращает словарь."""
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    positive = ["days", "min_calories", "max_calories", "budget_kopecks",
                "min_unique", "max_frequency", "population_size", "tournament_size",
                "evaluation_budget", "history_interval"]
    for name in positive:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name}: нужно положительное целое число")
    if config["min_calories"] > config["max_calories"]:
        raise ValueError("Нижняя граница калорий выше верхней")
    if config["tournament_size"] > config["population_size"]:
        raise ValueError("Турнир больше популяции")
    elite = config["elite_count"]
    if type(elite) is not int or not 0 <= elite < config["population_size"]:
        raise ValueError("Число элитных особей должно быть меньше популяции")
    if config["evaluation_budget"] < config["population_size"]:
        raise ValueError("Бюджета недостаточно для начальной популяции")
    for name in ["crossover_probability", "bit_mutation_rate", "swap_probability"]:
        if not 0 <= config[name] <= 1:
            raise ValueError(f"{name}: вероятность должна быть от 0 до 1")
    if not np.isfinite(config["penalty_weight"]) or config["penalty_weight"] <= 0:
        raise ValueError("Вес штрафа должен быть положительным конечным числом")
    if type(config["repair_attempts"]) is not int or config["repair_attempts"] < 0:
        raise ValueError("Число попыток ремонта должно быть неотрицательным целым")
    for seed in [config["data_seed"]] + config["seeds"]:
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed должен быть неотрицательным целым")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Список seed должен быть непустым, без повторов")
    return config


def load_catalog(path=ROOT / "data" / "dishes.csv"):
    """Читает порции из CSV; цены и калории возвращает также как массивы."""
    with open(path, encoding="utf-8", newline="") as file:
        dishes = list(csv.DictReader(file))
    identifiers = set()
    prices = []
    calories = []
    for dish in dishes:
        if not dish["id"] or dish["id"] in identifiers:
            raise ValueError("Идентификаторы блюд должны быть уникальными")
        identifiers.add(dish["id"])
        for field in ["portion_size", "price_kopecks", "calories"]:
            dish[field] = int(dish[field])
            if dish[field] <= 0:
                raise ValueError(f"{field}: нужно положительное число")
        if not dish["name"] or not dish["portion_unit"]:
            raise ValueError("Нужны название и единица порции")
        prices.append(dish["price_kopecks"])
        calories.append(dish["calories"])
    if not dishes:
        raise ValueError("Каталог пуст")
    return {"dishes": dishes, "prices": np.array(prices, dtype=np.int64),
            "calories": np.array(calories, dtype=np.int64)}


def evaluate_menu(menu, catalog, config):
    """Возвращает стоимость, нарушения и штрафную оценку одной матрицы меню."""
    menu = np.asarray(menu)
    expected_shape = (config["days"], len(catalog["dishes"]))
    if menu.shape != expected_shape:
        raise ValueError(f"Ожидается матрица {expected_shape}")
    if not np.all((menu == 0) | (menu == 1)):
        raise ValueError("Матрица должна содержать только 0 и 1")

    daily_calories = menu @ catalog["calories"]
    daily_cost = menu @ catalog["prices"]
    total_cost = int(daily_cost.sum())
    frequency = menu.sum(axis=0)
    unique_count = int(np.count_nonzero(frequency))

    calorie_shortage = np.maximum(0, config["min_calories"] - daily_calories)
    calorie_excess = np.maximum(0, daily_calories - config["max_calories"])
    budget_excess = max(0, total_cost - config["budget_kopecks"])
    diversity_shortage = max(0, config["min_unique"] - unique_count)
    frequency_excess = np.maximum(0, frequency - config["max_frequency"])

    violation = float((calorie_shortage.sum() + calorie_excess.sum()) / config["min_calories"])
    violation += budget_excess / config["budget_kopecks"]
    violation += diversity_shortage / config["min_unique"]
    violation += float(frequency_excess.sum() / config["max_frequency"])
    score = total_cost / config["budget_kopecks"] + config["penalty_weight"] * violation
    return {"total_cost": total_cost, "daily_cost": daily_cost,
            "daily_calories": daily_calories, "frequency": frequency,
            "unique_count": unique_count, "calorie_shortage": calorie_shortage,
            "calorie_excess": calorie_excess, "budget_excess": budget_excess,
            "diversity_shortage": diversity_shortage,
            "frequency_excess": frequency_excess, "violation": violation,
            "feasible": violation == 0, "score": score}


def rank_key(result):
    """Одинаковое правило отбора для обеих стратегий: штраф, нарушение, цена."""
    return result["score"], result["violation"], result["total_cost"]


def decode_menu(menu, catalog, config):
    """Преобразует матрицу в список дневных меню; использует общую проверку."""
    result = evaluate_menu(menu, catalog, config)
    days = []
    for day in range(config["days"]):
        dishes = []
        for index, dish in enumerate(catalog["dishes"]):
            if menu[day, index]:
                dishes.append(dish)
        days.append({"day": day + 1, "dishes": dishes,
                     "calories": int(result["daily_calories"][day]),
                     "cost_kopecks": int(result["daily_cost"][day])})
    return days


def json_ready(value):
    """Переводит массивы NumPy в списки для сохранения результатов в JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            converted[key] = json_ready(item)
        return converted
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def save_json(path, value):
    """Сохраняет словарь или список в читаемый UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding="utf-8")
