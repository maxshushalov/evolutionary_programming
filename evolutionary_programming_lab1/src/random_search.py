"""Равномерный независимый случайный поиск при том же бюджете HGBat."""

import numpy as np

from .genetic_algorithm import GAConfig, RunResult
from .objective import hgbat


def run_random_search(config: GAConfig, seed: int) -> RunResult:
    """Оценить ровно evaluation_budget независимых равномерных векторов.

    Пакеты ограничивают расход памяти; каждый вектор оценивается отдельно.
    Ни операторы ГА, ни координаты теоретического оптимума не используются.
    """
    rng = np.random.default_rng(seed)
    best_x = np.empty(config.dimension)
    best_fitness = float("inf")
    evaluations = 0
    while evaluations < config.evaluation_budget:
        count = min(1024, config.evaluation_budget - evaluations)
        candidates = rng.uniform(
            config.lower_bound, config.upper_bound, (count, config.dimension),
        )
        for candidate in candidates:
            fitness = hgbat(candidate)
            evaluations += 1
            if fitness < best_fitness:
                best_x, best_fitness = candidate.copy(), fitness
    return RunResult(seed, best_x, best_fitness, evaluations)
