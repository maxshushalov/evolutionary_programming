"""Вещественный ГА: турнир, арифметический кроссовер, мутация и элитизм."""

from dataclasses import dataclass, field

import numpy as np

from .objective import DIMENSION, LOWER_BOUND, UPPER_BOUND, hgbat


@dataclass(frozen=True)
class GAConfig:
    """Явные параметры одного запуска; sigma задаётся в единицах координат."""

    dimension: int = DIMENSION
    lower_bound: float = LOWER_BOUND
    upper_bound: float = UPPER_BOUND
    population_size: int = 100
    tournament_size: int = 3
    crossover_probability: float = 0.9
    mutation_probability: float = 1 / DIMENSION
    mutation_sigma: float = 3.0
    elite_count: int = 1
    evaluation_budget: int = 50_000

    def __post_init__(self) -> None:
        """Отклонить параметры, при которых цикл не может работать корректно."""
        for name in (
            "dimension", "population_size", "tournament_size",
            "elite_count", "evaluation_budget",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} должен быть целым числом")
        if self.dimension < 1 or self.population_size < 2:
            raise ValueError("dimension >= 1, population_size >= 2")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size должен быть от 1 до population_size")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count должен быть от 0 до population_size - 1")
        if self.evaluation_budget < self.population_size:
            raise ValueError("Бюджет должен покрывать всю начальную популяцию")
        if not (
            np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound)
            and self.lower_bound < self.upper_bound
        ):
            raise ValueError("Нужны конечные границы lower_bound < upper_bound")
        for name in ("crossover_probability", "mutation_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} должна лежать в [0, 1]")
        if not np.isfinite(self.mutation_sigma) or self.mutation_sigma < 0:
            raise ValueError("mutation_sigma должна быть конечной и неотрицательной")


@dataclass
class RunResult:
    """Лучшее решение за весь запуск и фактически израсходованный бюджет."""

    seed: int
    best_x: np.ndarray
    best_fitness: float
    evaluations: int
    generations: int = 0
    history: list[dict[str, int | float]] = field(default_factory=list)


def initialize_population(config: GAConfig, rng: np.random.Generator) -> np.ndarray:
    """Равномерно и независимо заполнить популяцию в области поиска."""
    return rng.uniform(
        config.lower_bound, config.upper_bound,
        size=(config.population_size, config.dimension),
    )


def evaluate_population(population: np.ndarray) -> np.ndarray:
    """Ровно один вызов HGBat на каждую переданную особь."""
    return np.array([hgbat(individual) for individual in population], dtype=float)


def tournament_select(
    population: np.ndarray, fitness: np.ndarray,
    tournament_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Выбрать минимум среди разных участников одного турнира.

    Между турнирами выбор независим: родитель может выбираться повторно.
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    winner = indices[np.argmin(fitness[indices])]
    return population[winner].copy()


def arithmetic_crossover(
    parent1: np.ndarray, parent2: np.ndarray,
    probability: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Использовать один новый alpha для пары или вернуть копии родителей."""
    if rng.random() >= probability:
        return parent1.copy(), parent2.copy()
    alpha = rng.uniform(0.0, 1.0)
    return (
        alpha * parent1 + (1.0 - alpha) * parent2,
        (1.0 - alpha) * parent1 + alpha * parent2,
    )


def gaussian_mutation(
    individual: np.ndarray, probability: float,
    sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Независимо мутировать каждую координату с заданной вероятностью."""
    child = individual.copy()
    mask = rng.random(child.size) < probability
    child[mask] += rng.normal(0.0, sigma, size=int(mask.sum()))
    return child


def clip_to_bounds(
    individual: np.ndarray, lower_bound: float, upper_bound: float,
) -> np.ndarray:
    """Вернуть координаты, обрезанные по границам области поиска."""
    return np.clip(individual, lower_bound, upper_bound)


def make_offspring(
    population: np.ndarray, fitness: np.ndarray, count: int,
    config: GAConfig, rng: np.random.Generator,
) -> np.ndarray:
    """Создать ровно count потомков, включая нечётный остаток."""
    children = []
    while len(children) < count:
        parent1 = tournament_select(population, fitness, config.tournament_size, rng)
        parent2 = tournament_select(population, fitness, config.tournament_size, rng)
        pair = arithmetic_crossover(parent1, parent2, config.crossover_probability, rng)
        for child in pair:
            child = gaussian_mutation(child, config.mutation_probability, config.mutation_sigma, rng)
            children.append(clip_to_bounds(child, config.lower_bound, config.upper_bound))
            if len(children) == count:
                break
    return np.asarray(children)


def record_history(
    generation: int, evaluations: int, fitness: np.ndarray, global_best: float,
) -> dict[str, int | float]:
    """Зафиксировать статистику текущей популяции и глобальный рекорд."""
    return {
        "generation": generation,
        "evaluation_count": evaluations,
        "best_fitness": float(fitness.min()),
        "mean_fitness": float(fitness.mean()),
        "worst_fitness": float(fitness.max()),
        "global_best_fitness": global_best,
        "population_size": len(fitness),
    }


def run_genetic_algorithm(config: GAConfig, seed: int) -> RunResult:
    """Минимизировать HGBat, израсходовав ровно evaluation_budget оценок.

    Элиту не оцениваем повторно. В последнем неполном поколении оставляем
    лучших N - count старых особей вместе с их оценками и добавляем count
    новых потомков. Размер популяции N и бюджет при этом сохраняются.
    """
    rng = np.random.default_rng(seed)
    population = initialize_population(config, rng)
    fitness = evaluate_population(population)
    evaluations = config.population_size
    best_index = int(np.argmin(fitness))
    best_x, best_fitness = population[best_index].copy(), float(fitness[best_index])
    generation = 0
    history = [record_history(generation, evaluations, fitness, best_fitness)]

    while evaluations < config.evaluation_budget:
        count = min(config.population_size - config.elite_count,
                    config.evaluation_budget - evaluations)
        children = make_offspring(population, fitness, count, config, rng)
        child_fitness = evaluate_population(children)
        retained = np.argsort(fitness, kind="stable")[:config.population_size - count]
        population = np.concatenate((population[retained], children))
        fitness = np.concatenate((fitness[retained], child_fitness))
        evaluations += count
        generation += 1
        best_index = int(np.argmin(fitness))
        if fitness[best_index] < best_fitness:
            best_x, best_fitness = population[best_index].copy(), float(fitness[best_index])
        history.append(record_history(generation, evaluations, fitness, best_fitness))

    return RunResult(seed, best_x, best_fitness, evaluations, generation, history)
