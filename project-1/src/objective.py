"""Целевая функция HGBat и область поиска варианта 20."""

import numpy as np

DIMENSION = 7
LOWER_BOUND = -15.0
UPPER_BOUND = 15.0


def hgbat(x: np.ndarray) -> float:
    """Вычислить HGBat для конечного непустого вектора (в варианте d=7).

    Размерность берётся из вектора, чтобы её можно было менять в конфигурации.
    Область поиска контролирует алгоритм, а не целевая функция.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x должен быть непустым одномерным конечным вектором")
    sum_squares = float(np.sum(x**2))
    sum_coordinates = float(np.sum(x))
    return float(
        np.sqrt(abs(sum_squares**2 - sum_coordinates**2))
        + (0.5 * sum_squares + sum_coordinates) / x.size
        + 0.5
    )


def is_feasible(
    x: np.ndarray,
    dimension: int = DIMENSION,
    lower_bound: float = LOWER_BOUND,
    upper_bound: float = UPPER_BOUND,
) -> bool:
    """Проверить размерность, конечность и включение в заданные границы."""
    x = np.asarray(x)
    return bool(
        x.shape == (dimension,)
        and np.isfinite(x).all()
        and np.all(x >= lower_bound)
        and np.all(x <= upper_bound)
    )


def self_test() -> None:
    """Проверить формулу в известных точках; алгоритмы эту функцию не вызывают."""
    assert np.isclose(hgbat(np.full(DIMENSION, -1.0)), 0.0, atol=1e-12)
    assert hgbat(np.zeros(DIMENSION)) == 0.5
    assert hgbat(np.ones(DIMENSION)) == 2.0


if __name__ == "__main__":
    self_test()
    print("HGBat: проверки пройдены")
