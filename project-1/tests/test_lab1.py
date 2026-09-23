"""Проверки инвариантов ГА, честности бюджета и сохранения артефактов."""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import genetic_algorithm as ga
from src import random_search as rs
from src.experiments import experiment_configs, load_config, run_experiments, summarize_results
from src.objective import hgbat, is_feasible, self_test


class ObjectiveTests(unittest.TestCase):
    def test_reference_values(self):
        self_test()
        self.assertIsInstance(hgbat(np.zeros(7)), float)
        x = np.arange(-3.0, 4.0)
        q, s = sum(float(v) ** 2 for v in x), sum(float(v) for v in x)
        self.assertAlmostEqual(hgbat(x), abs(q**2 - s**2)**0.5 + (0.5*q+s)/7 + 0.5)

    def test_invalid_vectors(self):
        for x in ([], [[1, 2]], [np.nan], [np.inf], 1):
            with self.subTest(x=x), self.assertRaises(ValueError):
                hgbat(x)

    def test_feasibility(self):
        self.assertTrue(is_feasible(np.full(7, -15)))
        self.assertTrue(is_feasible(np.full(7, 15)))
        self.assertFalse(is_feasible(np.full(7, 15.01)))
        self.assertFalse(is_feasible(np.full(6, 0)))
        self.assertFalse(is_feasible(np.full(7, np.nan)))


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.config = ga.GAConfig(population_size=10, evaluation_budget=123)

    def test_initialization(self):
        population = ga.initialize_population(self.config, self.rng)
        self.assertEqual(population.shape, (10, 7))
        self.assertTrue(all(is_feasible(x) for x in population))
        np.testing.assert_array_equal(
            population, ga.initialize_population(self.config, np.random.default_rng(42)),
        )

    def test_tournament_minimizes_and_returns_copy(self):
        population = np.arange(21.0).reshape(3, 7)
        selected = ga.tournament_select(population, np.array([3, 1, 2]), 3, self.rng)
        np.testing.assert_array_equal(selected, population[1])
        self.assertFalse(np.shares_memory(selected, population))

    def test_crossover_formula_and_no_parent_mutation(self):
        p1, p2 = np.full(7, -15.0), np.full(7, 15.0)
        children = ga.arithmetic_crossover(p1, p2, 1.0, self.rng)
        np.testing.assert_allclose(children[0] + children[1], p1 + p2)
        self.assertTrue(all(is_feasible(x) for x in children))
        self.assertEqual(len(np.unique(children[0])), 1)  # один alpha на пару
        copies = ga.arithmetic_crossover(p1, p2, 0.0, self.rng)
        for original, copy in zip((p1, p2), copies):
            np.testing.assert_array_equal(original, copy)
            self.assertFalse(np.shares_memory(original, copy))

    def test_mutation_probabilities_and_clipping(self):
        original = np.zeros(7)
        np.testing.assert_array_equal(ga.gaussian_mutation(original, 0, 3, self.rng), original)
        np.testing.assert_array_equal(ga.gaussian_mutation(original, 1, 0, self.rng), original)
        mutated = ga.gaussian_mutation(original, 1, 100, self.rng)
        self.assertTrue(np.all(mutated != original))
        self.assertTrue(is_feasible(ga.clip_to_bounds(mutated, -15, 15)))
        np.testing.assert_array_equal(original, np.zeros(7))

    def test_offspring_odd_count_and_bounds(self):
        population = ga.initialize_population(self.config, self.rng)
        fitness = ga.evaluate_population(population)
        children = ga.make_offspring(population, fitness, 9,
                                    replace(self.config, mutation_sigma=100), self.rng)
        self.assertEqual(children.shape, (9, 7))
        self.assertTrue(all(is_feasible(x) for x in children))


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.config = ga.GAConfig(population_size=10, evaluation_budget=123)

    def test_exact_call_budget_including_partial_generation(self):
        for budget in (10, 11, 18, 19, 20, 123):
            with self.subTest(budget=budget):
                config = replace(self.config, evaluation_budget=budget)
                with patch.object(ga, "hgbat", wraps=hgbat) as objective:
                    result = ga.run_genetic_algorithm(config, 0)
                    self.assertEqual(objective.call_count, budget)
                with patch.object(rs, "hgbat", wraps=hgbat) as objective:
                    random_result = rs.run_random_search(config, 0)
                    self.assertEqual(objective.call_count, budget)
                self.assertEqual(result.evaluations, random_result.evaluations)
                self.assertEqual(result.evaluations, budget)
                self.assertEqual(result.history[-1]["evaluation_count"], budget)

    def test_population_size_elite_and_history(self):
        snapshots = []
        original = ga.make_offspring

        def capture(population, fitness, count, config, rng):
            snapshots.append((population.copy(), fitness.copy()))
            return original(population, fitness, count, config, rng)

        with patch.object(ga, "make_offspring", side_effect=capture):
            result = ga.run_genetic_algorithm(self.config, 2)
        for population, fitness in snapshots:
            self.assertEqual(population.shape, (10, 7))
            self.assertTrue(all(is_feasible(x) for x in population))
            np.testing.assert_allclose(fitness, [hgbat(x) for x in population],
                                       rtol=1e-13, atol=1e-13)
        for (previous, fitness), (current, _) in zip(snapshots, snapshots[1:]):
            elite = previous[np.argmin(fitness)]
            self.assertTrue(any(np.array_equal(elite, x) for x in current))
        values = [h["best_fitness"] for h in result.history]
        self.assertTrue(np.all(np.diff(values) <= 0))
        self.assertTrue(all(h["population_size"] == 10 for h in result.history))
        self.assertAlmostEqual(hgbat(result.best_x), result.best_fitness)
        self.assertEqual(result.history[-1]["global_best_fitness"], result.best_fitness)

    def test_seed_reproducibility_and_variability(self):
        for runner in (ga.run_genetic_algorithm, rs.run_random_search):
            first, repeated, other = [runner(self.config, seed) for seed in (42, 42, 43)]
            np.testing.assert_array_equal(first.best_x, repeated.best_x)
            self.assertEqual(first.best_fitness, repeated.best_fitness)
            self.assertEqual(first.history, repeated.history)
            self.assertFalse(np.array_equal(first.best_x, other.best_x))

    def test_global_best_without_elitism(self):
        evaluated = []

        def track(x):
            value = hgbat(x)
            evaluated.append(value)
            return value

        with patch.object(ga, "hgbat", side_effect=track):
            result = ga.run_genetic_algorithm(replace(self.config, elite_count=0), 7)
        self.assertEqual(result.best_fitness, min(evaluated))

    def test_configurable_dimension_and_multiple_elites(self):
        config = replace(self.config, dimension=3, elite_count=4,
                         lower_bound=-2, upper_bound=4)
        result = ga.run_genetic_algorithm(config, 0)
        self.assertTrue(is_feasible(result.best_x, 3, -2, 4))
        self.assertEqual(result.evaluations, 123)
        self.assertTrue(np.all(np.diff([h["best_fitness"] for h in result.history]) <= 0))

    def test_invalid_configs(self):
        for changes in (
            {"dimension": 0}, {"population_size": 1}, {"population_size": 10.5},
            {"tournament_size": 0}, {"tournament_size": 101}, {"elite_count": 100},
            {"elite_count": -1}, {"evaluation_budget": 99}, {"mutation_sigma": -1},
            {"mutation_sigma": np.inf}, {"lower_bound": 15}, {"upper_bound": np.nan},
            {"mutation_probability": 1.1}, {"crossover_probability": np.nan},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ga.GAConfig(**changes)


class ExperimentTests(unittest.TestCase):
    def test_configs_differ_only_in_sigma(self):
        base, runs = load_config(Path(__file__).resolve().parents[1] / "config/default.json")
        configs = experiment_configs(base)
        first, second = asdict(configs["ga_config_a"]), asdict(configs["ga_config_b"])
        self.assertEqual([key for key in first if first[key] != second[key]], ["mutation_sigma"])
        self.assertEqual(first["mutation_sigma"], 1.5)
        self.assertEqual(second["mutation_sigma"], 3)
        self.assertEqual(runs, 20)
        self.assertTrue(all(config.evaluation_budget == 50_000 for config in configs.values()))

    def test_sample_standard_deviation(self):
        summary = summarize_results({"test": pd.DataFrame({"best_fitness": [1, 3]})})
        self.assertAlmostEqual(summary.iloc[0]["std"], np.sqrt(2))

    def test_smoke_outputs_in_temporary_directory(self):
        config = ga.GAConfig(population_size=10, evaluation_budget=31)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = run_experiments(config, 2, root)
            self.assertEqual(len(summary), 3)
            for method in experiment_configs(config):
                frame = pd.read_csv(root / "results/raw" / f"{method}.csv")
                self.assertEqual(frame["seed"].tolist(), [0, 1])
                self.assertTrue((frame["evaluations"] == 31).all())
                for _, row in frame.iterrows():
                    x = row[[f"x{i}" for i in range(1, 8)]].to_numpy(dtype=float)
                    self.assertAlmostEqual(row["best_fitness"], hgbat(x))
                if method != "random_search":
                    history = pd.read_csv(root / "results/raw" / f"convergence_{method}.csv")
                    self.assertTrue((history.groupby("seed")["evaluation_count"].max() == 31).all())
            for filename in ("convergence.png", "final_comparison.png"):
                self.assertTrue((root / "figures" / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue((root / "results/summary/statistics.csv").is_file())
            self.assertTrue((root / "results/summary/experiment_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
