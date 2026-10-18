"""Проверки математической модели, операторов и честного расходования бюджета."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import algorithm
from algorithm import METHODS, crossover, mutate, new_state, evaluate_counted, repair_menu, run_search
from data_setup import generate_catalog
from model import ROOT, load_config, load_catalog, evaluate_menu, decode_menu


class LabTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.catalog = load_catalog()
        self.examples = json.loads((ROOT / "data" / "examples.json").read_text())["examples"]

    def test_catalog_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalog.csv"
            generate_catalog(path, self.config["data_seed"])
            self.assertEqual(path.read_bytes(), (ROOT / "data" / "dishes.csv").read_bytes())
        self.assertGreaterEqual(len(self.catalog["dishes"]), 30)

    def test_examples_and_independent_sums(self):
        for index, example in enumerate(self.examples):
            menu = np.array(example["menu"])
            result = evaluate_menu(menu, self.catalog, self.config)
            self.assertEqual(result["feasible"], index < 2)
            # Прямые циклы здесь — независимая проверка матричных сумм модели.
            total = 0
            frequency = [0] * menu.shape[1]
            for day in range(menu.shape[0]):
                calories = 0
                for dish in range(menu.shape[1]):
                    if menu[day, dish]:
                        calories += self.catalog["dishes"][dish]["calories"]
                        total += self.catalog["dishes"][dish]["price_kopecks"]
                        frequency[dish] += 1
                self.assertEqual(calories, result["daily_calories"][day])
            self.assertEqual(total, result["total_cost"])
            self.assertEqual(frequency, result["frequency"].tolist())

    def test_inclusive_boundaries_and_kopecks(self):
        catalog = {"dishes": [{"name": "Тест"}], "prices": np.array([12345]), "calories": np.array([1800])}
        config = dict(self.config, days=1, min_unique=1, max_frequency=1, budget_kopecks=12345)
        menu = np.ones((1, 1), dtype=int)
        self.assertTrue(evaluate_menu(menu, catalog, config)["feasible"])
        catalog["calories"][0] = 2200
        self.assertTrue(evaluate_menu(menu, catalog, config)["feasible"])
        catalog["calories"][0] = 2201
        self.assertEqual(evaluate_menu(menu, catalog, config)["calorie_excess"][0], 1)
        catalog["prices"][0] += 1
        self.assertEqual(evaluate_menu(menu, catalog, config)["budget_excess"], 1)

    def test_reject_bad_genotype_and_config(self):
        with self.assertRaises(ValueError):
            evaluate_menu(np.zeros((2, 30)), self.catalog, self.config)
        with self.assertRaises(ValueError):
            evaluate_menu(np.full((7, 30), 0.5), self.catalog, self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            config = dict(self.config, elite_count=100)
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_mutations_and_crossover_do_not_modify_parents(self):
        first = np.array(self.examples[0]["menu"])
        second = np.array(self.examples[1]["menu"])
        original = first.copy()
        rng = np.random.default_rng(10)
        child = crossover(first, second, rng, 1)
        for day in range(7):
            self.assertTrue(np.array_equal(child[day], first[day]) or np.array_equal(child[day], second[day]))
        flipped = mutate(first, rng, dict(self.config, bit_mutation_rate=1), "bit")
        np.testing.assert_array_equal(flipped, 1 - first)
        replaced = mutate(first, rng, dict(self.config, swap_probability=1), "swap")
        np.testing.assert_array_equal(replaced.sum(axis=1), first.sum(axis=1))
        self.assertEqual(np.count_nonzero(replaced != first), 2)
        np.testing.assert_array_equal(first, original)

    def test_every_repair_probe_counts_even_if_unchanged(self):
        config = dict(self.config, evaluation_budget=8, repair_attempts=100)
        menu = np.zeros((7, 30), dtype=np.int8)
        state = new_state(config)
        result = evaluate_counted(menu, self.catalog, config, state, "candidate")
        # Ничего не исправляющие пробы обязаны израсходовать оставшиеся 7 оценок.
        with patch("algorithm.repair_proposal", return_value=menu.copy()):
            repair_menu(menu, result, np.random.default_rng(0), self.catalog, config, state)
        self.assertEqual(state["evaluations"], 8)
        self.assertEqual(state["repair_evaluations"], 7)
        self.assertEqual(state["repair_accepted"], 0)
        self.assertIsNone(state["best_menu"])
        with self.assertRaises(RuntimeError):
            evaluate_counted(menu, self.catalog, config, state, "candidate")

    def test_exact_budget_actual_calls_and_partial_generation(self):
        config = dict(self.config, evaluation_budget=237, history_interval=50, repair_attempts=8)
        for method in METHODS:
            with self.subTest(method=method):
                with patch("algorithm.evaluate_menu", wraps=evaluate_menu) as counted:
                    result = run_search(method, 8, self.catalog, config)
                self.assertEqual(counted.call_count, 237)
                self.assertEqual(result["evaluations"], 237)
                self.assertEqual(result["candidate_evaluations"] + result["repair_evaluations"], 237)
                for row in result["population_history"]:
                    self.assertEqual(row["population_size"], 100)
                if result["best_menu"] is not None:
                    self.assertTrue(evaluate_menu(result["best_menu"], self.catalog, config)["feasible"])

    def test_seed_reproducibility_and_initial_population(self):
        config = dict(self.config, evaluation_budget=350, history_interval=50)
        for method in METHODS:
            first = run_search(method, 4, self.catalog, config)
            second = run_search(method, 4, self.catalog, config)
            self.assertEqual(first["history"], second["history"])
            self.assertEqual(first["repair_evaluations"], second["repair_evaluations"])
            np.testing.assert_equal(first["best_menu"], second["best_menu"])
        initial_config = dict(self.config, evaluation_budget=100, history_interval=100)
        initial_scores = []
        for method in list(METHODS)[:4]:
            result = run_search(method, 1, self.catalog, initial_config)
            initial_scores.append(result["population_history"][0]["best_score"])
        self.assertEqual(len(set(initial_scores)), 1)

    def test_decoding(self):
        menu = np.array(self.examples[0]["menu"])
        days = decode_menu(menu, self.catalog, self.config)
        self.assertEqual(len(days), 7)
        self.assertEqual(sum(day["cost_kopecks"] for day in days), self.examples[0]["evaluation"]["total_cost"])


if __name__ == "__main__":
    unittest.main()
