"""Независимая проверка сохранённых меню, статистики и полного бюджета серии."""

import csv
import hashlib
import json
import unittest

import numpy as np
from model import ROOT, load_catalog, evaluate_menu

OUTPUT = ROOT / "results" / "experiment"


@unittest.skipUnless((OUTPUT / "metadata.json").exists(), "Сначала выполните полную серию")
class SavedResultsTests(unittest.TestCase):
    def test_all_runs_menus_and_budgets(self):
        metadata = json.loads((OUTPUT / "metadata.json").read_text())
        config = metadata["config"]
        catalog = load_catalog()
        total = 0
        for method in metadata["methods"]:
            self.assertGreaterEqual(len(metadata["seeds"]), 20)
            for seed in metadata["seeds"]:
                run = json.loads((OUTPUT / "raw" / f"{method}_{seed:02d}.json").read_text())
                self.assertEqual(run["evaluations"], config["evaluation_budget"])
                self.assertEqual(run["candidate_evaluations"] + run["repair_evaluations"], run["evaluations"])
                self.assertLessEqual(run["repair_accepted"], run["repair_evaluations"])
                total += run["evaluations"]
                last_cost = float("inf")
                for row in run["history"]:
                    if row["best_cost"] is not None:
                        self.assertLessEqual(row["best_cost"], last_cost)
                        last_cost = row["best_cost"]
                self.assertEqual(run["history"][-1]["evaluations"], config["evaluation_budget"])
                last_score = float("inf")
                for row in run["population_history"]:
                    self.assertEqual(row["population_size"], config["population_size"])
                    self.assertLessEqual(row["best_score"], last_score)
                    last_score = row["best_score"]
                if run["best_menu"] is None:
                    self.assertIsNone(run["best_result"])
                    continue
                menu = np.array(run["best_menu"])
                checked = evaluate_menu(menu, catalog, config)
                self.assertTrue(checked["feasible"])
                self.assertEqual(checked["total_cost"], run["best_result"]["total_cost"])
                self.assertEqual(last_cost, checked["total_cost"])
                # Дополнительная независимая проверка ограничений без вызова модели.
                counts = [0] * len(catalog["dishes"])
                cost = 0
                for day in menu:
                    calories = 0
                    for index, chosen in enumerate(day):
                        if chosen:
                            dish = catalog["dishes"][index]
                            counts[index] += 1
                            cost += dish["price_kopecks"]
                            calories += dish["calories"]
                    self.assertGreaterEqual(calories, config["min_calories"])
                    self.assertLessEqual(calories, config["max_calories"])
                self.assertLessEqual(cost, config["budget_kopecks"])
                self.assertGreaterEqual(sum(count > 0 for count in counts), config["min_unique"])
                self.assertLessEqual(max(counts), config["max_frequency"])
        self.assertEqual(total, metadata["total_evaluations"])

    def test_statistics_and_artifacts(self):
        with open(OUTPUT / "runs.csv", newline="") as file:
            runs = list(csv.DictReader(file))
        with open(OUTPUT / "statistics.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            costs = []
            selected = [run for run in runs if run["method"] == row["method"]]
            for run in selected:
                if run["cost_kopecks"]:
                    costs.append(int(run["cost_kopecks"]) / 100)
            self.assertEqual(int(row["successes"]), len(costs))
            self.assertAlmostEqual(float(row["success_rate"]), len(costs) / len(selected))
            if costs:
                self.assertEqual(float(row["best_rub"]), min(costs))
                self.assertAlmostEqual(float(row["mean_rub"]), sum(costs) / len(costs))
                self.assertEqual(float(row["median_rub"]), np.median(costs))
                self.assertEqual(float(row["worst_rub"]), max(costs))
            if len(costs) > 1:
                self.assertAlmostEqual(float(row["sample_std_rub"]), np.std(costs, ddof=1))
        for name in ["comparison", "convergence", "feasibility", "best_menu"]:
            path = OUTPUT / "figures" / f"{name}.png"
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_source_versions_match(self):
        metadata = json.loads((OUTPUT / "metadata.json").read_text())
        # Сохранённый поиск зависит от алгоритма, модели, параметров и каталога.
        # CLI и генератор каталога не участвуют в проверке готовой серии.
        for name in ["algorithm.py", "model.py", "config.json", "data/dishes.csv"]:
            digest = metadata["source_sha256"][name]
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        digest = metadata.get("analysis_sha256", metadata["source_sha256"]["analysis.py"])
        self.assertEqual(hashlib.sha256((ROOT / "analysis.py").read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
