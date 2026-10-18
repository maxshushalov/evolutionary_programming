"""Проверка полных артефактов после python -m src.experiments."""

from dataclasses import asdict
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.experiments import experiment_configs, load_config, summarize_results
from src.objective import hgbat, is_feasible

ROOT = Path(__file__).resolve().parents[1]


class SavedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "results/summary/experiment_metadata.json"
        if not path.exists():
            raise unittest.SkipTest("Полные эксперименты ещё не выполнены")
        cls.metadata = json.loads(path.read_text(encoding="utf-8"))
        cls.results = {
            method: pd.read_csv(ROOT / "results/raw" / f"{method}.csv", float_precision="round_trip")
            for method in experiment_configs(load_config(ROOT / "config/default.json")[0])
        }

    def test_full_experiment_parameters(self):
        config, runs = load_config(ROOT / "config/default.json")
        self.assertEqual(runs, 20)
        self.assertEqual(self.metadata["seeds"], list(range(20)))
        self.assertEqual(self.metadata["total_objective_evaluations"], 3_000_000)
        self.assertEqual(self.metadata["configurations"], {
            name: asdict(value) for name, value in experiment_configs(config).items()
        })
        for frame in self.results.values():
            self.assertEqual(frame["seed"].tolist(), list(range(20)))
            self.assertTrue((frame["evaluations"] == 50_000).all())

    def test_coordinates_and_recomputed_statistics(self):
        for frame in self.results.values():
            for _, row in frame.iterrows():
                x = row[[f"x{i}" for i in range(1, 8)]].to_numpy(dtype=float)
                self.assertTrue(is_feasible(x))
                self.assertAlmostEqual(row["best_fitness"], hgbat(x), places=10)
        saved = pd.read_csv(ROOT / "results/summary/statistics.csv")
        pd.testing.assert_frame_equal(saved, summarize_results(self.results),
                                      check_exact=False, rtol=1e-12, atol=1e-12)

    def test_complete_convergence_histories(self):
        for method in ("ga_config_a", "ga_config_b"):
            history = pd.read_csv(ROOT / "results/raw" / f"convergence_{method}.csv")
            self.assertEqual(len(history), 20 * 506)
            self.assertTrue((history["population_size"] == 100).all())
            for seed, run in history.groupby("seed"):
                self.assertEqual(run["generation"].tolist(), list(range(506)))
                self.assertEqual(run["evaluation_count"].tolist(),
                                 [100 + 99 * generation for generation in range(505)] + [50_000])
                self.assertTrue(np.all(np.diff(run["global_best_fitness"]) <= 0))
                np.testing.assert_allclose(run["best_fitness"], run["global_best_fitness"])
                self.assertTrue((run["best_fitness"] <= run["mean_fitness"] + 1e-12).all())
                self.assertTrue((run["mean_fitness"] <= run["worst_fitness"] + 1e-12).all())
                final = self.results[method].set_index("seed").loc[seed]
                self.assertEqual(final["generations"], 505)
                self.assertAlmostEqual(run.iloc[-1]["global_best_fitness"], final["best_fitness"])

    def test_figures_and_notebook_exist(self):
        for name in ("convergence", "final_comparison"):
            self.assertTrue((ROOT / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        notebook = json.loads((ROOT / "notebooks/lab1_analysis.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook>", "exec")


if __name__ == "__main__":
    unittest.main()
