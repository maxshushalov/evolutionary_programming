import copy
import itertools
import math
import random
import unittest
from unittest.mock import patch

import algorithms
from algorithms import geometric_solution, local_neighbor, mutate, order_crossover, random_solution, run_algorithm
from problem import (decode, encode, evaluate, fitness_key, generate_instance, load_json,
                     prepare_instance, tiny_instance, validate_solution)


def independent_score(instance, routes):
    length = 0.0
    capacity_excess = 0.0
    lateness = 0.0
    overtime = 0.0
    for route in routes:
        ids = [0] + route + [0]
        t = 0.0
        for source, target in zip(ids, ids[1:]):
            a = instance["clients"][source]
            b = instance["clients"][target]
            travel = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
            length += travel
            t += travel
            if target:
                t = max(t, b["a"])
                lateness += max(0.0, t - b["b"])
                t += b["service"]
        capacity_excess += max(0, sum(instance["clients"][i]["demand"] for i in route) - instance["capacity"])
        overtime += max(0, t - instance["horizon"])
    return length, capacity_excess, lateness, overtime


class Lab4Tests(unittest.TestCase):
    def setUp(self):
        self.instance = tiny_instance()
        self.params = {"population_size": 10, "elite_size": 2, "tournament_size": 3,
                       "crossover_rate": 0.9, "mutation_rate": 0.3, "evaluation_budget": 83,
                       "local_search_rate": 1.0, "local_search_steps": 5,
                       "history_interval": 10, "initialization": "random"}

    def test_hand_calculation_and_boundaries(self):
        instance = prepare_instance({"vehicles": 2, "capacity": 2, "horizon": 12, "clients": [
            {"x": 0, "y": 0, "demand": 0, "service": 0, "a": 0, "b": 12},
            {"x": 3, "y": 4, "demand": 2, "service": 2, "a": 5, "b": 5}]})
        solution = encode([[1], []])
        score = evaluate(instance, solution, details=True)
        self.assertTrue(score["feasible"])
        self.assertEqual(score["distance"], 10)
        self.assertEqual(score["routes"][0]["return_time"], 12)
        self.assertEqual(score["routes"][1]["return_time"], 0)
        instance["clients"][1]["a"] = 7
        instance["clients"][1]["b"] = 7
        instance["horizon"] = 14
        self.assertEqual(evaluate(instance, solution, True)["routes"][0]["visits"][0]["start"], 7)
        instance["clients"][1]["b"] = 6
        instance["horizon"] = 13
        instance["capacity"] = 1
        score = evaluate(instance, solution)
        self.assertEqual((score["capacity_excess"], score["lateness"], score["return_excess"]), (1, 1, 1))
        self.assertAlmostEqual(score["violation"], 1 + 2 / 13)

    def test_independent_calculation(self):
        rng = random.Random(17)
        for _ in range(100):
            solution = random_solution(6, 2, rng)
            score = evaluate(self.instance, solution)
            expected = independent_score(self.instance, decode(solution))
            for key, value in zip(["distance", "capacity_excess", "lateness", "return_excess"], expected):
                self.assertAlmostEqual(score[key], value)

    def test_operators_and_parents(self):
        rng = random.Random(20)
        first = random_solution(6, 2, rng)
        second = random_solution(6, 2, rng)
        saved = copy.deepcopy((first, second))
        for _ in range(200):
            for child in [order_crossover(first, second, rng), mutate(first, rng), local_neighbor(first, rng)]:
                validate_solution(self.instance, child)
                self.assertEqual(encode(decode(child)), child)
            self.assertEqual((first, second), saved)

    def test_invalid_encoding(self):
        with self.assertRaises(ValueError):
            validate_solution(self.instance, {"permutation": [1] * 6, "route_lengths": [3, 3]})
        with self.assertRaises(ValueError):
            validate_solution(self.instance, {"permutation": list(range(1, 7)), "route_lengths": [-1, 7]})

    def test_feasibility_priority(self):
        good = {"feasible": True, "distance": 100, "violation": 0}
        bad = {"feasible": False, "distance": 1, "violation": 0.0001}
        self.assertLess(fitness_key(good), fitness_key(bad))

    def test_budget_and_rejected_neighbors(self):
        with patch("algorithms.evaluate", wraps=evaluate) as counted:
            result = run_algorithm(self.instance, "memetic", 34, self.params)
        self.assertEqual(counted.call_count, 83)
        self.assertEqual(result["evaluations"], 83)
        self.assertGreater(result["local_evaluations"], result["accepted_local"])
        self.assertEqual(result["history"][-1]["evaluations"], 83)
        for a, b in zip(result["history"], result["history"][1:]):
            if a["feasible"]:
                self.assertTrue(b["feasible"])
                self.assertLessEqual(b["distance"], a["distance"])

    def test_partial_generation_best_preserved(self):
        params = dict(self.params, evaluation_budget=11, local_search_steps=5)
        scores = []
        def track(instance, solution):
            score = evaluate(instance, solution)
            scores.append(score)
            return score
        with patch("algorithms.evaluate", side_effect=track):
            result = run_algorithm(self.instance, "memetic", 8, params)
        self.assertEqual(fitness_key(result["best"]["score"]), min(map(fitness_key, scores)))
        self.assertEqual(len(scores), 11)

    def test_reproducibility_and_shared_initialization(self):
        first = run_algorithm(self.instance, "memetic", 10, self.params)
        second = run_algorithm(self.instance, "memetic", 10, self.params)
        first.pop("seconds")
        second.pop("seconds")
        self.assertEqual(first, second)
        params = dict(self.params, evaluation_budget=10)
        ga = run_algorithm(self.instance, "ga", 10, params)
        ma = run_algorithm(self.instance, "memetic", 10, params)
        self.assertEqual(ga["best"], ma["best"])

    def test_generator(self):
        for n, k, seed in [(20, 3, 202620), (40, 5, 202640), (60, 7, 202660)]:
            instance, witness = generate_instance(n, k, seed)
            prepare_instance(instance)
            validate_solution(instance, witness)
            self.assertTrue(evaluate(instance, witness)["feasible"])
            self.assertTrue(all(value == 0 for value in independent_score(instance, decode(witness))[1:]))

    def test_exact_optimum_independently(self):
        best = math.inf
        for order in itertools.permutations(range(1, 7)):
            for cut in range(7):
                values = independent_score(self.instance, [list(order[:cut]), list(order[cut:])])
                if all(value == 0 for value in values[1:]):
                    best = min(best, values[0])
        expected = 2 * (1 + 1 + math.sqrt(2) + math.sqrt(10))
        self.assertAlmostEqual(best, expected)
        if __import__("pathlib").Path("results/exact_control.json").exists():
            saved = load_json("results/exact_control.json")
            self.assertEqual(saved["checked"], 5040)
            self.assertAlmostEqual(saved["best"]["score"]["distance"], best)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run_algorithm(self.instance, "ga", 1, dict(self.params, evaluation_budget=1))

    def test_geometric_initialization(self):
        rng = random.Random(99)
        for _ in range(100):
            validate_solution(self.instance, geometric_solution(self.instance, rng))
        params = dict(self.params, initialization="mixed", evaluation_budget=10)
        first = run_algorithm(self.instance, "ga", 10, params)
        second = run_algorithm(self.instance, "memetic", 10, params)
        self.assertEqual(first["best"], second["best"])

    def test_summary_with_failures(self):
        from plots import summarize
        result = run_algorithm(self.instance, "ga", 10, self.params)
        result["instance"] = "tiny"
        result["best"]["score"]["feasible"] = False
        summary = summarize([result])[0]
        self.assertEqual(summary["feasible"], 0)
        self.assertIsNone(summary["mean"])
        result["best"]["score"]["feasible"] = True
        summary = summarize([result])[0]
        self.assertEqual(summary["feasible"], 1)
        self.assertIsNone(summary["std"])
        summary = summarize([result, result])[0]
        self.assertEqual(summary["std"], 0)


if __name__ == "__main__":
    unittest.main()
