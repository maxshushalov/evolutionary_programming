import math
import random
import unittest
from algorithm import child, crowding, dominates, fronts, optimize
from analysis import hypervolume
from game import choose, entropy, heuristic, play, winner
from model import random_genes


class Lab3Tests(unittest.TestCase):
    def test_game_and_heuristic(self):
        self.assertEqual(winner([1, 1, 1, 0, -1, 0, 0, 0, -1]), 1)
        self.assertEqual(winner([1, -1, 1, 1, -1, -1, -1, 1, 1]), 0)
        genes = (1, 0, 1, 0, 1, 0)
        board = [1, 1, 0, -1, 0, 0, 0, 0, 0]
        self.assertGreater(heuristic(board, 1, genes), 0)
        move, _ = choose(board, 1, genes, random.Random(1))
        self.assertEqual(move, 2)
        self.assertEqual(board, [1, 1, 0, -1, 0, 0, 0, 0, 0])

    def test_repeatability_and_entropy(self):
        genes = (2, 1, 1, 1, 1, 0)
        a = play(genes, 'tactical', 1, 10, 11)
        self.assertEqual(a, play(genes, 'tactical', 1, 10, 11))
        value, _ = entropy([0] * 9, 1, (1, 0, 0, 0, 0, 0.5))
        self.assertAlmostEqual(value, 1)
        self.assertTrue(0 <= entropy([0] * 9, 1, genes)[0] <= 1)

    def test_fronts_and_crowding(self):
        rows = [{'objectives': x} for x in ((1, 2, 2), (2, 1, 2), (2, 2, 2), (1, 2, 2))]
        self.assertTrue(dominates(rows[0], rows[2]))
        self.assertFalse(dominates(rows[0], rows[3]))
        self.assertEqual(set(fronts(rows)[0]), {0, 1, 3})
        self.assertEqual(crowding(rows, [0, 3]), {0: math.inf, 3: math.inf})
        constant = [{'objectives': (1, 1, 1)} for _ in range(3)]
        self.assertEqual(crowding(constant, [0, 1, 2]), {0: 0.0, 1: 0.0, 2: 0.0})

    def test_operators_and_budget(self):
        rng = random.Random(12)
        first = random_genes(rng)
        second = random_genes(rng)
        old = first
        offspring = child(first, second, rng, 1)
        self.assertEqual(first, old)
        self.assertIn(offspring[0], (1, 2, 3))
        self.assertTrue(all(0 <= value <= 1 for value in offspring[1:5]))
        self.assertTrue(0 <= offspring[5] <= 0.5)
        calls = []
        def evaluate(genes):
            calls.append(genes)
            return {'genes': list(genes), 'objectives': [genes[1], genes[2], genes[3]]}
        result = optimize('nsga2', 3, 13, 5, 1 / 6, evaluate, 4)
        self.assertEqual(len(calls), 13)
        self.assertEqual(result['evaluations'], 13)
        self.assertEqual(result['history'][-1]['evaluations'], 13)
        again = optimize('nsga2', 3, 13, 5, 1 / 6, evaluate, 4)
        self.assertEqual([x['genes'] for x in result['evaluated']], [x['genes'] for x in again['evaluated']])

    def test_hypervolume(self):
        points = [(0.5, 0.5, 0.5), (1, 1, 1)]
        self.assertEqual(hypervolume([], points, 1)[0], 0)
        self.assertEqual(hypervolume([(0.5, 0.5, 0.5)], points, 1)[0], 1)
        self.assertEqual(hypervolume([(0.5, 0.5, 0.5), (0.9, 0.9, 0.9)], points, 1)[0], 1)


if __name__ == '__main__':
    unittest.main()
