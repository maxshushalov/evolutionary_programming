import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm import dominates
from game import entropy, play

ROOT = Path(__file__).resolve().parents[1]


class SavedSeriesTests(unittest.TestCase):
    def test_saved_runs(self):
        directory = ROOT / 'results' / 'experiment'
        if not (directory / 'summary.json').exists():
            self.skipTest('Полная серия ещё не завершена')
        config = json.loads((ROOT / 'config.json').read_text())
        train = json.loads((ROOT / 'data' / 'train.json').read_text())
        paths = sorted(directory.glob('run_*.json'))
        self.assertEqual(len(paths), config['runs'] * 3)
        for path in paths:
            run = json.loads(path.read_text())
            self.assertEqual(run['evaluations'], config['evaluation_budget'])
            self.assertEqual(len(run['evaluated']), config['evaluation_budget'])
            self.assertEqual(run['history'][-1]['evaluations'], config['evaluation_budget'])
            archive = run['archive']
            self.assertGreaterEqual(len(archive), 3)
            for i, item in enumerate(archive):
                self.assertFalse(any(dominates(other, item) for j, other in enumerate(archive) if i != j))
            item = archive[0]
            genes = tuple(item['genes'])
            outcomes = [play(genes, row['opponent'], row['sign'], row['agent_seed'], row['opponent_seed'])[0] for row in train['matches']]
            self.assertEqual(sum(outcome == 1 for outcome in outcomes), item['wins'])
            diversity = sum(entropy(row['board'].copy(), row['player'], genes)[0] for row in train['positions']) / len(train['positions'])
            self.assertAlmostEqual(diversity, item['diversity'])


if __name__ == '__main__':
    unittest.main()
