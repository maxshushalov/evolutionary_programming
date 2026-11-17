import random
import statistics
import time
from game import choose, entropy, play


def random_genes(rng):
    return (rng.randint(1, 3), *(rng.random() for _ in range(4)), rng.random() * 0.5)


def evaluate(genes, data, time_scale, time_repeats=2):
    wins = draws = losses = nodes = 0
    breakdown = {}
    for match in data['matches']:
        outcome, count = play(genes, match['opponent'], match['sign'], match['agent_seed'], match['opponent_seed'])
        nodes += count
        wins += outcome == 1
        draws += outcome == 0
        losses += outcome == -1
        key = f"{match['opponent']}_{match['sign']}"
        entry = breakdown.setdefault(key, {'wins': 0, 'draws': 0, 'losses': 0})
        entry['wins' if outcome == 1 else 'draws' if outcome == 0 else 'losses'] += 1
    entropies = [entropy(row['board'].copy(), row['player'], genes)[0] for row in data['positions']]
    timing = []
    for repeat in range(time_repeats + 1):
        rng = random.Random(91000 + repeat)
        started = time.perf_counter_ns()
        for row in data['positions']:
            choose(row['board'].copy(), row['player'], genes, rng)
        elapsed = time.perf_counter_ns() - started
        if repeat:
            timing.append(elapsed / len(data['positions']) / 1e9)
    time_value = statistics.median(timing)
    win_rate = wins / len(data['matches'])
    diversity = statistics.mean(entropies)
    return {
        'genes': list(genes), 'wins': wins, 'draws': draws, 'losses': losses,
        'win_rate': win_rate, 'diversity': diversity, 'time_seconds': time_value,
        'time_samples': timing, 'nodes': nodes, 'breakdown': breakdown,
        'objectives': [1 - win_rate, time_value / (time_value + time_scale), 1 - diversity],
    }
