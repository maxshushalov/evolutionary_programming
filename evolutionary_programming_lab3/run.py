import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from algorithm import WEIGHTS, optimize, update_archive
from analysis import ROOT, hypervolume, samples, select_three, summarize
from data_setup import create
from game import FIXED_AGENT
from model import evaluate


def load_inputs():
    if not (ROOT / 'data' / 'train.json').exists():
        create()
    config = json.loads((ROOT / 'config.json').read_text())
    train = json.loads((ROOT / 'data' / 'train.json').read_text())
    test = json.loads((ROOT / 'data' / 'test.json').read_text())
    return config, train, test


def calibrate(config, train):
    calibration = ROOT / 'data' / 'time_scale.json'
    if calibration.exists():
        return json.loads(calibration.read_text())['time_scale']
    result = evaluate(FIXED_AGENT, train, 1, config['time_repeats'])
    scale = result['time_seconds']
    calibration.write_text(json.dumps({'time_scale': scale, 'samples': result['time_samples']}, indent=2))
    return scale


def run_one(method, seed, config, train, test, time_scale, directory):
    started = time.perf_counter()
    budget = config['evaluation_budget']
    population = config['population_size']
    rate = config['mutation_rates'][method]
    scorer = lambda genes: evaluate(genes, train, time_scale, config['time_repeats'])
    if method == 'weighted':
        evaluations = 0
        archive = []
        history = []
        all_candidates = []
        for index, weights in enumerate(WEIGHTS):
            result = optimize(weights, seed * 100 + index, budget // 6, population, rate, scorer, config['checkpoint'])
            for item in result['evaluated']:
                item['evaluation'] += evaluations
                all_candidates.append(item)
                archive = update_archive(archive, item)
                if item['evaluation'] % config['checkpoint'] == 0 or item['evaluation'] == budget:
                    history.append({'evaluations': item['evaluation'], 'front': [row['objectives'] for row in archive]})
            evaluations += result['evaluations']
    else:
        result = optimize('nsga2', seed, budget, population, rate, scorer, config['checkpoint'])
        evaluations = result['evaluations']
        archive = result['archive']
        history = result['history']
        all_candidates = result['evaluated']
    chosen = select_three(archive)
    tested = [evaluate(tuple(item['genes']), test, time_scale, config['time_repeats']) for item in chosen]
    record = {'method': method, 'seed': seed, 'evaluations': evaluations, 'seconds': time.perf_counter() - started,
              'archive': archive, 'history': history, 'evaluated': all_candidates, 'selected_test': tested}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'run_{method}_{seed:02d}.json').write_text(json.dumps(record, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--single', choices=('nsga2', 'nsga2_high_mutation', 'weighted'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    config, train, test = load_inputs()
    time_scale = calibrate(config, train)
    if args.pilot:
        config['evaluation_budget'] = 48
        config['checkpoint'] = 24
        methods, seeds = ('nsga2', 'weighted'), (1000,)
        directory = ROOT / 'results' / 'pilot'
    elif args.single:
        methods, seeds = (args.single,), (args.seed,)
        directory = ROOT / 'results' / 'single'
    elif args.all:
        methods = tuple(config['mutation_rates'])
        seeds = range(config['runs'])
        directory = ROOT / 'results' / 'experiment'
    else:
        parser.print_help()
        return
    if any(directory.glob('run_*.json')):
        raise SystemExit(f'Каталог результатов уже содержит запуски: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'config': config, 'time_scale': time_scale, 'python': sys.version, 'platform': platform.platform()}
    (directory / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    for method in methods:
        for seed in seeds:
            record = run_one(method, seed, config, train, test, time_scale, directory)
            print(method, seed, record['evaluations'], f"{record['seconds']:.1f}s", flush=True)
    print(json.dumps(summarize(directory, config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
