import csv
import json
import math
import random
import statistics
from pathlib import Path
from algorithm import update_archive

ROOT = Path(__file__).resolve().parent


def samples(count, seed, reference=1.1):
    rng = random.Random(seed)
    return [tuple(rng.random() * reference for _ in range(3)) for _ in range(count)]


def hypervolume(front, points, reference=1.1):
    if not front:
        return 0.0, 0.0
    objectives = [item['objectives'] if isinstance(item, dict) else item for item in front]
    covered = sum(any(all(a <= b for a, b in zip(solution, point)) for solution in objectives) for point in points)
    proportion = covered / len(points)
    volume = reference ** 3
    return volume * proportion, volume * math.sqrt(proportion * (1 - proportion) / len(points))


def select_three(archive):
    if len(archive) < 3:
        return []
    first = min(archive, key=lambda row: (-row['win_rate'], row['objectives'], row['genes']))
    rest = [row for row in archive if row is not first]
    second = min(rest, key=lambda row: (row['time_seconds'], row['objectives'], row['genes']))
    rest = [row for row in rest if row is not second]
    third = min(rest, key=lambda row: (max(row['objectives']), row['objectives'], row['genes']))
    return [first, second, third]


def summarize(directory, config):
    points = samples(config['hypervolume_points'], config['hypervolume_seed'], config['reference_point'])
    rows = []
    histories = {}
    for path in sorted(directory.glob('run_*.json')):
        run = json.loads(path.read_text())
        hv, error = hypervolume(run['archive'], points, config['reference_point'])
        rows.append({'method': run['method'], 'seed': run['seed'], 'hypervolume': hv, 'mc_error': error,
                     'front_size': len(run['archive']), 'evaluations': run['evaluations'], 'seconds': run['seconds']})
        histories[(run['method'], run['seed'])] = [(step['evaluations'], hypervolume(step['front'], points, config['reference_point'])[0]) for step in run['history']]
    if not rows:
        return []
    with (directory / 'runs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in sorted({row['method'] for row in rows}):
        group = [row for row in rows if row['method'] == method]
        values = [row['hypervolume'] for row in group]
        summary.append({'method': method, 'runs': len(group), 'best': max(values), 'mean': statistics.mean(values),
                        'median': statistics.median(values), 'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'worst': min(values), 'mean_seconds': statistics.mean(row['seconds'] for row in group),
                        'mean_front_size': statistics.mean(row['front_size'] for row in group)})
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    make_plots(directory, rows, histories)
    plot_distribution(directory, rows)
    return summary


def make_plots(directory, rows, histories):
    methods = sorted({row['method'] for row in rows})
    colors = {'nsga2': '#215fa8', 'nsga2_high_mutation': '#e07820', 'weighted': '#32a070'}
    width, height = 800, 480
    for method in methods:
        curves = [series for (name, _), series in histories.items() if name == method]
        xs = sorted({x for series in curves for x, _ in series})
        if not xs:
            continue
        ys = [statistics.mean(y for series in curves for x, y in series if x == step) for step in xs]
        points = ' '.join(f'{60 + (x / max(xs)) * 700:.1f},{420 - y / 1.331 * 360:.1f}' for x, y in zip(xs, ys))
        path = directory / f'convergence_{method}.svg'
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480"><rect width="800" height="480" fill="white"/><path d="M60 40 V420 H760" fill="none" stroke="black"/><polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="3"/><text x="60" y="25" font-size="20">Средний гиперобъём: {method}</text><text x="330" y="462">Число оценок</text><text x="8" y="235" transform="rotate(-90 8 235)">Гиперобъём</text><text x="30" y="420">0</text><text x="20" y="50">1.331</text></svg>')


def plot_front(directory, rows):
    median_row = min((row for row in rows if row['method'] == 'nsga2'), key=lambda row: abs(row['hypervolume'] - statistics.median(r['hypervolume'] for r in rows if r['method'] == 'nsga2')))
    run = json.loads((directory / f"run_nsga2_{median_row['seed']:02d}.json").read_text())
    front = run['archive']
    projections = (("win_rate", "time_seconds", "Доля побед", "Время хода, мс"),
                   ("win_rate", "diversity", "Доля побед", "Разнообразие"),
                   ("time_seconds", "diversity", "Время хода, мс", "Разнообразие"))
    for index, (x_name, y_name, x_label, y_label) in enumerate(projections, 1):
        xs = [row[x_name] * (1000 if x_name == 'time_seconds' else 1) for row in front]
        ys = [row[y_name] * (1000 if y_name == 'time_seconds' else 1) for row in front]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dots = ''.join(f'<circle cx="{70 + 650 * (x-x_min)/(x_max-x_min or 1):.1f}" cy="{410 - 340 * (y-y_min)/(y_max-y_min or 1):.1f}" r="5" fill="#215fa8"/>' for x, y in zip(xs, ys))
        (directory / f'front_{index}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480"><rect width="800" height="480" fill="white"/><path d="M70 40 V410 H740" fill="none" stroke="black"/>{dots}<text x="70" y="25" font-size="20">Фронт Парето, seed {median_row["seed"]}</text><text x="330" y="460">{x_label}</text><text x="10" y="230" transform="rotate(-90 10 230)">{y_label}</text><text x="70" y="430">{x_min:.3f}</text><text x="670" y="430">{x_max:.3f}</text><text x="28" y="410">{y_min:.3f}</text><text x="28" y="55">{y_max:.3f}</text></svg>')


def plot_distribution(directory, rows):
    methods = sorted({row['method'] for row in rows})
    low = min(row['hypervolume'] for row in rows)
    high = max(row['hypervolume'] for row in rows)
    pieces = []
    for index, method in enumerate(methods):
        values = sorted(row['hypervolume'] for row in rows if row['method'] == method)
        x = 160 + index * 230
        for value in values:
            y = 400 - 340 * (value - low) / (high - low or 1)
            pieces.append(f'<circle cx="{x}" cy="{y:.1f}" r="4" opacity="0.6" fill="#215fa8"/>')
        median = statistics.median(values)
        y_median = 400 - 340 * (median - low) / (high - low or 1)
        pieces.append(f'<path d="M{x-35} {y_median:.1f} H{x+35}" stroke="#e07820" stroke-width="4"/>')
        pieces.append(f'<text x="{x}" y="440" text-anchor="middle" font-size="13">{method}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480"><rect width="800" height="480" fill="white"/><path d="M50 40 V410 H760" fill="none" stroke="black"/><text x="65" y="25" font-size="20">Гиперобъём по 20 запускам</text><text x="5" y="260" transform="rotate(-90 5 260)">Гиперобъём</text><text x="50" y="40">{high:.3f}</text><text x="50" y="405">{low:.3f}</text>{"".join(pieces)}</svg>'
    (directory / 'hypervolume_distribution.svg').write_text(svg)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    directory = Path(args.directory)
    summary = summarize(directory, config)
    rows = list(csv.DictReader((directory / 'runs.csv').open()))
    for row in rows:
        row['hypervolume'] = float(row['hypervolume'])
        row['seed'] = int(row['seed'])
    if any(row['method'] == 'nsga2' for row in rows):
        plot_front(directory, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


