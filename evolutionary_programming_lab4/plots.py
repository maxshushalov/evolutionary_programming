import argparse
import csv
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problem import file_hash, load_instance, load_json, save_json


ROOT = Path(__file__).resolve().parent
LABELS = {"ga": "ГА", "memetic": "Меметический, 5 шагов", "memetic_steps1": "Меметический, 1 шаг"}
COLORS = {"ga": "#3366aa", "memetic": "#d46929", "memetic_steps1": "#258567"}


def variant(result):
    if result["method"] == "memetic" and result["params"]["local_search_steps"] == 1:
        return "memetic_steps1"
    return result["method"]


def read_runs(folder):
    runs = []
    for path in sorted(Path(folder).glob("*.json")):
        if path.name != "metadata.json":
            runs.append(load_json(path))
    return runs


def group_runs(runs):
    groups = {}
    for result in runs:
        groups.setdefault((result["instance"], variant(result)), []).append(result)
    return groups


def describe(values):
    if not values:
        return {key: None for key in ["min", "mean", "median", "std", "max"]}
    return {"min": min(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else None, "max": max(values)}


def summarize(runs):
    rows = []
    for (instance, method), group in sorted(group_runs(runs).items()):
        feasible = [r for r in group if r["best"]["score"]["feasible"]]
        failed = [r for r in group if not r["best"]["score"]["feasible"]]
        row = {"instance": instance, "method": method, "runs": len(group), "feasible": len(feasible),
               "failed": len(failed), "success_rate": len(feasible) / len(group)}
        row.update(describe([r["best"]["score"]["distance"] for r in feasible]))
        row["failed_violation_mean"] = statistics.mean(r["best"]["score"]["violation"] for r in failed) if failed else None
        for key in ["seconds", "evaluations", "local_evaluations", "generations", "accepted_local"]:
            row[f"{key}_mean"] = statistics.mean(r[key] for r in group)
        rows.append(row)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(runs):
    groups = group_runs(runs)
    rows = []
    for instance in sorted(set(r["instance"] for r in runs)):
        ga = {r["seed"]: r for r in groups[(instance, "ga")]}
        for method in ["memetic", "memetic_steps1"]:
            if (instance, method) not in groups:
                continue
            differences = []
            for result in groups[(instance, method)]:
                base = ga[result["seed"]]["best"]["score"]
                score = result["best"]["score"]
                if score["feasible"] and base["feasible"]:
                    differences.append((base["distance"] - score["distance"]) / base["distance"] * 100)
            row = {"instance": instance, "comparison": f"ga_vs_{method}", "both_feasible": len(differences),
                   "wins": sum(x > 1e-9 for x in differences), "ties": sum(abs(x) <= 1e-9 for x in differences),
                   "losses": sum(x < -1e-9 for x in differences)}
            row.update(describe(differences))
            rows.append(row)
    return rows


def finish(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def convergence(groups, instance, output):
    methods = [method for name, method in groups if name == instance]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for method in methods:
        group = groups[(instance, method)]
        grid = [point["evaluations"] for point in group[0]["history"]]
        matrix = np.array([[p["distance"] if p["distance"] is not None else np.nan for p in r["history"]] for r in group])
        count = np.sum(np.isfinite(matrix), axis=0)
        median = []
        lower = []
        upper = []
        for column in matrix.T:
            values = column[np.isfinite(column)]
            quantiles = np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else [np.nan] * 3
            lower.append(quantiles[0])
            median.append(quantiles[1])
            upper.append(quantiles[2])
        axes[0].plot(grid, count / len(group), label=LABELS[method], color=COLORS[method])
        axes[1].plot(grid, median, label=LABELS[method], color=COLORS[method])
        axes[1].fill_between(grid, lower, upper, alpha=0.15, color=COLORS[method])
        axes[2].plot(grid, count, label=LABELS[method], color=COLORS[method])
    axes[0].set(title=f"{instance}: достижение допустимости", ylabel="Доля допустимых запусков", ylim=(-0.03, 1.03))
    axes[1].set(title="Лучшая длина: медиана и 25–75% среди допустимых", ylabel="Длина маршрутов")
    axes[2].set(ylabel="Число допустимых запусков", xlabel="Вычисления качества")
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.25)
    finish(fig, output / f"convergence_{instance}.png")


def distribution(groups, instance, output):
    methods = [method for name, method in groups if name == instance]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i, method in enumerate(methods, 1):
        group = groups[(instance, method)]
        values = [r["best"]["score"]["distance"] for r in group if r["best"]["score"]["feasible"]]
        if values:
            ax.boxplot([values], positions=[i], widths=0.45, patch_artist=True,
                       boxprops={"facecolor": COLORS[method], "alpha": 0.35}, medianprops={"color": "black"})
            offsets = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(i + offsets, values, s=12, alpha=0.55, color=COLORS[method])
    labels = []
    for method in methods:
        failed = sum(not r["best"]["score"]["feasible"] for r in groups[(instance, method)])
        labels.append(f"{LABELS[method]}\nнеудач: {failed}/{len(groups[(instance, method)])}")
    ax.set_xticks(range(1, len(methods) + 1), labels)
    ax.set(title=f"{instance}: итоговая длина допустимых решений", ylabel="Суммарная длина маршрутов")
    all_values = [r["best"]["score"]["distance"] for method in methods
                  for r in groups[(instance, method)] if r["best"]["score"]["feasible"]]
    if all_values and max(all_values) - min(all_values) < 1e-8:
        ax.set_ylim(min(all_values) - 0.01, max(all_values) + 0.01)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.grid(axis="y", alpha=0.25)
    finish(fig, output / f"distribution_{instance}.png")


def representative(group):
    feasible = [r for r in group if r["best"]["score"]["feasible"]]
    if not feasible:
        return min(group, key=lambda r: (r["best"]["score"]["violation"], r["seed"]))
    median = statistics.median(r["best"]["score"]["distance"] for r in feasible)
    return min(feasible, key=lambda r: (abs(r["best"]["score"]["distance"] - median), r["seed"]))


def route_figure(result, output):
    instance = load_instance(ROOT / "data" / f"{result['instance']}.json")
    clients = instance["clients"]
    routes = result["best"]["validation"]["routes"]
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for i, route in enumerate(routes):
        ids = [0] + route["clients"] + [0]
        ax.plot([clients[j]["x"] for j in ids], [clients[j]["y"] for j in ids],
                marker="o", markersize=4, label=f"Машина {i + 1}, загрузка {route['load']}/{instance['capacity']}")
        for j in route["clients"]:
            ax.annotate(str(j), (clients[j]["x"], clients[j]["y"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.scatter([clients[0]["x"]], [clients[0]["y"]], marker="s", color="black", s=90, zorder=5, label="Склад")
    score = result["best"]["score"]
    ax.set(title=f"{result['instance']}, {LABELS[variant(result)]}, seed={result['seed']}\n"
                 f"L={score['distance']:.2f}, допустимость: {score['feasible']}",
           xlabel="Координата x", ylabel="Координата y", aspect="equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    finish(fig, output / f"routes_{result['instance']}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(clients) * 0.15)),
                             gridspec_kw={"width_ratios": [3, 1]}, layout="constrained")
    y = 0
    ticks = []
    labels = []
    for vehicle, route in enumerate(routes):
        for visit in route["visits"]:
            i = visit["client"]
            axes[0].plot([clients[i]["a"], clients[i]["b"]], [y, y], color="#a8b9cf", linewidth=4)
            axes[0].scatter([visit["start"]], [y], color="#224466" if visit["lateness"] == 0 else "red", s=15, zorder=3)
            ticks.append(y)
            labels.append(f"{vehicle + 1}:{i}")
            y += 1
    axes[0].set_yticks(ticks, labels, fontsize=6)
    axes[0].set(title="Окно и начало обслуживания (точка)", xlabel="Время", ylabel="Машина:клиент")
    axes[1].barh(range(1, len(routes) + 1), [r["load"] for r in routes], color="#547ba5")
    axes[1].axvline(instance["capacity"], linestyle="--", color="red", label="Вместимость")
    axes[1].set(title="Загрузка", xlabel="Единицы груза", ylabel="Машина")
    axes[1].set_yticks(range(1, len(routes) + 1))
    axes[1].legend(fontsize=8)
    finish(fig, output / f"schedule_{result['instance']}.png")


def algorithm_figure(output):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 12))
    ax.axis("off")
    blocks = [
        (10.6, "Данные, параметры и seed"),
        (9.1, "Общая начальная популяция и её оценка"),
        (7.6, "Элита, турнирный отбор родителей"),
        (6.1, "OX, мутация, оценка потомка"),
        (4.6, "Меметический вариант: с вероятностью 0.2\nдо s локальных соседей с учётом бюджета"),
        (3.1, "Обновление лучшего решения и истории\nДобавление потомка в новое поколение"),
        (1.1, "При исчерпании бюджета: сохранить результат")]
    for y, label in blocks:
        box = FancyBboxPatch((1, y - 0.4), 8, 0.9, boxstyle="round,pad=0.08", facecolor="#edf2f8", edgecolor="#47688c")
        ax.add_patch(box)
        ax.text(5, y + 0.05, label, ha="center", va="center", fontsize=10)
    for (upper, _), (lower, _) in zip(blocks, blocks[1:]):
        ax.annotate("", xy=(5, lower + 0.6), xytext=(5, upper - 0.5), arrowprops={"arrowstyle": "->"})
    ax.plot([9.2, 9.7, 9.7, 9.2], [3.1, 3.1, 7.6, 7.6], color="#47688c")
    ax.annotate("", xy=(9, 7.6), xytext=(9.7, 7.6), arrowprops={"arrowstyle": "->"})
    ax.text(9.45, 5.3, "Бюджет остался", rotation=90, ha="center", va="center", fontsize=9)
    ax.text(5.2, 2.2, "Бюджет исчерпан", fontsize=9)
    ax.set_title("Схема генетического и меметического поиска", fontsize=13)
    finish(fig, output / "algorithm.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results" / "main"))
    parser.add_argument("--output", default=str(ROOT / "figures"))
    args = parser.parse_args()
    folder = Path(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = read_runs(folder)
    if not runs:
        raise ValueError("Нет результатов")
    rows = summarize(runs)
    write_csv(folder / "summary.csv", rows)
    write_csv(folder / "paired.csv", paired_comparisons(runs))
    run_rows = []
    for r in runs:
        row = {"instance": r["instance"], "method": variant(r), "seed": r["seed"], **r["best"]["score"]}
        row.update({key: r[key] for key in ["seconds", "evaluations", "local_evaluations", "generations", "first_feasible"]})
        run_rows.append(row)
    write_csv(folder / "runs.csv", run_rows)
    groups = group_runs(runs)
    representatives = []
    for instance in sorted(set(r["instance"] for r in runs)):
        convergence(groups, instance, output)
        distribution(groups, instance, output)
        result = representative(groups[(instance, "memetic")])
        route_figure(result, output)
        representatives.append({"instance": instance, "method": result["method"], "seed": result["seed"],
                                "distance": result["best"]["score"]["distance"]})
    save_json(output / "representatives.json", representatives)
    algorithm_figure(output)
    save_json(output / "manifest.json", {"input_hashes": {p.name: file_hash(p) for p in folder.glob("*.json")},
                                       "plot_source_hash": file_hash(__file__)})
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
