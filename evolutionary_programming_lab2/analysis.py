"""Статистика, графики и понятная таблица лучшего меню из сохранённых запусков."""

import json
import hashlib
import os
from pathlib import Path

import numpy as np
from model import ROOT, decode_menu, evaluate_menu, save_json
from run import write_csv


os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"penalty_bit": "Штрафы · биты", "penalty_swap": "Штрафы · замена",
          "combined_bit": "Комбинир. · биты", "combined_swap": "Комбинир. · замена",
          "random_repair": "Случайный + ремонт"}


def relaxed_lower_bound(catalog, config):
    # Ослабленная задача разрешает дроби и игнорирует разнообразие. Её минимум
    # не выше стоимости любого допустимого недельного меню исходной задачи.
    remaining = config["days"] * config["min_calories"]
    total = 0
    order = np.argsort(catalog["prices"] / catalog["calories"], kind="stable")
    for index in order:
        price = int(catalog["prices"][index])
        calories = int(catalog["calories"][index])
        capacity = config["max_frequency"] * calories
        if remaining >= capacity:
            total += config["max_frequency"] * price
            remaining -= capacity
        else:
            total += remaining * price // calories
            return total
    if remaining > 0:
        raise ValueError("Даже ослабленная задача не набирает требуемые калории")
    return total


def statistics(runs, methods):
    """Считает статистику цен только допустимых результатов; неудачи считает отдельно."""
    rows = []

    for method in methods:
        selected = []
        costs = []

        for run in runs:
            if run["method"] == method:
                selected.append(run)
                if run["best_result"] is not None:
                    costs.append(run["best_result"]["total_cost"] / 100)
        row = {"method": method, "runs": len(selected), "successes": len(costs),
               "success_rate": len(costs) / len(selected), "best_rub": None,
               "mean_rub": None, "median_rub": None, "sample_std_rub": None, "worst_rub": None}
        if costs:
            row["best_rub"] = float(np.min(costs))
            row["mean_rub"] = float(np.mean(costs))
            row["median_rub"] = float(np.median(costs))
            row["worst_rub"] = float(np.max(costs))

        if len(costs) >= 2:
            row["sample_std_rub"] = float(np.std(costs, ddof=1))
        row["mean_seconds"] = float(np.mean([run["seconds"] for run in selected]))
        row["mean_repair_evaluations"] = float(np.mean([run["repair_evaluations"] for run in selected]))
        row["mean_generations"] = float(np.mean([run["generations"] for run in selected]))
        rows.append(row)
    return rows


def plot_results(runs, methods, figures, catalog, config, best):

    """Строит сходимость по общему бюджету, долю успехов, распределение и меню."""

    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for method in methods:
        selected = [run for run in runs if run["method"] == method]
        grid = [row["evaluations"] for row in selected[0]["history"]]
        means, lows, highs, success = [], [], [], []
        for position in range(len(grid)):
            values = []
            for run in selected:
                cost = run["history"][position]["best_cost"]
                if cost is not None:
                    values.append(cost / 100)
            success.append(len(values) / len(selected))
            means.append(float(np.mean(values)) if values else np.nan)
            lows.append(min(values) if values else np.nan)
            highs.append(max(values) if values else np.nan)
        line = axes[0].plot(grid, means, label=LABELS[method])[0]
        axes[0].fill_between(grid, lows, highs, color=line.get_color(), alpha=0.09)
        axes[1].plot(grid, success, label=LABELS[method])
    axes[0].set_title("Цена лучшего допустимого меню: среднее и min–max среди успешных запусков")
    axes[0].set_ylabel("Стоимость недели, ₽")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Доля запусков с допустимым меню")
    axes[1].set_xlabel("Все оценки, включая неудачные пробы ремонта")
    axes[1].set_ylim(-0.02, 1.02)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "convergence.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))

    for position, method in enumerate(methods, start=1):
        costs = []
        for run in runs:
            if run["method"] == method and run["best_result"] is not None:
                costs.append(run["best_result"]["total_cost"] / 100)
        if costs:
            axis.boxplot([costs], positions=[position], widths=0.5)
            axis.scatter(np.linspace(position - 0.15, position + 0.15, len(costs)), costs, s=15, zorder=3)
    axis.set_xticks(range(1, len(methods) + 1), [LABELS[method] for method in methods], fontsize=9)
    axis.set_ylabel("Стоимость лучшего допустимого меню, ₽")
    axis.set_title("Итоги серии: каждая точка — отдельный запуск; неудачи исключены из цен")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "comparison.png", dpi=150)
    plt.close(fig)

    if best is not None:
        menu = np.array(best["best_menu"])
        result = evaluate_menu(menu, catalog, config)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        axes[0].imshow(menu, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[0].set_xticks(range(menu.shape[1]), [dish["id"] for dish in catalog["dishes"]], rotation=90)
        axes[0].set_yticks(range(config["days"]), range(1, config["days"] + 1))
        axes[0].set_ylabel("День")
        axes[0].set_title("Лучшее меню: синий — блюдо включено; расшифровка ID в dishes.csv")
        axes[1].bar(range(1, config["days"] + 1), result["daily_calories"], color="#4f87b5")
        axes[1].axhline(config["min_calories"], color="#a33", linestyle="--", label="Границы калорий")
        axes[1].axhline(config["max_calories"], color="#a33", linestyle="--")
        axes[1].set_xlabel("День")
        axes[1].set_ylabel("ккал")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(figures / "best_menu.png", dpi=150)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for method in methods:
        selected = [run for run in runs if run["method"] == method and run["population_history"]]
        if not selected:
            continue
        grid = np.linspace(0, config["evaluation_budget"], 101)
        trajectories = []
        for run in selected:
            trajectory = []
            records = run["population_history"]
            position = 0
            for budget in grid:
                while position + 1 < len(records) and records[position + 1]["evaluations"] <= budget:
                    position += 1
                value = records[position]["feasible_fraction"]
                if budget < records[0]["evaluations"]:
                    value = np.nan
                trajectory.append(value)
            trajectories.append(trajectory)
        # Первую точку до завершения инициализации не отображаем.
        axis.plot(grid[1:], np.mean(np.array(trajectories)[:, 1:], axis=0), label=LABELS[method])
    axis.set_xlabel("Все оценки, включая ремонт")
    axis.set_ylabel("Средняя доля допустимых особей")
    axis.set_title("Состав последней завершённой популяции; случайный метод популяции не имеет")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "feasibility.png", dpi=150)
    plt.close(fig)


def menu_markdown(run, catalog, config):

    """Пишет таблицу блюд, порций и дневных сумм для лучшего запуска."""


    menu = np.array(run["best_menu"])
    result = evaluate_menu(menu, catalog, config)
    lines = ["# Лучшее найденное меню", "", f"Метод: {run['method']}; seed: {run['seed']}.",
             "", "| День | Блюда и стандартные порции | Ккал | Цена, ₽ |",
             "|---|---|---:|---:|"]
    for day in decode_menu(menu, catalog, config):
        names = []
        for dish in day["dishes"]:
            names.append(f"{dish['id']} {dish['name']} ({dish['portion_size']} {dish['portion_unit']})")
        lines.append(f"| {day['day']} | {'; '.join(names)} | {day['calories']} | {day['cost_kopecks'] / 100:.2f} |")
    lines.extend(["", f"Всего: **{result['total_cost'] / 100:.2f} ₽**; разных блюд: **{result['unique_count']}**.",
                  f"Максимальная частота: **{int(result['frequency'].max())}**. Все ограничения выполнены.",
                  "", "| Блюдо | Число дней |", "|---|---:|"])
    for index, dish in enumerate(catalog["dishes"]):
        lines.append(f"| {dish['id']} {dish['name']} | {result['frequency'][index]} |")
    return "\n".join(lines) + "\n"


def summarize(output, catalog):
    """Читает только перечисленные в метаданных запуски; строит таблицы и графики."""
    metadata = json.loads((output / "metadata.json").read_text())
    config = metadata["config"]
    methods = metadata["methods"]
    runs = []
    for method in methods:
        for seed in metadata["seeds"]:
            run = json.loads((output / "raw" / f"{method}_{seed:02d}.json").read_text())
            if run["best_menu"] is not None:
                checked = evaluate_menu(np.array(run["best_menu"]), catalog, config)
                if not checked["feasible"] or checked["total_cost"] != run["best_result"]["total_cost"]:
                    raise ValueError("Сохранённое меню не прошло повторную проверку")
            runs.append(run)
    rows = statistics(runs, methods)
    write_csv(output / "statistics.csv", rows)
    best = None
    for run in runs:
        if run["best_result"] is not None:
            if best is None or run["best_result"]["total_cost"] < best["best_result"]["total_cost"]:
                best = run
    if best is not None:
        (output / "best_menu.md").write_text(menu_markdown(best, catalog, config), encoding="utf-8")
        save_json(output / "best_menu.json", {"method": best["method"], "seed": best["seed"],
                                             "menu": best["best_menu"], "evaluation": best["best_result"]})
    plot_results(runs, methods, output / "figures", catalog, config, best)
    lines = ["# Фактические результаты серии", "", "Цены приведены только для допустимых результатов. Std — выборочное, ddof=1.",
             "", "| Метод | Успехи | Best, ₽ | Mean, ₽ | Median, ₽ | Std, ₽ | Worst, ₽ | Среднее время, с |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    
    for row in rows:
        values = []
        for name in ["best_rub", "mean_rub", "median_rub", "sample_std_rub", "worst_rub", "mean_seconds"]:
            values.append("—" if row[name] is None else f"{row[name]:.2f}")
        lines.append(f"| {row['method']} | {row['successes']}/{row['runs']} | {' | '.join(values)} |")
    lines.extend(["", "| Метод | Среднее число проб ремонта | Среднее число поколений |", "|---|---:|---:|"])
    
    for row in rows:
        lines.append(f"| {row['method']} | {row['mean_repair_evaluations']:.1f} | {row['mean_generations']:.1f} |")
    bound = relaxed_lower_bound(catalog, config)
    lines.extend(["", f"Нижняя граница стоимости ослабленной задачи: **{bound / 100:.2f} ₽**."])
    if best is not None:
        cost = best["best_result"]["total_cost"]
        lines.append(f"Лучший результат: **{cost / 100:.2f} ₽**, {best['method']}, seed {best['seed']}.")
        lines.append(f"Превышение нижней границы: **{100 * (cost - bound) / bound:.2f}%**; это не доказательство оптимальности.")
    lines.extend(["", "![Сходимость](figures/convergence.png)", "", "![Итоги](figures/comparison.png)",
                  "", "![Допустимость популяции](figures/feasibility.png)", "", "![Меню](figures/best_menu.png)"])
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Версия оформления хранится отдельно от исходников поискового эксперимента.
    metadata["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["matplotlib"] = matplotlib.__version__
    save_json(output / "metadata.json", metadata)
