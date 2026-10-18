"""Запуски ГА и Random Search, CSV, статистика и графики для отчёта."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .genetic_algorithm import GAConfig, RunResult, run_genetic_algorithm
from .random_search import run_random_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "ga_config_a": "GA Config A (σ = 1.5)",
    "ga_config_b": "GA Config B (σ = 3.0)",
    "random_search": "Random Search",
}


def load_config(path: Path) -> tuple[GAConfig, int]:
    """Загрузить параметры ГА и количество независимых запусков из JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    number_of_runs = data.pop("number_of_runs", 20)
    if type(number_of_runs) is not int or number_of_runs < 2:
        raise ValueError("number_of_runs должно быть целым числом >= 2 для sample std")
    return GAConfig(**data), number_of_runs


def experiment_configs(base: GAConfig) -> dict[str, GAConfig]:
    """A и B отличаются только sigma: соответственно 5% и 10% диапазона."""
    width = base.upper_bound - base.lower_bound
    return {
        "ga_config_a": replace(base, mutation_sigma=0.05 * width),
        "ga_config_b": replace(base, mutation_sigma=0.10 * width),
        "random_search": base,
    }


def result_row(result: RunResult, include_generations: bool = True) -> dict:
    """Развернуть результат в одну строку CSV с координатами x1 ... xd."""
    row = {
        "seed": result.seed, "best_fitness": result.best_fitness,
        "evaluations": result.evaluations,
    }
    if include_generations:
        row["generations"] = result.generations
    row.update({f"x{i + 1}": float(x) for i, x in enumerate(result.best_x)})
    return row


def summarize_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Вычислить статистику итогов; std — выборочное отклонение (ddof=1)."""
    rows = []
    for method, frame in results.items():
        values = frame["best_fitness"]
        rows.append({
            "method": method, "best": values.min(), "mean": values.mean(),
            "median": values.median(), "std": values.std(ddof=1),
            "worst": values.max(), "runs": len(values), "std_ddof": 1,
        })
    return pd.DataFrame(rows)


def plot_convergence(
    histories: dict[str, pd.DataFrame], figures_dir: Path, labels: dict[str, str],
) -> None:
    """Показать среднюю траекторию и min/max между независимыми запусками."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    max_generation = max(int(frame["generation"].max()) for frame in histories.values())
    zoom_start = min(20, max_generation // 2)
    for (method, frame), color in zip(histories.items(), ("#2563eb", "#e87924")):
        grouped = frame.groupby("generation")["global_best_fitness"].agg(["min", "mean", "max"])
        for index, ax in enumerate(axes):
            shown = grouped if index == 0 else grouped.loc[zoom_start:]
            ax.plot(shown.index, shown["mean"], label=labels[method], color=color)
            ax.fill_between(shown.index, shown["min"], shown["max"], color=color, alpha=0.15)
    axes[0].set_title("Весь запуск: среднее и диапазон min–max")
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_title(f"После поколения {zoom_start}: линейная шкала")
    axes[1].set_xlim(zoom_start, max(1, max_generation))
    for ax in axes:
        ax.set_xlabel("Поколение (0 — начальная популяция)")
        ax.set_ylabel("Лучшее найденное значение HGBat")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Сходимость генетического алгоритма (слева — symlog)")
    fig.savefig(figures_dir / "convergence.png", dpi=180)
    plt.close(fig)


def plot_final_comparison(
    results: dict[str, pd.DataFrame], figures_dir: Path, labels: dict[str, str],
) -> None:
    """Сравнить распределения результатов; точки показывают все запуски."""
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    values = [frame["best_fitness"].to_numpy() for frame in results.values()]
    boxes = ax.boxplot(values, tick_labels=[labels[key] for key in results],
                       patch_artist=True, showfliers=False)
    for position, (box, samples, color) in enumerate(
        zip(boxes["boxes"], values, ("#2563eb", "#e87924", "#16a34a")), start=1,
    ):
        box.set_facecolor(color)
        box.set_alpha(0.25)
        ax.scatter(position + np.linspace(-0.08, 0.08, len(samples)), samples,
                   color=color, s=20, alpha=0.7,
                   label="Отдельные запуски" if position == 1 else None)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("Итоговое значение HGBat (меньше — лучше), symlog")
    ax.set_xlabel("Метод")
    ax.set_title("Результаты при одинаковом бюджете вычислений")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(figures_dir / "final_comparison.png", dpi=180)
    plt.close(fig)


def run_method(method: str, config: GAConfig, seed: int) -> RunResult:
    """Запустить выбранный метод без изменения бюджета."""
    if method == "random_search":
        return run_random_search(config, seed)
    if method in ("ga_config_a", "ga_config_b"):
        return run_genetic_algorithm(config, seed)
    raise ValueError(f"Неизвестный метод: {method}")


def run_experiments(base: GAConfig, number_of_runs: int, output_dir: Path) -> pd.DataFrame:
    """Выполнить три серии с seed 0..number_of_runs-1 и сохранить артефакты."""
    if type(number_of_runs) is not int or number_of_runs < 2:
        raise ValueError("Нужно минимум два запуска для выборочного std")
    raw_dir = output_dir / "results" / "raw"
    summary_dir = output_dir / "results" / "summary"
    figures_dir = output_dir / "figures"
    for directory in (raw_dir, summary_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    configs = experiment_configs(base)
    results, histories = {}, {}
    for method, config in configs.items():
        rows, history_rows = [], []
        for seed in range(number_of_runs):
            result = run_method(method, config, seed)
            rows.append(result_row(result, include_generations=method != "random_search"))
            history_rows.extend({"seed": seed, **record} for record in result.history)
            print(f"{method}: seed={seed:2d}, f={result.best_fitness:.8g}, "
                  f"evaluations={result.evaluations}", flush=True)
        results[method] = pd.DataFrame(rows)
        results[method].to_csv(raw_dir / f"{method}.csv", index=False)
        if history_rows:
            histories[method] = pd.DataFrame(history_rows)
            histories[method].to_csv(raw_dir / f"convergence_{method}.csv", index=False)
    statistics = summarize_results(results)
    statistics.to_csv(summary_dir / "statistics.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
        "seeds": list(range(number_of_runs)), "std_ddof": 1,
        "total_objective_evaluations": 3 * number_of_runs * base.evaluation_budget,
        "configurations": {name: asdict(config) for name, config in configs.items()},
    }
    (summary_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    labels = {**LABELS, **{
        method: f"GA Config {letter} (σ = {configs[method].mutation_sigma:g})"
        for method, letter in (("ga_config_a", "A"), ("ga_config_b", "B"))
    }}
    plot_convergence(histories, figures_dir, labels)
    plot_final_comparison(results, figures_dir, labels)
    return statistics


def main() -> None:
    """Без аргументов — полная серия; --single — один воспроизводимый запуск."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--single", choices=tuple(LABELS))
    parser.add_argument("--seed", type=int, default=0, help="seed для --single")
    args = parser.parse_args()
    base, number_of_runs = load_config(args.config)
    if args.single:
        config = experiment_configs(base)[args.single]
        result = run_method(args.single, config, args.seed)
        payload = {"method": args.single, "config": asdict(config),
                   **result_row(result), "history": result.history}
        directory = args.output / "results" / "single"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.single}_seed_{args.seed}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result_row(result), ensure_ascii=False, indent=2))
        print(f"Сохранено: {path}")
    else:
        print(run_experiments(base, number_of_runs, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
