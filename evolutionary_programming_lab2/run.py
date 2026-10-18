"""Командная строка: один запуск или полная серия с сохранением результатов."""

import argparse
import csv
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from algorithm import METHODS, run_search
from model import ROOT, load_config, load_catalog, save_json


def write_csv(path, rows):

    """Сохраняет список однотипных словарей в CSV."""

    if not rows:
        return
    
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():

    """Запоминает версии исходников, конфигурации и каталога для воспроизводимости."""

    paths = list(ROOT.glob("*.py"))
    paths.extend([ROOT / "config.json", ROOT / "data" / "dishes.csv"])
    result = {}

    for path in sorted(paths):
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        
    return result


def run_experiments(config, catalog, output, methods, seeds):

    """Выполняет серию; сразу сохраняет каждый завершённый запуск."""

    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    runs = []
    rows = []

    for method in methods:
        for seed in seeds:
            result = run_search(method, seed, catalog, config)
            save_json(raw / f"{method}_{seed:02d}.json", result)
            runs.append(result)
            best_cost = None
            if result["best_result"] is not None:
                best_cost = result["best_result"]["total_cost"]
            rows.append({"method": method, "seed": seed, "feasible": best_cost is not None,
                         "cost_kopecks": best_cost, "evaluations": result["evaluations"],
                         "candidate_evaluations": result["candidate_evaluations"],
                         "repair_evaluations": result["repair_evaluations"],
                         "repair_accepted": result["repair_accepted"],
                         "first_feasible_evaluation": result["first_feasible_evaluation"],
                         "generations": result["generations"], "seconds": result["seconds"],
                         "repair_seconds": result["repair_seconds"]})
            print(method, seed, "цена, ₽:", None if best_cost is None else best_cost / 100,
                  "оценок:", result["evaluations"], "ремонт:", result["repair_evaluations"], flush=True)
    write_csv(output / "runs.csv", rows)
    metadata = {"config": config, "methods": methods, "seeds": seeds,
                "python": platform.python_version(), "numpy": np.__version__,
                "platform": platform.platform(), "finished_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_hashes(), "total_evaluations": sum(row["evaluations"] for row in rows),
                "budget_definition": "Каждый вызов evaluate_menu в поиске, включая все пробы ремонта"}
    save_json(output / "metadata.json", metadata)
    return runs


def main():

    """Разбирает простые параметры запуска; по умолчанию выполняет полную серию."""

    parser = argparse.ArgumentParser(description="Лабораторная №2: недельное меню")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--single", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", action="store_true", help="Пересобрать графики из сохранённых запусков")
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = load_catalog()

    if args.output is not None:
        output = args.output.resolve()
        # Результаты сохраняются только внутри текущего проекта.
        if not output.is_relative_to(ROOT):
            parser.error("Каталог результатов должен находиться внутри папки проекта")
    elif args.single:
        output = ROOT / "results" / "single" / f"{args.single}_{args.seed}"
    else:
        output = ROOT / "results" / "experiment"
    if not args.summarize:
        methods = list(METHODS) if args.single is None else [args.single]
        seeds = config["seeds"] if args.single is None else [args.seed]
        run_experiments(config, catalog, output, methods, seeds)
    from analysis import summarize
    summarize(output, catalog)


if __name__ == "__main__":
    main()
