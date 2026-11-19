import argparse
import itertools
import math
import platform
from importlib.metadata import version
from pathlib import Path

from algorithms import run_algorithm
from problem import (evaluate, file_hash, load_instance, load_json, prepare_instance,
                     generate_instance, save_json, tiny_instance, validate_solution)


ROOT = Path(__file__).resolve().parent


def metadata(config):
    return {"config": config, "python": platform.python_version(), "platform": platform.platform(),
            "libraries": {name: version(name) for name in ["matplotlib", "numpy"]},
            "sources": {path.name: file_hash(path) for path in sorted(ROOT.glob("*.py"))}}


def generate(config):
    for spec in config["instances"]:
        instance, witness = generate_instance(spec["n"], spec["vehicles"], spec["seed"], config["data_version"])
        save_json(ROOT / "data" / f"instance_{spec['n']}.json", instance)
        prepared = prepare_instance(instance)
        validate_solution(prepared, witness)
        score = evaluate(prepared, witness, details=True)
        if not score["feasible"]:
            raise ValueError("Ошибка генератора: контрольный маршрут недопустим")
        save_json(ROOT / "data" / "validation" / f"witness_{spec['n']}.json", {"solution": witness, "score": score})
    tiny = tiny_instance()
    tiny.pop("distances")
    save_json(ROOT / "data" / "tiny.json", tiny)


def save_run(instance_path, method, seed, params, output):
    instance = load_instance(instance_path)
    result = run_algorithm(instance, method, seed, params)
    result["instance"] = instance["name"]
    result["data_hash"] = file_hash(instance_path)
    result["best"]["validation"] = evaluate(instance, result["best"]["solution"], details=True)
    result["source_hashes"] = {name: file_hash(ROOT / name) for name in ["algorithms.py", "problem.py"]}
    save_json(output, result)
    score = result["best"]["score"]
    print(f"{instance['name']} {method} seed={seed} допустимость={score['feasible']} "
          f"L={score['distance']:.3f} v={score['violation']:.4f} {result['seconds']:.2f} с", flush=True)
    return result


def experiments(config, pilot=False, output=None):
    folder = Path(output) if output else ROOT / "results" / ("pilot_v1" if pilot else "main")
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Папка {folder} не пуста. Выберите новую папку --output")
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "metadata.json", metadata(config))
    for source in ROOT.glob("*.py"):
        snapshot = folder / "sources" / source.name
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    seed_config = config["pilot"] if pilot else config["seeds"]
    for spec in config["instances"]:
        path = ROOT / "data" / f"instance_{spec['n']}.json"
        for method in ["ga", "memetic"]:
            params = dict(config["algorithm"])
            if pilot:
                params["evaluation_budget"] = seed_config["evaluation_budget"]
            for seed in range(seed_config["start"], seed_config["start"] + seed_config["count"]):
                save_run(path, method, seed, params, folder / f"{spec['n']}_{method}_{seed}.json")
        if spec["n"] == 40 and not pilot:
            params = dict(config["algorithm"], local_search_steps=1)
            for seed in range(seed_config["start"], seed_config["start"] + seed_config["count"]):
                save_run(path, "memetic", seed, params, folder / f"40_memetic_steps1_{seed}.json")


def exact_control():
    instance = load_instance(ROOT / "data" / "tiny.json")
    best = None
    checked = 0
    for order in itertools.permutations(range(1, 7)):
        for cut in range(7):
            solution = {"permutation": list(order), "route_lengths": [cut, 6 - cut]}
            score = evaluate(instance, solution)
            checked += 1
            if score["feasible"] and (best is None or score["distance"] < best["score"]["distance"]):
                best = {"solution": solution, "score": score}
    save_json(ROOT / "results" / "exact_control.json", {"checked": checked, "best": best})
    print(f"Полный перебор: {checked} решений, оптимум {best['score']['distance']:.9f}")


def validate_history(result, score, path):
    history = result["history"]
    budget = result["params"]["evaluation_budget"]
    interval = result["params"]["history_interval"]
    if result["evaluations"] != budget:
        raise ValueError(f"Неверный бюджет: {path}")
    if [point["evaluations"] for point in history] != [*range(interval, budget, interval), budget]:
        raise ValueError(f"Неверные контрольные точки: {path}")

    first = result["first_feasible"]
    if (first is not None) != score["feasible"] or (
        first is not None and (type(first) is not int or not 1 <= first <= budget)
    ):
        raise ValueError(f"Неверный номер первой допустимой оценки: {path}")

    previous_distance = previous_violation = math.inf
    previous_local = 0
    for point in history:
        feasible = first is not None and first <= point["evaluations"]
        distance, violation, local = point["distance"], point["violation"], point["local_evaluations"]
        if point["feasible"] is not feasible or (violation == 0) != feasible:
            raise ValueError(f"Несогласованная допустимость в истории: {path}")
        if not math.isfinite(violation) or not 0 <= violation <= previous_violation:
            raise ValueError(f"Некорректное нарушение в истории: {path}")
        if feasible:
            if distance is None or not math.isfinite(distance) or not 0 <= distance <= previous_distance:
                raise ValueError(f"Некорректная длина в истории: {path}")
            previous_distance = distance
        elif distance is not None:
            raise ValueError(f"Длина указана до первой допустимой оценки: {path}")
        if not previous_local <= local <= point["evaluations"]:
            raise ValueError(f"Неверный счётчик локальных оценок: {path}")
        previous_violation, previous_local = violation, local

    last = history[-1]
    distance = score["distance"] if score["feasible"] else None
    if last["distance"] != distance or last["violation"] != score["violation"]:
        raise ValueError(f"Последняя точка истории не совпадает с итогом: {path}")
    if (previous_local != result["local_evaluations"]
        or not 0 <= result["accepted_local"] <= previous_local
        or (result["method"] == "ga" and previous_local != 0)
    ):
        raise ValueError(f"Несогласованный итог локального поиска: {path}")


def validate_results(folder):
    paths = sorted(path for path in Path(folder).glob("*.json") if path.name != "metadata.json")
    if not paths:
        raise ValueError(f"Нет сохранённых запусков: {folder}")
    for path in paths:
        result = load_json(path)
        instance_path = ROOT / "data" / f"{result['instance']}.json"
        if result["data_hash"] != file_hash(instance_path):
            raise ValueError(f"Изменены входные данные: {path}")
        instance = load_instance(instance_path)
        validate_solution(instance, result["best"]["solution"])
        score = evaluate(instance, result["best"]["solution"], details=True)
        summary = {key: value for key, value in score.items() if key != "routes"}
        saved_score = result["best"]["score"]
        if summary != saved_score or score["feasible"] is not saved_score["feasible"]:
            raise ValueError(f"Несогласованная оценка: {path}")
        if score != result["best"]["validation"]:
            raise ValueError(f"Несогласованные маршруты: {path}")
        validate_history(result, score, path)
    save_json(Path(folder) / "validation" / "checks.json", {"runs_checked": len(paths), "passed": True})
    print(f"Проверены {len(paths)} сохранённых запусков")


def main():
    parser = argparse.ArgumentParser(description="Лабораторная №4: доставка с временными окнами")
    parser.add_argument("command", choices=["generate", "single", "pilot", "experiments", "validate", "exact"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--instance", default=str(ROOT / "data" / "instance_40.json"))
    parser.add_argument("--method", choices=["ga", "memetic"], default="memetic")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.command == "generate":
        generate(config)
    elif args.command == "single":
        output = args.output or ROOT / "results" / f"single_{args.method}_{args.seed}.json"
        save_run(args.instance, args.method, args.seed, config["algorithm"], output)
    elif args.command in ["pilot", "experiments"]:
        experiments(config, args.command == "pilot", args.output)
    elif args.command == "exact":
        exact_control()
    elif args.command == "validate":
        validate_results(args.output or ROOT / "results" / "main")


if __name__ == "__main__":
    main()
