import argparse
import csv
import itertools
import platform
from importlib.metadata import version
from pathlib import Path

from algorithms import run_algorithm
from problem import (encode, evaluate, file_hash, load_instance, load_json, prepare_instance,
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


def validate_results(folder):
    paths = sorted(Path(folder).glob("*.json"))
    checked = 0
    for path in paths:
        if path.name == "metadata.json":
            continue
        result = load_json(path)
        instance_path = ROOT / "data" / f"{result['instance']}.json"
        if result["data_hash"] != file_hash(instance_path):
            raise ValueError(f"Изменены входные данные: {path}")
        instance = load_instance(instance_path)
        validate_solution(instance, result["best"]["solution"])
        score = evaluate(instance, result["best"]["solution"], details=True)
        for key in ["distance", "violation", "capacity_excess", "lateness", "return_excess"]:
            if abs(score[key] - result["best"]["score"][key]) > 1e-8:
                raise ValueError(f"Несогласованная оценка: {path}, {key}")
        if score != result["best"]["validation"]:
            raise ValueError(f"Несогласованные маршруты: {path}")
        if result["evaluations"] != result["params"]["evaluation_budget"]:
            raise ValueError(f"Неверный бюджет: {path}")
        previous = None
        for point in result["history"]:
            if point["distance"] is not None:
                if previous is not None and point["distance"] > previous + 1e-9:
                    raise ValueError("Ухудшение лучшей допустимой длины")
                previous = point["distance"]
        checked += 1
    save_json(Path(folder) / "validation" / "checks.json", {"runs_checked": checked, "passed": True})
    print(f"Проверены {checked} сохранённых запусков")


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
