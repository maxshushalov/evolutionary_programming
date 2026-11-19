import hashlib
import json
import math
import random
from pathlib import Path


TOLERANCE = 1e-9


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_instance(instance):
    clients = instance["clients"]
    if instance["capacity"] <= 0 or instance["horizon"] <= 0 or instance["vehicles"] < 1:
        raise ValueError("Некорректные параметры машин")
    for client in clients[1:]:
        if client["demand"] <= 0 or client["service"] < 0 or client["a"] > client["b"]:
            raise ValueError("Некорректные параметры клиента")
    instance["distances"] = [
        [math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for b in clients]
        for a in clients
    ]
    return instance


def load_instance(path):
    return prepare_instance(load_json(path))


def encode(routes):
    return {"permutation": [client for route in routes for client in route],
            "route_lengths": [len(route) for route in routes]}


def decode(solution):
    routes = []
    offset = 0
    for length in solution["route_lengths"]:
        routes.append(solution["permutation"][offset:offset + length])
        offset += length
    return routes


def validate_solution(instance, solution):
    n = len(instance["clients"]) - 1
    lengths = solution["route_lengths"]
    if sorted(solution["permutation"]) != list(range(1, n + 1)):
        raise ValueError("Каждый клиент должен встречаться ровно один раз")
    if len(lengths) != instance["vehicles"] or sum(lengths) != n:
        raise ValueError("Некорректное разбиение маршрутов")
    if any(type(length) is not int or length < 0 for length in lengths):
        raise ValueError("Длины маршрутов должны быть неотрицательными целыми")


def positive_violation(value):
    return value if value > TOLERANCE else 0.0


def evaluate(instance, solution, details=False):
    clients = instance["clients"]
    distances = instance["distances"]
    capacity = instance["capacity"]
    horizon = instance["horizon"]
    total_distance = 0.0
    capacity_excess = 0.0
    lateness = 0.0
    return_excess = 0.0
    route_details = []

    for route in decode(solution):
        previous = 0
        time = 0.0
        load = 0
        length = 0.0
        visits = []
        for client_id in route:
            client = clients[client_id]
            distance = distances[previous][client_id]
            arrival = time + distance
            start = max(arrival, client["a"])
            late = positive_violation(start - client["b"])
            time = start + client["service"]
            load += client["demand"]
            length += distance
            lateness += late
            previous = client_id
            if details:
                visits.append({"client": client_id, "arrival": arrival, "start": start,
                               "departure": time, "lateness": late})
        length += distances[previous][0]
        time += distances[previous][0]
        excess = positive_violation(load - capacity)
        overtime = positive_violation(time - horizon)
        capacity_excess += excess
        return_excess += overtime
        total_distance += length
        if details:
            route_details.append({"clients": route, "load": load, "distance": length,
                                  "return_time": time, "capacity_excess": excess,
                                  "return_excess": overtime, "visits": visits})

    violation = capacity_excess / capacity + (lateness + return_excess) / horizon
    result = {"distance": total_distance, "violation": violation, "feasible": violation == 0,
              "capacity_excess": capacity_excess, "lateness": lateness,
              "return_excess": return_excess}
    if details:
        result["routes"] = route_details
    return result


def fitness_key(score):
    return (0, 0.0, score["distance"]) if score["feasible"] else (1, score["violation"], score["distance"])


def generate_instance(n, vehicles, seed, version="v1"):
    rng = random.Random(seed)
    clients = [{"x": 50.0, "y": 50.0, "demand": 0, "service": 0, "a": 0, "b": 0}]
    for _ in range(n):
        clients.append({"x": rng.uniform(0, 100), "y": rng.uniform(0, 100),
                        "demand": rng.randint(1, 9), "service": rng.randint(2, 5), "a": 0, "b": 0})
    order = sorted(range(1, n + 1), key=lambda i: math.atan2(clients[i]["y"] - 50, clients[i]["x"] - 50))
    routes = []
    offset = 0
    for vehicle in range(vehicles):
        size = n // vehicles + (vehicle < n % vehicles)
        routes.append(order[offset:offset + size])
        offset += size
    capacity = math.ceil(1.15 * max(sum(clients[i]["demand"] for i in route) for route in routes))
    returns = []
    for route in routes:
        previous = 0
        time = 0.0
        for i in route:
            time += math.hypot(clients[i]["x"] - clients[previous]["x"], clients[i]["y"] - clients[previous]["y"])
            clients[i]["a"] = max(0, time - rng.uniform(15, 45))
            clients[i]["b"] = time + rng.uniform(15, 45)
            time += clients[i]["service"]
            previous = i
        returns.append(time + math.hypot(clients[previous]["x"] - 50, clients[previous]["y"] - 50))
    horizon = max(returns) + 30
    clients[0]["b"] = horizon
    instance = {"name": f"instance_{n}", "version": version, "data_seed": seed,
                "vehicles": vehicles, "capacity": capacity, "horizon": horizon, "clients": clients}
    return instance, encode(routes)


def tiny_instance():
    clients = [{"x": 0, "y": 0, "demand": 0, "service": 0, "a": 0, "b": 40}]
    for x, y in [(1, 0), (2, 0), (3, 1), (0, 1), (0, 2), (1, 3)]:
        clients.append({"x": x, "y": y, "demand": 1, "service": 1, "a": 0, "b": 25})
    return prepare_instance({"name": "tiny", "vehicles": 2, "capacity": 3, "horizon": 40, "clients": clients})
