"""Генерация учебного каталога; контрольные меню не используются в поиске."""

import csv
import numpy as np
from model import ROOT, load_config, load_catalog, evaluate_menu, save_json

# Название, порция (г), базовые ккал, базовая цена (рубли).
# Это вручную заданные учебные ориентиры, а не таблица пищевой ценности.
DISH_TEMPLATES = [
    ("Овсяная каша с бананом", 300, 430, 65),
    ("Гречка с грибами", 300, 390, 80),
    ("Рис с овощами", 300, 410, 70),
    ("Паста с томатным соусом", 300, 510, 95),
    ("Плов с курицей", 300, 590, 130),
    ("Картофельная запеканка", 300, 470, 90),
    ("Чечевица с овощами", 300, 420, 85),
    ("Фасоль в томатном соусе", 300, 380, 80),
    ("Курица с гречкой", 350, 560, 135),
    ("Рыба с рисом", 350, 490, 150),
    ("Индейка с картофелем", 350, 520, 155),
    ("Омлет с сыром", 250, 460, 95),
    ("Сырники со сметаной", 250, 570, 110),
    ("Творожная запеканка", 250, 480, 100),
    ("Блины с творогом", 250, 540, 105),
    ("Борщ со сметаной", 400, 290, 80),
    ("Гороховый суп", 400, 370, 75),
    ("Куриный суп с лапшой", 400, 310, 85),
    ("Тыквенный крем-суп", 350, 320, 100),
    ("Лазанья овощная", 300, 530, 140),
    ("Кускус с овощами", 300, 440, 85),
    ("Булгур с курицей", 350, 550, 125),
    ("Перловка с грибами", 300, 400, 65),
    ("Пшённая каша с тыквой", 300, 450, 70),
    ("Вареники с картофелем", 300, 580, 105),
    ("Пельмени со сметаной", 250, 620, 135),
    ("Котлета с пюре", 350, 610, 140),
    ("Тушёная капуста с курицей", 350, 390, 110),
    ("Салат с фасолью и сухариками", 300, 450, 95),
    ("Салат с курицей и картофелем", 300, 500, 120),
]


def generate_catalog(path, seed):
    """Записывает 30 блюд: к базовым ценам и калориям добавляет небольшой шум."""
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name", "portion_size", "portion_unit", "price_kopecks", "calories"])
        for index, template in enumerate(DISH_TEMPLATES):
            name, portion, calories, price = template
            calories += 5 * int(rng.integers(-4, 5))
            price += int(rng.integers(-8, 9))
            writer.writerow([f"D{index + 1:02d}", name, portion, "г", price * 100, calories])


def create_examples(catalog, config):
    """Строит два независимых от ГА контрольных меню и два ошибочных примера."""
    # Отдельный seed: этими случайными числами алгоритм поиска не пользуется.
    rng = np.random.default_rng(config["data_seed"] + 1)
    examples = []
    attempts = 0
    while len(examples) < 2 and attempts < 100000:
        attempts += 1
        menu = np.zeros((config["days"], len(catalog["dishes"])), dtype=np.int8)
        for day in range(config["days"]):
            calories = 0
            for index in rng.permutation(len(catalog["dishes"])):
                menu[day, index] = 1
                calories += int(catalog["calories"][index])
                if calories >= config["min_calories"]:
                    break
        result = evaluate_menu(menu, catalog, config)
        if result["feasible"]:
            examples.append({"name": f"feasible_{len(examples) + 1}", "menu": menu, "evaluation": result})
    if len(examples) != 2:
        raise RuntimeError("Не удалось подтвердить совместимость. Ограничения не изменены.")
    empty = np.zeros_like(examples[0]["menu"])
    repeated = np.tile(examples[0]["menu"][0], (config["days"], 1))
    for name, menu in [("invalid_empty", empty), ("invalid_repeated_day", repeated)]:
        examples.append({"name": name, "menu": menu, "evaluation": evaluate_menu(menu, catalog, config)})
    return {"seed": config["data_seed"] + 1, "construction_attempts": attempts, "examples": examples}


if __name__ == "__main__":
    config = load_config()
    generate_catalog(ROOT / "data" / "dishes.csv", config["data_seed"])
    examples = create_examples(load_catalog(), config)
    save_json(ROOT / "data" / "examples.json", examples)
    for example in examples["examples"]:
        result = example["evaluation"]
        print(example["name"], "допустимо:", result["feasible"], "цена:", result["total_cost"] / 100)
