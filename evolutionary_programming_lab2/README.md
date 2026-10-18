# Лабораторная работа №2

Вариант 20. Формирование меню на неделю с помощью генетического алгоритма.

Программа минимизирует стоимость меню из 30 доступных блюд с ограничениями
по калорийности, бюджету и разнообразию. Данные синтетические.

## Установка

Команды выполняются из папки `evolutionary_programming_lab2`.
Проверено на Python 3.14.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

В Windows для активации окружения:

```bat
.venv\Scripts\activate
```

## Запуск

Один запуск алгоритма с ремонтом и битовой мутацией:

```bash
python run.py --single combined_bit --seed 0
```

Результат сохраняется в `results/single/combined_bit_0`.

Полная серия: пять методов по 20 запусков:

```bash
python run.py
```

Результаты сохраняются в `results/experiment`.
Повторный запуск перезаписывает результаты серии.

Перестроить таблицы и графики по сохранённым результатам:

```bash
python run.py --summarize
```

Запустить тесты:

```bash
python -m unittest discover -s tests -v
```

## Данные и результаты

- `config.json` — параметры задачи и алгоритма.
- `data/dishes.csv` — каталог блюд.
- `report.md` — отчёт.
- `results/experiment/statistics.csv` — итоговая статистика.
- `results/experiment/best_menu.md` — лучшее найденное меню.
- `results/experiment/figures/` — графики PNG.
