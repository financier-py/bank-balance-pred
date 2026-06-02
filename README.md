# Bank Deposit Forecasting Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-6D25D9?style=for-the-badge&logo=catboost&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## Немного о задаче

Надо было построить модель прогнозирования денег клиентов на 12 месяцев вперед по трем продуктам: вклады, накопительные счета, карты.

Нужно было побить бейзлайн, где мы просто берем последнее известное значение и фиксируем его на весь горизонт вперед. 

Метрики - MAE и WAPE.

## Стек технологий

- **Генерация данных:** `numpy` и `pandas`
- **Фича инжиниринг:** `pyspark`
- **Модели:** `catboost` и `pytorch`

## Структура проекта

Было решено разделить логику генерации, создания признаков и подготовки обучающих выборок для разных моделей.

```text
bank-ts-pred/
├── data/               
├── src/
│   ├── __init__.py
│   ├── config.py # тут все пути и гиперпараметры
│   ├── data_gen/
│   │   ├── __init__.py
│   │   ├── profiles.py # Генерация профилей
│   │   └── movements.py # Генерация балансов
│   ├── features/ 
│   │   ├── __init__.py
│   │   ├── conformed.py # Создание непрерывной сетки
│   │   └── engineering.py # Расчет всяких фич
│   └── targets/ # приклеиваем таргеты
│       ├── __init__.py
│       ├── multi_target.py # + 12 столбцов таргетов
│       └── global_horizon.py # + 1 таргет и фича какой месяц
├── notebooks/ # тут уже сами модели
│   ├── 01_independent_models.ipynb # 12 независимых моделей
│   ├── 02_global_model.ipynb # Одна единая модель
│   └── 03_train_tft.ipynb # TDDO: трансформер
├── pyproject.toml
└── README.md
```

## Как запустить

Если вы используете `uv`:

```bash
uv sync
```

Если все же через pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
Далее, какой порядок запуска скриптов:

```bash
# Шаг 1. Генерируем синтетическую историю клиентов
python -m src.data_gen.profiles
python -m src.data_gen.movements
```

```bash
# Шаг 2. Считаем фичи
python -m src.features.engineering
```
```bash
# Шаг 3. Готовим train dataset под выбранную архитектуру
python -m src.targets.global_horizon
```

## Архитектуры моделей

В проекте будет протестировано 3 разных идеи:

1. Обучение 12 независимых моделей на CatBoost. Каждая модель предсказывает именно свой горизонт.
   - Плюс: выше точность, нежели предсказывать рекурсивно
   - Минус: сложно поддерживать целых 12 моделей
2. Обучение одной глобальной модели CatBoost, на вход которой мы подаем столбец на какой месяц вперед мы предсказываем.
3. Temporal Fusion Transformer



## Результаты

Пока была обучена модель, предсказывающая только месяц вперед и вот ее результаты. 

| Модель | MAE (руб.) | WAPE (%) |
| :--- | :---: | :---: |
| **Baseline** | 14 200.48 | 13.57% |
| **CatBoost** | 12 937.27 | 12.36% |

## Планы

Так как у нас может быть очень много клиентов, то я планирую дальше написать скрипт для распределенного обучения

Ну и две оставшихся модели надо сделать