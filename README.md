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
│       ├── global_horizon.py # + 1 таргет и фича какой месяц
│       └── tft_dataset.py # добавляем столбец time_idx
├── notebooks/ # тут уже сами модели
│   ├── 01_independent_models.ipynb # 12 независимых моделей
│   ├── 02_global_model.ipynb # Одна единая модель
│   └── 03_train_tft.ipynb # Temporal Fusion Transformer
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

Или, например, датасет для трансформера:
```bash
python -m src.targets.tft_dataset.py
```

## Архитектуры моделей

В проекте будет протестировано 3 разных идеи:

1. Обучение 12 независимых моделей на CatBoost. Каждая модель предсказывает именно свой горизонт.
   - Плюс: выше точность, нежели предсказывать рекурсивно
   - Минус: сложно поддерживать целых 12 моделей
2. Обучение одной глобальной модели CatBoost, на вход которой мы подаем столбец на какой месяц вперед мы предсказываем.
3. Temporal Fusion Transformer

## Результаты

### 1. Прототип CatBoost, предсказывающая только на 1 месяц вперед:

| Модель | MAE (руб.) | WAPE (%) |
| :--- | :---: | :---: |
| **Baseline** | 14 200.48 | 13.57% |
| **CatBoost** | 12 937.27 | 12.36% |

### 2. Global Horizon Model (прогнозирование дельты на 1-12м):

| Горизонт (мес.) | Baseline WAPE | Моя умная модель WAPE |
| :---: | :---: | :---: |
| **1** | 13.88% | **12.39%** | 
| **2** | 15.55% | **13.22%** | 
| **3** | 17.13% | **13.91%** | 
| **4** | 16.43% | **13.16%** | 
| **5** | 15.12% | **13.06%** | 
| **6** | 15.76% | **13.53%** | 
| **7** | 15.72% | **13.59%** | 
| **8** | 27.61% | **14.35%** | 
| **9** | 15.10% | **12.91%** |
| **10** | 14.67% | **12.81%** |
| **11** | 15.11% | **12.92%** | 
| **12** | 14.40% | **12.44%** |

### 3. Итоговая табличка:

| Горизонт (мес.) | Baseline | Global CatBoost | TFT (Медиана) |
| :---: | :---: | :---: | :---: |
| **1** | 13.88% | 12.39% | **10.143%** |
| **2** | 15.55% | 13.22% | **11.183%** |
| **3** | 17.13% | 13.91% | **11.734%** |
| **4** | 16.43% | 13.16% | **11.016%** |
| **5** | 15.12% | 13.06% | **11.235%** |
| **6** | 15.76% | 13.53% | **11.166%** |
| **7** | 15.72% | 13.59% | **11.688%** |
| **8** | 27.61% | 14.35% | **11.460%** |
| **9** | 15.10% | 12.91% | **11.247%** |
| **10** | 14.67% | 12.81% | **10.931%** |
| **11** | 15.11% | 12.92% | **11.189%** |
| **12** | 14.40% | 12.44% | **10.677%** |

## Планы

- Улучшить генерацию профилей клиентов и их балансов, добавив больше зависимостей
- Улучшить фича инжиниринг, чтобы доставать признаки по нулям и относительные фичи
- Написать скрипт для распределенного обучения на CatBoost
- ~~Реализовать Temporal Fusion Transformer~~