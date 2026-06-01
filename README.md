# Bank Deposit Forecasting Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-6D25D9?style=for-the-badge&logo=catboost&logoColor=white)

## Немного о задаче

Надо было построить модель прогнозирования денег клиентов на 12 месяцев вперед по трем продуктам: вклады, накопительные счета, карты.

Нужно было побить бейзлайн, где мы просто берем последнее известное значение и фиксируем его на весь горизонт вперед. 

Метрики - MAE и WAPE.

## Стек технологий

- `Numpy` и `Pandas` для создания клиентов и их счетов
- `PySpark` для фича инжиниринга и подготовки данных для модели
- `CatBoost` для обучения модели

## Структура проекта

Я писал все в разных модулях, чтобы не запутаться. 

* `config.py`
* `gen_profiles.py` - генерация профилей клиентов (возраст, пол, регион, доход и т.д.)
* `gen_movements.py` - генерация истории денег 
* `features.py` - тут делаем фичи _(оконные функции: лаги, скользящие средние, волатильность)._
* `supervised.py` - тут добавляем таргеты для модели
* `train.ipynb` - тут фильтруем данные, делаем сплит, обучаем модель и считаем метрики

## Архитектура

Я решил, что в качестве прототипа лучше будет построить 12 отдельных моделей на каждый горизонт планирования и уже тюнить каждую из полученных моделей. 

Также можно было использовать рекурсивность. То есть строить только одну модель.

## Результаты

Пока была обучена модель, предсказывающая только месяц вперед. 

* **Baseline MAE:** 14 200.48 руб.
* **CatBoost MAE:** 12 937.27 руб. 

* **Baseline WAPE:** 13.57%
* **CatBoost WAPE:** 12.36%

## Планы

Так как у нас может быть очень много клиентов, то я планирую дальше распределенно обучить