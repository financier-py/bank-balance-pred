"""
Тут генерируем профили клиентов.

Профиль - это статистика по клиенту. Пока тут будет: пол, возраст, сегмент, регион,
когда открыл счет, зарплатный клиент или нет.

В целом, значимые признаки для нашей задачи.
"""

import numpy as np
import pandas as pd

import config

SEGMENTS = {
    "mass": 0.85,
    "premium": 0.13, # сберпремьер
    "first": 0.02, # сберпервый
}

AGE_RANGE = {
    "mass": (20, 60),
    "premium": (30, 70),
    "first": (40, 80),
}

REGIONS = {
    "Москва": 0.30,
    "Санкт-Петербург": 0.15,
    "Казань": 0.10,
    "Новосибирск": 0.08,
    "Екатеринбург": 0.07,
    "Прочие": 0.30,
}


def draw_age(rng: np.random.Generator, segments: np.ndarray):
    ages = np.empty(len(segments), dtype=int)
    for seg, (low, high) in AGE_RANGE.items():
        mask = segments == seg
        ages[mask] = rng.integers(low, high, size=mask.sum())
    return ages


def draw_open_date(rng: np.random.Generator, n: int):
    start_year = pd.Timestamp(config.DATA_START).year
    came_early = rng.random(n) < 0.7

    years = np.where(
        came_early,
        rng.integers(2015, start_year, size=n),
        rng.integers(start_year, 2026, size=n),
    )
    months = rng.integers(1, 13, size=n)
    return [
        pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y, m in zip(years, months)
    ]


def draw_is_salary(rng: np.random.Generator, age: np.ndarray):
    prob = np.where((age >= 25) & (age <= 65), 0.6, 0.4)
    return rng.random(len(age)) < prob


def build_profiles():
    rng = np.random.default_rng(config.SEED)
    n = config.N_CLIENTS

    segment = rng.choice(list(SEGMENTS.keys()), size=n, p=list(SEGMENTS.values()))
    age = draw_age(rng, segment)

    return pd.DataFrame({
        'client_id': [i for i in range(n)],
        'gender': rng.choice(['M', 'F'], size=n),
        'age': age,
        'segment': segment,
        'region': rng.choice(list(REGIONS.keys()), size=n, p=list(REGIONS.values())),
        'open_date': draw_open_date(rng, n),
        'is_salary': draw_is_salary(rng, age)
    })


if __name__ == '__main__':
    profiles = build_profiles()
    profiles.to_parquet(config.PROFILES_PATH, index=False)
 
    print(profiles.head(10).to_string(index=False))