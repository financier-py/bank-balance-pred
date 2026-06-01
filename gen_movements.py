"""
Тут генерируем сколько денег у клиента на каждом продукте в каждый месяц.

Суммы будут зависеть от профиля и времни

  - зависит от сегмента
  - часть клиентов не пользуется продуктом: стоит нуль
  - у активов бывает иногда нолик
  - баланс медленно растет или падает
  - есть сезонность: в декабре, наверно, премии
  - когда-то бывают выбросы
"""

import numpy as np
import pandas as pd

import config

BASE_LEVEL = {
    "deposit": {"mass": 80_000, "premium": 700_000, "first": 4_000_000},
    "savings": {"mass": 40_000, "premium": 200_000, "first": 1_000_000},
    "card": {"mass": 25_000, "premium": 60_000, "first": 150_000},
}

# Вероятность, что клиент вообще не пользуется продуктом
STRUCTURAL_ZERO = {
    "deposit": {"mass": 0.45, "premium": 0.20, "first": 0.10},
    "savings": {"mass": 0.50, "premium": 0.30, "first": 0.15},
    "card": {"mass": 0.03, "premium": 0.01, "first": 0.01},
}


def month_grid(open_date: pd.Timestamp | str):
    """Создает сетку месяцев для отдельного клиента"""
    start = max(pd.Timestamp(open_date), pd.Timestamp(config.DATA_START))
    return pd.date_range(start, config.DATA_END, freq="ME")


def seasonal_factor(months, salary):
    """Добавляет сезонность"""
    factor = np.ones(len(months))
    m = months.month
    factor[m == 12] *= 1.4 if salary else 1.2
    factor[np.isin(m, [6, 7, 8])] *= 0.95
    return factor


def make_series(rng: np.random.Generator, months, level: int, salary: bool):
    """Создает ряд баланса для одного клиента И одного продукта на ВСЕ его месяцы"""
    n = len(months)
    client_level = level * rng.lognormal(mean=0.0, sigma=0.5)

    slope = rng.normal(0, 0.15)
    trend = np.linspace(1.0, 1.0 + slope, n).clip(0.3, None)

    season = seasonal_factor(months, salary)
    noise = rng.lognormal(
        0, 0.08 if salary else 0.15, n
    )  # зарплатные как бы стабильнее

    series = client_level * trend * season * noise

    # иногда клиент депает в казик
    if rng.random() < 0.15:
        gap = rng.integers(1, 5)
        pos = rng.integers(0, max(1, n - gap))
        series[pos : pos + gap] = 0

    # пришло наследство
    if rng.random() < 0.05:
        series[rng.integers(0, n)] *= rng.uniform(3, 8)

    return np.round(series, -2)  # округляем до сотен


def build_movements(profiles):
    rng = np.random.default_rng(config.SEED + 1)
    rows = []

    for client in profiles.itertuples(index=False):
        months = month_grid(client.open_date)
        if len(months) == 0:
            continue

        balances = {}
        for product in config.PRODUCTS:
            # часть клиентов может не пользоваться продуктом: поэтому нулик
            if rng.random() < STRUCTURAL_ZERO[product][client.segment]:
                balances[product] = np.zeros(len(months))
            else:
                level = BASE_LEVEL[product][client.segment]
                balances[product] = make_series(rng, months, level, client.is_salary)

        block = pd.DataFrame(
            {
                "client_id": client.client_id,
                "report_date": months,
                "deposit": balances["deposit"],
                "savings": balances["savings"],
                "card": balances["card"],
            }
        )
        
        # создаем вид, что у нас есть пропуски в данных
        # if len(block) > 3:
        #     drop = rng.random(len(block)) < 0.02
        #     drop[0] = False
        #     block = block[~drop]

        rows.append(block)

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    profiles = pd.read_parquet(config.PROFILES_PATH)
    movements = build_movements(profiles)
    movements.to_parquet(config.MOVEMENTS_PATH, index=False)

    print(movements.head(15).to_string(index=False), "\n")
    print(f"всего строк: {len(movements):,}")
    print(
        f"месяцев на клиента в среднем: {movements.groupby('client_id').size().mean()}\n"
    )

    print("доля нулей по продуктам:")
    for p in config.PRODUCTS:
        print(f"{p}: {(movements[p] == 0).mean():.3f}")
