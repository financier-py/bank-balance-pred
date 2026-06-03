"""
Делает фичи

Берем нашу сетку и доклеиваем признаки.

Три продукта делаем в длинный формат. Будет колонка product и одна balance под нее.
Далее считаем оконные фичи один раз для каждого (клиент, продукт)

Какие у нас будут фичи:
  - лаги, скользящие средние, волатильность, тренд, мин и макс
  - нули TODO
  - календарь
  - пол, возраст, сегмент, регион, зарплатный, сколько в банке уже

Все фичи должны смотреть только в прошлое
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession

import src.config as config


def to_long(conformed):
    """Deposit/savings/card в колонки product + balance."""
    return conformed.select(
        "client_id",
        "report_date",
        F.expr(
            "stack(3, 'deposit', deposit, 'savings', savings, 'card', card)"
            " as (product, balance)"
        ),
    )


def add_history_features(df):
    """Лаги, скользящие средние, волатильность, тренд, мин/макс - по (клиент, продукт)."""
    w = Window.partitionBy("client_id", "product").orderBy("report_date")

    # лаги
    for k in (1, 2, 3, 6, 12):
        df = df.withColumn(f"lag_{k}", F.lag("balance", k).over(w))

    # скользящие средние за последние n месяцев
    for k in (3, 6, 12):
        ww = w.rowsBetween(-(k - 1), 0)
        df = df.withColumn(f"roll_mean_{k}", F.avg("balance").over(ww))

    # волатильность
    df = df.withColumn("roll_std_6", F.stddev("balance").over(w.rowsBetween(-5, 0)))

    # мин макс за год
    df = df.withColumn("roll_min_12", F.min("balance").over(w.rowsBetween(-11, 0)))
    df = df.withColumn("roll_max_12", F.max("balance").over(w.rowsBetween(-11, 0)))

    # тренд за 3 месяца
    df = df.withColumn("trend_3", F.col("balance") - F.col("lag_3"))

    return df


def add_calendar_features(df):
    """Фича с календарем"""
    return (
        df.withColumn("month", F.month("report_date"))
        .withColumn("quarter", F.quarter("report_date"))
        .withColumn("is_december", (F.month("report_date") == 12).cast("int"))
        .withColumn("is_summer", F.month("report_date").isin(6, 7, 8).cast("int"))
    )


def add_profile_features(df, profiles):
    """Соединяем нашу статистику и считаем стаж клиента"""
    df = df.join(profiles, "client_id", "left")
    df = df.withColumn(
        "tenure_months",
        F.months_between("report_date", "open_date").cast("int"),
    ).drop("open_date")
    return df


def build_features(conformed, profiles):
    df = to_long(conformed)
    df = add_history_features(df)
    df = add_calendar_features(df)
    df = add_profile_features(df, profiles)
    return df


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("features").getOrCreate()

    conformed = spark.read.parquet(str(config.CONFORMED_PATH))
    profiles = spark.read.parquet(str(config.PROFILES_PATH))

    feats = build_features(conformed, profiles)
    feats.write.mode("overwrite").parquet(str(config.FEATURES_PATH))

    print(f"строк: {feats.count()}")
    print(f"колонок: {len(feats.columns)}")

    print(", ".join(feats.columns))

    feats.show(1)

    spark.stop()
