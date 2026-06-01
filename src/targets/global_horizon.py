"""
Тут собираем training dataset в длинном формате для одной модели
"""

from pyspark.sql import functions as F
from pyspark.sql import Window, DataFrame
from pyspark.sql import SparkSession

import src.config as config


def build_global_horizon_dataset(df: DataFrame):
    """
    Принимает датасет с фичами, считает лиды и разворачивает в длинный формат
    """
    w = Window.partitionBy("client_id", "product").orderBy("report_date")

    for i in range(1, 13):
        df = df.withColumn(f"target_{i}m", F.lead("balance", i).over(w))

    stack_pairs = ", ".join([f"{i}, target_{i}m" for i in range(1, 13)])
    stack_expr = f"stack(12, {stack_pairs}) as (horizon, target)"

    df_long = df.select("*", F.expr(stack_expr))

    drop_cols = [f"target_{i}m" for i in range(1, 13)]
    df_long = df_long.drop(*drop_cols)

    # добавляем фичу: целевой месяц
    df_long = df_long.withColumn(
        "target_date", F.add_months(F.col("report_date"), F.col("horizon"))
    )

    df_long = df_long.withColumn(
        "target_month", F.month(F.col("target_date")).cast("string")
    ).drop("target_date")

    # удаляем строки, где нет известного таргета
    df_long = df_long.filter(F.col("target").isNotNull())

    return df_long


if __name__ == "__main__":
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("supervised_horizon")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )

    feats = spark.read.parquet(str(config.FEATURES_PATH))
    supervised_df = build_global_horizon_dataset(feats)
    supervised_df.write.mode("overwrite").parquet(str(config.SUPERVISED_HORIZON_PATH))

    print(f"Всего строк у нас {supervised_df.count()}")
    print(f"Какие колонки у нас: {' '.join(supervised_df.columns)}\n")
    print("Пример схемы данных:")
    supervised_df.printSchema()

    print("Первая строка:")
    supervised_df.show(1, truncate=False)

    spark.stop()
