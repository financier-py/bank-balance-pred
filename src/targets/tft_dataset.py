"""
Подготовка датасета для Temporal Fusion Transformer
"""

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

import src.config as config


def prepare_tft_dataset(df):
    min_date_row = df.select(F.min("report_date").alias("min_date")).collect()[0]
    min_date = min_date_row["min_date"]

    df = df.withColumn(
        "time_idx",
        F.round(F.months_between(F.col("report_date"), F.lit(min_date))).cast(
            "integer"
        ),
    )

    lag_cols = [
        c for c in df.columns if any(x in c for x in ["lag_", "roll_", "trend_"])
    ]
    df = df.fillna(0.0, subset=lag_cols)

    for col in config.TFT_ALL_CATEGORICALS:
        df = df.withColumn(col, F.col(col).cast("string"))

    for col in config.TFT_ALL_REALS:
        df = df.withColumn(col, F.col(col).cast("float"))

    return df


if __name__ == "__main__":
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("tft_dataset")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )

    feats = spark.read.parquet(str(config.FEATURES_PATH))
    tft_df = prepare_tft_dataset(feats)

    tft_df.write.mode("overwrite").parquet(str(config.TFT_DS_PATH))

    print(f"Всего строк: {tft_df.count():,}")

    spark.stop()
