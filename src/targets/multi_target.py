"""
Собираем training dataset.
Берем собранные фичи и добавляем к ним таргеты на 12 месяцев вперед.
"""


from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession

import src.config as config

def add_targets(df):
    w = Window.partitionBy("client_id", "product").orderBy("report_date")

    for i in range(1, 13):
        df = df.withColumn(f"target_{i}m", F.lead("balance", i).over(w))
        
    return df

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("supervised").getOrCreate()

    feats = spark.read.parquet(str(config.FEATURES_PATH))
    supervised_df = add_targets(feats)

    supervised_df.write.mode("overwrite").parquet(str(config.SUPERVISED_PATH))

    spark.stop()