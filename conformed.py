"""
Берем сырые данные, убираем дубли, делаем непрерывную календарную сетку
по каждому клиенту
"""

from pyspark.sql import functions as F
from pyspark.sql import Window


# TODO