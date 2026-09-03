from pyspark.sql import functions as F

def make_skewed_data(spark, n=1_000_000, hot_key="K1", hot_frac=0.5):
    n_hot = int(n * hot_frac)
    hot = spark.range(n_hot).withColumn("key", F.lit(hot_key)) \
                             .withColumn("value", (F.rand() * 100).cast("int"))
    rest = spark.range(n - n_hot).withColumn(
        "key", (F.rand() * 1000).cast("int").cast("string")
    ).withColumn("value", (F.rand() * 100).cast("int"))
    return hot.unionByName(rest)