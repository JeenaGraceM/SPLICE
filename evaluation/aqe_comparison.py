import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_FILE = "evaluation/results/aqe_comparison.csv"


def run_join(spark, aqe_enabled):
    spark.conf.set("spark.sql.adaptive.enabled", str(aqe_enabled).lower())
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", str(aqe_enabled).lower())

    # Large skewed fact table
    hot_rows = 100_000
    normal_rows = 100_000

    fact_hot = (
        spark.range(hot_rows)
        .withColumn("key", F.lit("HOT"))
        .withColumn("value", F.lit(1))
    )

    fact_normal = (
        spark.range(normal_rows)
        .withColumn("key", (F.col("id") % 999).cast("string"))
        .withColumn("value", F.lit(1))
    )

    fact = fact_hot.unionByName(fact_normal)

    # Small dimension table
    dim = spark.range(1000).withColumn(
        "key",
        F.col("id").cast("string")
    )

    # Make the hot key available in the dimension table
    dim = dim.unionByName(
        spark.createDataFrame(
            [(-1, "HOT")],
            ["id", "key"]
        )
    )

    fact = fact.repartition(10, "key")
    dim = dim.repartition(10, "key")

    start = time.perf_counter()

    result = (
        fact.join(dim, "key", "inner")
        .groupBy("key")
        .agg(F.sum("value").alias("total"))
    )

    result.count()

    return time.perf_counter() - start


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SPLICE-AQE-Comparison")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )

    results = []

    try:
        for aqe_enabled in [False, True]:
            for trial in range(1, 4):
                print(
                    f"Running join with AQE "
                    f"{'enabled' if aqe_enabled else 'disabled'} "
                    f"(trial {trial}/3)..."
                )

                runtime = run_join(spark, aqe_enabled)

                results.append({
                    "aqe_enabled": aqe_enabled,
                    "trial": trial,
                    "runtime_seconds": runtime,
                })

                print(f"Runtime: {runtime:.4f} seconds")

        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        print("\nAQE comparison complete.")
        print(f"Results saved to: {OUTPUT_FILE}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()