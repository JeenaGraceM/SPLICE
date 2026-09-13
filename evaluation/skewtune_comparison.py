import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_FILE = "evaluation/results/skewtune_comparison.csv"


def make_data(spark, n=200_000, hot_frac=0.5):
    n_hot = int(n * hot_frac)

    hot = (
        spark.range(n_hot)
        .withColumn("key", F.lit("K1"))
        .withColumn("value", F.lit(1))
    )

    normal = (
        spark.range(n - n_hot)
        .withColumn("key", (F.rand() * 1000).cast("int").cast("string"))
        .withColumn("value", F.lit(1))
    )

    return hot.unionByName(normal).repartition(10, "key")


def run_vanilla(spark, df):
    start = time.perf_counter()

    result = (
        df.groupBy("key")
        .agg(F.sum("value").alias("total"))
    )

    result.collect()

    return time.perf_counter() - start


def run_skewtune_style(spark, df, n_splits=10):
    """
    SkewTune-style oracle splitting:
    the hot key is known in advance and its records are
    divided across multiple salt buckets before aggregation.
    """

    start = time.perf_counter()

    split_df = (
        df.withColumn(
            "split_id",
            F.when(
                F.col("key") == "K1",
                (F.rand() * n_splits).cast("int")
            ).otherwise(F.lit(0))
        )
    )

    partial = (
        split_df
        .groupBy("key", "split_id")
        .agg(F.sum("value").alias("partial_total"))
    )

    result = (
        partial
        .groupBy("key")
        .agg(F.sum("partial_total").alias("total"))
    )

    result.collect()

    return time.perf_counter() - start


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SPLICE-SkewTune-Comparison")
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )

    results = []

    try:
        for skew_ratio in [0.3, 0.5, 0.7, 0.8]:
            df = make_data(
                spark,
                n=200_000,
                hot_frac=skew_ratio
            )
            df.cache()
            df.count()

            for trial in range(1, 4):

                vanilla_runtime = run_vanilla(spark, df)

                skt_runtime = run_skewtune_style(
                    spark,
                    df,
                    n_splits=10
                )

                results.append({
                    "skew_percent": skew_ratio * 100,
                    "trial": trial,
                    "vanilla_runtime": vanilla_runtime,
                    "skewtune_style_runtime": skt_runtime,
                    "speedup": vanilla_runtime / skt_runtime,
                })

                print(
                    f"Skew={skew_ratio * 100:.0f}% "
                    f"Trial={trial}: "
                    f"Vanilla={vanilla_runtime:.4f}s | "
                    f"SkewTune-style={skt_runtime:.4f}s | "
                    f"Speedup={vanilla_runtime / skt_runtime:.3f}x"
                )

            df.unpersist()

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nSkewTune-style comparison complete.")
        print(f"Results saved to: {OUTPUT_FILE}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()