from pyspark.sql import functions as F
import numpy as np


def make_skewed_data(spark, n=1_000_000, hot_key="K1", hot_frac=0.5):
    """
    Unchanged. Single-hot-key skew: hot_frac of rows get hot_key,
    the rest get a random key from "0".."999".
    """
    n_hot = int(n * hot_frac)
    hot = spark.range(n_hot).withColumn("key", F.lit(hot_key)) \
                             .withColumn("value", (F.rand() * 100).cast("int"))
    rest = spark.range(n - n_hot).withColumn(
        "key", (F.rand() * 1000).cast("int").cast("string")
    ).withColumn("value", (F.rand() * 100).cast("int"))
    return hot.unionByName(rest)


def make_zipfian_data(spark, n=1_000_000, num_keys=1000, zipf_param=1.5, seed=42):
    """
    Generates data with a Zipfian key distribution instead of a single
    hardcoded hot key -- i.e. several moderately-to-very hot keys with a
    smooth power-law falloff, rather than one artificially dominant key.

    This matches the paper's evaluation requirement for "Zipfian and
    single-hot-key distributions with controllable skew intensity" --
    make_skewed_data() above covers the single-hot-key case, this
    covers the Zipfian case.

    Parameters:
        n (int): total number of rows to generate.
        num_keys (int): size of the key space (keys are "0".."num_keys-1").
        zipf_param (float): Zipf distribution exponent (a in numpy's
            zipf). Higher = more skewed (a few keys dominate more).
            Values around 1.1-2.0 are typical for "organic" skew.
        seed (int): RNG seed for reproducibility.

    Returns:
        Spark DataFrame with columns "key" (string) and "value" (int),
        driven to executors as a single partition then repartitioned by
        the caller/detector as needed.
    """
    rng = np.random.default_rng(seed)

    # Sample n key-indices from a Zipf distribution, clipped into
    # [0, num_keys) so a heavy tail doesn't produce out-of-range keys.
    raw = rng.zipf(zipf_param, size=n)
    key_indices = (raw % num_keys)

    values = rng.integers(0, 100, size=n)

    rows = [(str(int(k)), int(v)) for k, v in zip(key_indices, values)]
    return spark.createDataFrame(rows, ["key", "value"])