from pyspark.sql import functions as F

def parse_person1_output(skewed_partitions, df=None, key_col=None):
    """
    Converts Person 1's skewed_only() output into the hot_keys format
    our mitigation functions (salted_aggregate, split_join) expect.

    skewed_partitions: list of dicts like
        {"partition_id": 2, "is_skewed": true, "size": 850, "hot_keys": ["HOT"]}
    """
    hot_keys = []

    for p in skewed_partitions:
        if p["hot_keys"] is not None:
            # Synthetic / key-aware mode — keys given directly
            hot_keys.extend([(k, p["size"]) for k in p["hot_keys"]])
        else:
            # Real Spark metrics mode — hot_keys is null, sample it ourselves
            if df is None or key_col is None:
                raise ValueError(
                    f"Partition {p['partition_id']} has no hot_keys and no df/key_col "
                    "provided to sample it ourselves."
                )
            sampled = (
                df.withColumn("_partition_id", F.spark_partition_id())
                  .filter(F.col("_partition_id") == p["partition_id"])
                  .groupBy(key_col)
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(3)
                  .collect()
            )
            hot_keys.extend([(row[key_col], row["count"]) for row in sampled])

    return hot_keys