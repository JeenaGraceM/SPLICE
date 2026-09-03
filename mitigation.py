from pyspark.sql import functions as F

def choose_mitigation(operator_type):
    """Decision rule: associative aggregations -> salting, everything else -> splitting."""
    if operator_type == "associative_agg":
        return "salting"
    else:
        return "splitting"


def salted_aggregate(df, key_col, value_col, hot_keys, n_salts=10, agg_fn="sum"):
    """
    Mitigates skew for associative aggregations (sum, count) by splitting
    hot keys into salted sub-keys, aggregating independently, then merging.
    """
    hot_key_set = {k for k, _ in hot_keys}

    salted = df.withColumn(
        "salt",
        F.when(F.col(key_col).isin(hot_key_set), (F.rand() * n_salts).cast("int"))
         .otherwise(F.lit(0))
    ).withColumn("salted_key", F.concat_ws("_", key_col, "salt"))

    # phase 1: partial aggregation on salted keys (spreads hot key across n_salts tasks)
    partial = salted.groupBy("salted_key", key_col).agg(
        {value_col: agg_fn}
    ).withColumnRenamed(f"{agg_fn}({value_col})", "partial_val")

    # phase 2: merge partials back down to the real key
    final = partial.groupBy(key_col).agg(F.sum("partial_val").alias("final_val"))
    return final


def split_join(big_df, small_df, key_col, hot_keys, n_splits=10):
    """
    Mitigates skew for joins by sub-dividing the hot side of the join
    into balanced chunks, joined against a replicated build-side.
    """
    hot_key_set = {k for k, _ in hot_keys}

    big_hot = big_df.filter(F.col(key_col).isin(hot_key_set)) \
                     .withColumn("split_id", (F.rand() * n_splits).cast("int"))
    big_cold = big_df.filter(~F.col(key_col).isin(hot_key_set))

    small_hot = small_df.filter(F.col(key_col).isin(hot_key_set))
    small_replicated = small_hot.crossJoin(
        big_df.sparkSession.range(n_splits).withColumnRenamed("id", "split_id")
    )

    joined_hot = big_hot.join(small_replicated, [key_col, "split_id"])
    joined_cold = big_cold.join(small_df, key_col)

    return joined_hot.drop("split_id").unionByName(joined_cold)