from pyspark.sql import functions as F

# Aggregations that are safe to salt: the same reduction can be applied
# independently per salt bucket, then merged with a *matching* reducer
# in phase 2 without changing the mathematical result.
_SALT_SAFE_AGGS = {
    "sum": F.sum,
    "count": F.sum,   # partial counts merge by summing
    "min": F.min,
    "max": F.max,
}

# Aggregations that CANNOT be correctly salted this way -- merging
# partials does not equal the true result (e.g. avg of averages !=
# overall average). Route these to split/other handling instead.
_SALT_UNSAFE_AGGS = {"avg", "mean", "stddev", "variance", "median"}


def choose_mitigation(operator_type, agg_fn=None):
    """
    Decision rule for which mitigation path to use.

    operator_type: "associative_agg" or "join" (or any other join-like type)
    agg_fn: optional, the specific aggregation function e.g. "sum", "avg".
        If provided, catches aggregations that look associative but
        aren't safe to salt (e.g. "avg") and routes them to splitting.
        If omitted, behaves exactly as before (backward compatible).

    Returns: "salting" or "splitting"
    """
    if operator_type == "associative_agg":
        if agg_fn is not None and agg_fn.lower() in _SALT_UNSAFE_AGGS:
            return "splitting"
        return "salting"
    else:
        return "splitting"


def salted_aggregate(df, key_col, value_col, hot_keys, n_salts=10, agg_fn="sum"):
    """
    Mitigates skew for associative aggregations (sum, count, min, max) by
    splitting hot keys into salted sub-keys, aggregating independently,
    then merging with a reducer that matches agg_fn.

    Raises ValueError if agg_fn is not a verified-safe associative
    aggregation (e.g. "avg"), since salting would silently produce an
    incorrect result for those.
    """
    agg_key = agg_fn.lower()
    if agg_key not in _SALT_SAFE_AGGS:
        raise ValueError(
            f"salted_aggregate: '{agg_fn}' is not a verified associative "
            f"aggregation and cannot be safely salted (would silently "
            f"produce incorrect results). Supported: {sorted(_SALT_SAFE_AGGS)}. "
            f"Use split_join or a dedicated non-associative path instead."
        )

    hot_key_set = {k for k, _ in hot_keys}

    salted = df.withColumn(
        "salt",
        F.when(F.col(key_col).isin(hot_key_set), (F.rand() * n_salts).cast("int"))
         .otherwise(F.lit(0))
    ).withColumn("salted_key", F.concat_ws("_", key_col, "salt"))

    # phase 1: partial aggregation on salted keys (spreads hot key across n_salts tasks)
    partial = salted.groupBy("salted_key", key_col).agg(
        {value_col: agg_key}
    ).withColumnRenamed(f"{agg_key}({value_col})", "partial_val")

    # phase 2: merge partials back down to the real key, using the
    # reducer that matches agg_fn (previously this always used F.sum
    # here regardless of agg_fn, which silently broke min/max/avg)
    merge_fn = _SALT_SAFE_AGGS[agg_key]
    final = partial.groupBy(key_col).agg(merge_fn("partial_val").alias("final_val"))
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