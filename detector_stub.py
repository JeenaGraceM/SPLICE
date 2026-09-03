# placeholder detector, shaped like Person 1's real JSON output

from pyspark.sql import functions as F

def stub_detect_skew(df, key_col, threshold_multiplier=3.0):
    counts = df.groupBy(key_col).count()
    median = counts.approxQuantile("count", [0.5], 0.01)[0]
    threshold = median * threshold_multiplier

    rows = counts.collect()
    partitions = [
        {"id": i, "size": row["count"], "skewed": row["count"] > threshold}
        for i, row in enumerate(rows)
    ]
    return {
        "stage_id": 0,
        "threshold": threshold,
        "median": median,
        "partitions": partitions,
        "_key_lookup": {i: row[key_col] for i, row in enumerate(rows)}  # for our own use
    }