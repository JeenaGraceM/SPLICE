# LEGACY / UNUSED: placeholder detector, shaped like Person 1's real
# JSON output. This was scaffolding used before detection/detector.py
# (Person 1's real SkewDetector) existed, so the rest of the pipeline
# could be built and tested against a fake-but-plausible shape.
#
# main.py imports the real detector via skew_detector.py, which
# re-exports detection.detector.SkewDetector -- this stub is NOT part
# of that import chain and does not run in any current pipeline.
# Kept here for reference / as an offline mock if Spark isn't available.
# Safe to delete once confirmed nothing depends on it.

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