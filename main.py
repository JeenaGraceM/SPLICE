from pyspark.sql import SparkSession
from skew_data import make_skewed_data
from skew_detector import SkewDetector
from adapter import parse_person1_output
from mitigation import salted_aggregate, choose_mitigation

spark = SparkSession.builder.appName("skew-mitigation-test").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1. Generate skewed test data (pre-shuffle — important, see note below)
df = make_skewed_data(spark, n=200_000, hot_key="K1", hot_frac=0.6)
print(f"Generated {df.count()} rows")

# 2. Run Person 1's real detector
detector = SkewDetector()
result = detector.detect_from_dataframe(stage_id=7, df=df, key_col="key")
skewed_partitions = SkewDetector.skewed_only(result)
print(f"Skewed partitions: {skewed_partitions}")

# 3. Adapt their output into what our mitigation functions expect
hot_keys = parse_person1_output(skewed_partitions, df=df, key_col="key")
print(f"Detected hot keys: {hot_keys}")

# 4. Apply mitigation (this is an associative agg -> salting path)
strategy = choose_mitigation("associative_agg", agg_fn="sum")
print(f"Chosen strategy: {strategy}")

mitigated_result = salted_aggregate(df, "key", "value", hot_keys, agg_fn="sum")

# 5. Validate correctness against plain groupBy (ground truth)
ground_truth = df.groupBy("key").agg({"value": "sum"}).withColumnRenamed("sum(value)", "final_val")

mitigated_sorted = mitigated_result.orderBy("key").collect()
truth_sorted = ground_truth.orderBy("key").collect()

mismatches = [
    (a, b) for a, b in zip(mitigated_sorted, truth_sorted)
    if abs(a["final_val"] - b["final_val"]) > 0.001
]

print(f"\nRows compared: {len(truth_sorted)}")
print(f"Mismatches: {len(mismatches)}")
print("CORRECTNESS CHECK PASSED" if not mismatches else "CORRECTNESS CHECK FAILED")

# ---------------------------------------------------------------
# 6. Sanity check: min/max salting (previously untested end-to-end,
#    and previously silently WRONG before the mitigation.py fix,
#    since phase-2 merge always used sum() regardless of agg_fn).
# ---------------------------------------------------------------
for check_fn in ["min", "max"]:
    strategy = choose_mitigation("associative_agg", agg_fn=check_fn)
    mitigated_check = salted_aggregate(df, "key", "value", hot_keys, agg_fn=check_fn)
    truth_check = df.groupBy("key").agg({"value": check_fn}) \
                     .withColumnRenamed(f"{check_fn}(value)", "final_val")

    mit_sorted = mitigated_check.orderBy("key").collect()
    truth_sorted_check = truth_check.orderBy("key").collect()
    mismatches_check = [
        (a, b) for a, b in zip(mit_sorted, truth_sorted_check)
        if a["final_val"] != b["final_val"]
    ]
    status = "PASSED" if not mismatches_check else "FAILED"
    print(f"[{check_fn}] strategy={strategy} mismatches={len(mismatches_check)} -> {status}")

# ---------------------------------------------------------------
# 7. Guardrail check: avg should be refused, not silently wrong.
# ---------------------------------------------------------------
try:
    salted_aggregate(df, "key", "value", hot_keys, agg_fn="avg")
    print("[avg] ERROR: expected ValueError, but none was raised")
except ValueError as e:
    print(f"[avg] correctly rejected: {e}")

spark.stop()