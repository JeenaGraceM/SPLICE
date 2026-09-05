
import time

from skew_detector import SkewDetector
from skew_data import make_skewed_data
from adapter import parse_person1_output
from mitigation import salted_aggregate, choose_mitigation


def run_single_experiment(
    spark,
    skew_ratio,
    num_records=200_000,
    num_partitions=10
):
    """
    Run one SPLICE evaluation experiment.

    Measures:
    - baseline aggregation runtime
    - detection runtime
    - mitigation runtime
    - end-to-end runtime
    - speedups
    - detected skew
    - correctness
    """

    print("=" * 60)
    print(f"Running experiment: {skew_ratio * 100:.0f}% skew")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Generate data
    # ---------------------------------------------------------

    df = make_skewed_data(
        spark,
        n=num_records,
        hot_key="K1",
        hot_frac=skew_ratio
    )

    df = df.repartition(num_partitions)

    df.cache()
    df.count()

    # ---------------------------------------------------------
    # 2. Baseline aggregation
    # ---------------------------------------------------------

    start = time.perf_counter()

    baseline = (
        df.groupBy("key")
          .agg({"value": "sum"})
          .withColumnRenamed("sum(value)", "baseline_val")
          .collect()
    )

    baseline_runtime = time.perf_counter() - start

    print(f"Baseline runtime: {baseline_runtime:.4f} seconds")

    # ---------------------------------------------------------
    # 3. Detection
    # ---------------------------------------------------------

    detector = SkewDetector()

    start = time.perf_counter()

    detection_result = detector.detect_from_dataframe(
        stage_id=7,
        df=df,
        key_col="key",
        num_partitions=num_partitions
    )

    detection_runtime = time.perf_counter() - start

    print(f"Detection runtime: {detection_runtime:.4f} seconds")

    # ---------------------------------------------------------
    # 4. Extract skewed partitions / hot keys
    # ---------------------------------------------------------

    skewed_partitions = SkewDetector.skewed_only(
        detection_result
    )

    hot_keys = parse_person1_output(skewed_partitions)

    print("Detected skewed partitions:", len(skewed_partitions))
    print("Detected hot keys:", hot_keys)

    # ---------------------------------------------------------
    # 5. Mitigation
    # ---------------------------------------------------------

    mitigation_type = choose_mitigation("associative_agg")

    start = time.perf_counter()

    mitigated = salted_aggregate(
        df=df,
        key_col="key",
        value_col="value",
        hot_keys=hot_keys,
        n_salts=10,
        agg_fn="sum"
    )

    mitigated_result = mitigated.collect()

    mitigation_runtime = time.perf_counter() - start

    print(f"Mitigation runtime: {mitigation_runtime:.4f} seconds")

    # ---------------------------------------------------------
    # 6. End-to-end runtime
    # ---------------------------------------------------------

    end_to_end_runtime = (
        detection_runtime +
        mitigation_runtime
    )

    # ---------------------------------------------------------
    # 7. Speedup
    # ---------------------------------------------------------

    mitigation_speedup = (
        baseline_runtime / mitigation_runtime
        if mitigation_runtime > 0
        else None
    )

    end_to_end_speedup = (
        baseline_runtime / end_to_end_runtime
        if end_to_end_runtime > 0
        else None
    )

    # ---------------------------------------------------------
    # 8. Correctness
    # ---------------------------------------------------------

    baseline_dict = {
        row["key"]: row["baseline_val"]
        for row in baseline
    }

    mitigated_dict = {
        row["key"]: row["final_val"]
        for row in mitigated_result
    }

    correctness = baseline_dict == mitigated_dict

    print("Correctness:", correctness)

    # ---------------------------------------------------------
    # 9. Return results
    # ---------------------------------------------------------

    return {
        "records": num_records,
        "skew_ratio": skew_ratio,
        "skew_percent": skew_ratio * 100,
        "num_partitions": num_partitions,

        "baseline_runtime": baseline_runtime,
        "detection_runtime": detection_runtime,
        "mitigation_runtime": mitigation_runtime,
        "end_to_end_runtime": end_to_end_runtime,

        "mitigation_speedup": mitigation_speedup,
        "end_to_end_speedup": end_to_end_speedup,

        "num_skewed_partitions": len(skewed_partitions),
        "hot_keys": hot_keys,

        "mitigation_type": mitigation_type,
        "correctness": correctness
    }
