import csv
import os
from pyspark.sql import SparkSession

from evaluation.experiment import run_single_experiment


SKEW_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80]
TRIALS_PER_LEVEL = 3

OUTPUT_FILE = "evaluation/results/experiment_results.csv"


def main():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("SPLICE-Sweep")
        .getOrCreate()
    )

    results = []

    try:
        for skew_ratio in SKEW_LEVELS:
            for trial in range(1, TRIALS_PER_LEVEL + 1):
                print(
                    f"\nRunning skew={skew_ratio * 100:.0f}% "
                    f"(trial {trial}/{TRIALS_PER_LEVEL})"
                )

                result = run_single_experiment(
                    spark=spark,
                    skew_ratio=skew_ratio,
                    num_records=200_000,
                    num_partitions=10,
                )

                result["trial"] = trial
                results.append(result)

                print(
                    f"Baseline: {result['baseline_runtime']:.4f}s | "
                    f"Mitigation: {result['mitigation_runtime']:.4f}s | "
                    f"E2E: {result['end_to_end_runtime']:.4f}s | "
                    f"Correct: {result['correctness']}"
                )

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        fieldnames = [
            "trial",
            "records",
            "skew_ratio",
            "skew_percent",
            "num_partitions",
            "baseline_runtime",
            "detection_runtime",
            "mitigation_runtime",
            "end_to_end_runtime",
            "mitigation_speedup",
            "end_to_end_speedup",
            "num_skewed_partitions",
            "hot_keys",
            "mitigation_type",
            "correctness",
        ]

        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = result.copy()
                row["hot_keys"] = str(row["hot_keys"])
                writer.writerow(row)

        print("\n" + "=" * 60)
        print("SWEEP COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {OUTPUT_FILE}")
        print(f"Total experiments: {len(results)}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()