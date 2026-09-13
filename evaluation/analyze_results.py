import pandas as pd

INPUT_FILE = "evaluation/results/experiment_results.csv"
OUTPUT_FILE = "evaluation/results/experiment_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    summary = (
        df.groupby("skew_percent")
        .agg(
            baseline_mean=("baseline_runtime", "mean"),
            baseline_median=("baseline_runtime", "median"),
            detection_mean=("detection_runtime", "mean"),
            mitigation_mean=("mitigation_runtime", "mean"),
            end_to_end_mean=("end_to_end_runtime", "mean"),
            mitigation_speedup_mean=("mitigation_speedup", "mean"),
            end_to_end_speedup_mean=("end_to_end_speedup", "mean"),
            correctness_rate=("correctness", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nExperimental Summary")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()