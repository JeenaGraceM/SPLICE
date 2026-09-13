import pandas as pd

INPUT_FILE = "evaluation/results/experiment_results.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    metrics = {
        "baseline_runtime": "Baseline",
        "mitigation_runtime": "SPLICE mitigation",
        "end_to_end_runtime": "SPLICE end-to-end",
    }

    print("\nRuntime Percentiles")
    print("=" * 70)

    for column, label in metrics.items():
        p50 = df[column].quantile(0.50)
        p99 = df[column].quantile(0.99)

        print(f"{label}:")
        print(f"  P50 = {p50:.4f} seconds")
        print(f"  P99 = {p99:.4f} seconds")


if __name__ == "__main__":
    main()