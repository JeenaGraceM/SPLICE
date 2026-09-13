import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "evaluation/results/experiment_summary.csv"
OUTPUT_FILE = "evaluation/results/runtime_comparison.png"

df = pd.read_csv(INPUT_FILE)

plt.figure(figsize=(9, 5))

plt.plot(
    df["skew_percent"],
    df["baseline_mean"],
    marker="o",
    label="Vanilla Spark"
)

plt.plot(
    df["skew_percent"],
    df["mitigation_mean"],
    marker="o",
    label="SPLICE mitigation"
)

plt.plot(
    df["skew_percent"],
    df["end_to_end_mean"],
    marker="o",
    label="SPLICE end-to-end"
)

plt.xlabel("Skew level (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison Across Skew Levels")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print(f"Chart saved to: {OUTPUT_FILE}")