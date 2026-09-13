"""
Sensitivity / accuracy measurement tools for the skew detector.

This module is scoped narrowly to Person 1's responsibility: producing
the RAW true/false-positive numbers and threshold comparisons the
paper's Phase 3 ("test sensitivity across ... single extreme hot key,
several moderately hot keys, and no skew at all") and evaluation
section ("Detection Accuracy" metric) call for.

Turning these numbers into the actual report -- charts, tables next to
runtime/speedup/overhead metrics -- is Person 3's evaluation work and
deliberately stays out of this file.
"""

import random

from .detector import SkewDetector
from .threshold import AdaptiveThreshold, FixedThreshold


def generate_synthetic_stage(num_partitions=20, base_size=1000, noise=0.05,
                              num_hot=1, skew_ratio=5.0, seed=None):
    """
    Builds a synthetic stage with known ground-truth hot partitions,
    so detection results can be scored against a known answer instead
    of eyeballed.

    num_hot=0            -> the "no skew" test case
    num_hot=1             -> the "one extreme hot partition" test case
    num_hot>1, moderate skew_ratio -> "several moderately hot" case

    Returns:
        (partition_sizes: dict, ground_truth_hot_ids: set)
    """
    rng = random.Random(seed)
    hot_ids = set(rng.sample(range(num_partitions), num_hot)) if num_hot > 0 else set()

    sizes = {}
    for pid in range(num_partitions):
        size = base_size * (1 + rng.uniform(-noise, noise))
        if pid in hot_ids:
            size *= skew_ratio
        sizes[pid] = round(size)

    return sizes, hot_ids


def score_detection(partition_sizes, ground_truth_hot_ids, threshold_strategy):
    """
    Runs the detector once and scores its output against the known
    ground truth.

    Returns a dict of true_positive / false_positive / false_negative /
    true_negative counts, plus the raw set of flagged partition ids.
    """
    detector = SkewDetector(threshold_strategy=threshold_strategy)
    result = detector.detect(stage_id=0, partition_sizes=partition_sizes)

    flagged = {p["partition_id"] for p in result["partitions"] if p["is_skewed"]}
    all_ids = set(partition_sizes.keys())

    tp = len(flagged & ground_truth_hot_ids)
    fp = len(flagged - ground_truth_hot_ids)
    fn = len(ground_truth_hot_ids - flagged)
    tn = len(all_ids - flagged - ground_truth_hot_ids)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "flagged": sorted(flagged),
    }


def run_sensitivity_sweep(
    skew_scenarios=None,
    strategies=None,
    num_partitions=20,
    trials_per_scenario=20,
):
    """
    Runs score_detection() across a grid of skew scenarios and
    threshold strategies, averaging over several random trials per
    scenario for stability.

    skew_scenarios: list of dicts describing each scenario, e.g.
        [
          {"label": "no_skew",            "num_hot": 0, "skew_ratio": 1.0},
          {"label": "one_extreme",        "num_hot": 1, "skew_ratio": 8.0},
          {"label": "several_moderate",   "num_hot": 4, "skew_ratio": 2.5},
        ]
      Defaults to exactly the three cases the paper's methodology
      calls for if not provided.

    strategies: dict of {name: threshold_strategy_instance}.
      Defaults to comparing AdaptiveThreshold vs FixedThreshold.

    Returns:
        list of result rows (one per scenario x strategy), each a flat
        dict ready to be handed to Person 3 for tabulation/plotting.
    """
    if skew_scenarios is None:
        skew_scenarios = [
            {"label": "no_skew", "num_hot": 0, "skew_ratio": 1.0},
            {"label": "one_extreme_hot_partition", "num_hot": 1, "skew_ratio": 8.0},
            {"label": "several_moderately_hot_partitions", "num_hot": 4, "skew_ratio": 2.5},
        ]

    if strategies is None:
        strategies = {
            "adaptive_median_mad": AdaptiveThreshold(),
            "fixed_3x_median": FixedThreshold(skew_factor=3.0),
        }

    rows = []
    for scenario in skew_scenarios:
        for strategy_name, strategy in strategies.items():
            totals = {"true_positive": 0, "false_positive": 0,
                      "false_negative": 0, "true_negative": 0}

            for trial in range(trials_per_scenario):
                sizes, ground_truth = generate_synthetic_stage(
                    num_partitions=num_partitions,
                    num_hot=scenario["num_hot"],
                    skew_ratio=scenario["skew_ratio"],
                    seed=trial,
                )
                scored = score_detection(sizes, ground_truth, strategy)
                for k in totals:
                    totals[k] += scored[k]

            precision = (
                totals["true_positive"] /
                (totals["true_positive"] + totals["false_positive"])
                if (totals["true_positive"] + totals["false_positive"]) > 0 else None
            )
            recall = (
                totals["true_positive"] /
                (totals["true_positive"] + totals["false_negative"])
                if (totals["true_positive"] + totals["false_negative"]) > 0 else None
            )

            rows.append({
                "scenario": scenario["label"],
                "strategy": strategy_name,
                "trials": trials_per_scenario,
                "true_positive": totals["true_positive"],
                "false_positive": totals["false_positive"],
                "false_negative": totals["false_negative"],
                "true_negative": totals["true_negative"],
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
            })

    return rows
