import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.detector import SkewDetector


class TestSkewDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SkewDetector()

    def flagged_ids(self, result):
        return sorted(p["partition_id"] for p in result["partitions"] if p["is_skewed"])

    # ------------------------------------------------------------
    # Required test case A: no skew
    # ------------------------------------------------------------
    def test_no_skew(self):
        sizes = {0: 100, 1: 105, 2: 98, 3: 110, 4: 103, 5: 107, 6: 101}
        result = self.detector.detect(stage_id=1, partition_sizes=sizes)
        self.assertEqual(self.flagged_ids(result), [])

    # ------------------------------------------------------------
    # Required test case B: one extreme hot partition
    # ------------------------------------------------------------
    def test_one_extreme_hot_partition(self):
        sizes = {0: 100, 1: 105, 2: 98, 3: 110, 4: 103, 5: 900, 6: 101}
        result = self.detector.detect(stage_id=2, partition_sizes=sizes)
        self.assertEqual(self.flagged_ids(result), [5])

    # ------------------------------------------------------------
    # Required test case C: several moderately hot partitions
    # (this is the case a fixed threshold tends to miss)
    # ------------------------------------------------------------
    def test_multiple_moderately_hot_partitions(self):
        sizes = {0: 100, 1: 105, 2: 300, 3: 280, 4: 103, 5: 290, 6: 101}
        result = self.detector.detect(stage_id=3, partition_sizes=sizes)
        self.assertEqual(self.flagged_ids(result), [2, 3, 5])

    # ------------------------------------------------------------
    # Edge case: empty input
    # ------------------------------------------------------------
    def test_empty_input(self):
        result = self.detector.detect(stage_id=4, partition_sizes={})
        self.assertEqual(result["partitions"], [])
        self.assertEqual(result["threshold"], 0.0)

    # ------------------------------------------------------------
    # Edge case: all partitions identical (mad == 0 fallback path)
    # ------------------------------------------------------------
    def test_all_partitions_equal(self):
        sizes = {0: 100, 1: 100, 2: 100, 3: 100}
        result = self.detector.detect(stage_id=5, partition_sizes=sizes)
        self.assertEqual(self.flagged_ids(result), [])

    # ------------------------------------------------------------
    # Key-aware mode: hot_keys should be populated for the
    # flagged partition and None everywhere else
    # ------------------------------------------------------------
    def test_hot_key_identification(self):
        partition_keys = {
            0: {"key_a": 40, "key_b": 60},
            1: {"key_c": 50, "key_d": 55},
            2: {"HOT": 800, "key_e": 50},
            3: {"key_f": 45, "key_g": 65},
        }
        result = self.detector.detect(stage_id=6, partition_keys=partition_keys)
        skewed = SkewDetector.skewed_only(result)

        self.assertEqual(len(skewed), 1)
        self.assertEqual(skewed[0]["partition_id"], 2)
        self.assertEqual(skewed[0]["hot_keys"], ["HOT"])

        # non-skewed partitions must have hot_keys == None
        for p in result["partitions"]:
            if not p["is_skewed"]:
                self.assertIsNone(p["hot_keys"])

    # ------------------------------------------------------------
    # Handoff format: skewed_only() must match Person 2's schema exactly
    # ------------------------------------------------------------
    def test_skewed_only_schema(self):
        sizes = {0: 100, 1: 105, 2: 98, 3: 850}
        result = self.detector.detect(stage_id=7, partition_sizes=sizes)
        skewed = SkewDetector.skewed_only(result)
        self.assertEqual(len(skewed), 1)
        self.assertEqual(
            set(skewed[0].keys()),
            {"partition_id", "is_skewed", "size", "hot_keys"}
        )


if __name__ == "__main__":
    unittest.main()
