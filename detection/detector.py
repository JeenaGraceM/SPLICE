from statistics import median, mean

from .threshold import AdaptiveThreshold


class SkewDetector:
    """
    Detects skewed partitions using a workload-adaptive threshold.

    Two input modes:

    1. Size-only mode -- partition_sizes = {partition_id: size}
       Matches real Spark shuffle-write metrics, which report bytes
       per partition but not which keys are inside. hot_keys will
       be None in this mode.

    2. Key-aware mode -- partition_keys = {partition_id: {key: count, ...}}
       Used for synthetic test data where the actual key distribution
       is known (e.g. Person 3's generated datasets). Partition size
       is derived as sum(counts), and hot_keys is populated with the
       key(s) responsible for most of a flagged partition's size.
    """

    def __init__(self, z=3.5, hot_key_share=0.5, threshold_strategy=None):
        """
        threshold_strategy: an object with a .calculate(sizes) method,
        e.g. AdaptiveThreshold() (default) or FixedThreshold(). Lets
        the same detector be run with different thresholding schemes
        for direct comparison (see detection/sensitivity.py).
        """
        self.threshold_calculator = threshold_strategy or AdaptiveThreshold(z=z)
        # A key counts as "hot" if it accounts for at least this
        # fraction of its partition's total size.
        self.hot_key_share = hot_key_share

    def detect(self, stage_id, partition_sizes=None, partition_keys=None):
        """
        Detect skewed partitions for a Spark stage.

        Parameters:
            stage_id (int): Spark stage ID.
            partition_sizes (dict, optional): partition_id -> size.
            partition_keys (dict, optional): partition_id -> {key: count}.
                If provided, partition_sizes is derived automatically
                and hot_keys will be populated for skewed partitions.

        Returns:
            dict: stage-level stats plus a per-partition breakdown.
        """
        if partition_keys is not None:
            partition_sizes = {
                pid: sum(keys.values())
                for pid, keys in partition_keys.items()
            }

        if not partition_sizes:
            return {
                "stage_id": stage_id,
                "median_partition_size": 0.0,
                "average_partition_size": 0.0,
                "threshold": 0.0,
                "threshold_ratio": 0.0,
                "partitions": []
            }

        sizes = list(partition_sizes.values())
        median_size = median(sizes)
        average_size = mean(sizes)
        threshold = self.threshold_calculator.calculate(sizes)
        threshold_ratio = 0.0 if median_size == 0 else threshold / median_size

        partitions = []
        for partition_id, size in partition_sizes.items():
            skew_ratio = 0.0 if median_size == 0 else round(size / median_size, 2)
            is_skewed = size > threshold

            hot_keys = None
            if is_skewed and partition_keys is not None:
                keys_in_partition = partition_keys[partition_id]
                hot_keys = [
                    k for k, cnt in keys_in_partition.items()
                    if cnt >= self.hot_key_share * size
                ]

            partitions.append({
                "partition_id": partition_id,
                "size": size,
                "skew_ratio": skew_ratio,
                "is_skewed": is_skewed,
                "hot_keys": hot_keys
            })

        return {
            "stage_id": stage_id,
            "median_partition_size": round(median_size, 2),
            "average_partition_size": round(average_size, 2),
            "threshold": round(threshold, 2),
            "threshold_ratio": round(threshold_ratio, 2),
            "partitions": partitions
        }

    def detect_from_dataframe(self, stage_id, df, key_col, num_partitions=None):
        """
        Connects the detector to a real Spark DataFrame instead of a
        hand-built dict.

        Skew only exists relative to the SHUFFLE that groupBy/join
        would trigger on key_col -- a DataFrame's current partitioning
        (e.g. from spark.range()) tells you nothing about that. So
        this method repartitions df by key_col first (the same hash
        partitioning Spark itself would do for a real groupBy/join
        shuffle), THEN inspects the resulting per-partition, per-key
        record counts. That mirrors what the Shuffle-Write Monitor
        described in the paper actually observes: sizes as they will
        land for the reduce stage, not before it.

        Parameters:
            stage_id (int): label for this detection run.
            df (pyspark.sql.DataFrame): the DataFrame BEFORE the
                shuffle (i.e. its raw, unshuffled form).
            key_col (str): the column being grouped/joined on.
            num_partitions (int, optional): number of post-shuffle
                partitions to simulate. Defaults to df's current
                partition count.

        Returns:
            dict: same shape as detect().
        """
        from pyspark.sql import functions as F

        n = num_partitions or df.rdd.getNumPartitions()
        shuffled = df.repartition(n, key_col)

        counted = (
            shuffled.withColumn("_partition_id", F.spark_partition_id())
                    .groupBy("_partition_id", key_col)
                    .count()
                    .collect()
        )

        partition_keys = {}
        for row in counted:
            pid = row["_partition_id"]
            key = row[key_col]
            cnt = row["count"]
            partition_keys.setdefault(pid, {})[key] = cnt

        return self.detect(stage_id=stage_id, partition_keys=partition_keys)

    def detect_from_rdd(self, stage_id, rdd, key_fn=None, num_partitions=None):
        """
        RDD-level counterpart to detect_from_dataframe(). Covers jobs
        expressed directly against the RDD API (reduceByKey/groupByKey)
        which bypass Catalyst entirely and therefore can't be reached
        via a DataFrame-only detection path -- this is the gap the
        paper calls out explicitly (RDD pipelines fall outside AQE's
        coverage for the same reason).

        Parameters:
            stage_id (int): label for this detection run.
            rdd: either an RDD of (key, value) pairs already, or a
                plain RDD of arbitrary records plus a key_fn to
                extract the key from each record.
            key_fn (callable, optional): record -> key. Required if
                rdd is not already a pair RDD.
            num_partitions (int, optional): number of post-shuffle
                partitions to simulate. Defaults to rdd's current
                partition count.

        Returns:
            dict: same shape as detect().
        """
        paired = rdd.map(lambda record: (key_fn(record), record)) if key_fn else rdd

        n = num_partitions or paired.getNumPartitions()
        # partitionBy hash-partitions by key -- the same shuffle
        # reduceByKey/groupByKey would trigger -- so we're measuring
        # real post-shuffle partition sizes, not the RDD's original
        # (pre-shuffle) layout.
        shuffled = paired.partitionBy(n)

        def count_per_partition(index, iterator):
            counts = {}
            for key, _ in iterator:
                counts[key] = counts.get(key, 0) + 1
            if counts:
                yield (index, counts)

        collected = shuffled.mapPartitionsWithIndex(count_per_partition).collect()
        partition_keys = dict(collected)

        return self.detect(stage_id=stage_id, partition_keys=partition_keys)

    def detect_from_listener(self, stage_id, listener):
        """
        Uses REAL shuffle-write metrics captured by a live
        ShuffleWriteListener (see live_monitor.py) instead of
        simulating a shuffle. This is the direct implementation of
        the paper's C1 (taps per-task shuffle-write metrics as map
        tasks complete).

        Parameters:
            stage_id (int): the Spark stage to inspect.
            listener (ShuffleWriteListener): an already-attached
                listener that has observed the stage's tasks complete.

        Returns:
            dict: same shape as detect(). hot_keys will be None,
            since real shuffle-write metrics report bytes per
            partition, not key contents.
        """
        partition_sizes = listener.get_partition_sizes(stage_id)
        return self.detect(stage_id=stage_id, partition_sizes=partition_sizes)

    @staticmethod
    def skewed_only(result):
        """
        Filter a detect() result down to just the skewed partitions,
        in the flat shape Person 2's mitigation module consumes:

            {"partition_id": ..., "is_skewed": True, "size": ..., "hot_keys": ...}
        """
        return [
            {
                "partition_id": p["partition_id"],
                "is_skewed": p["is_skewed"],
                "size": p["size"],
                "hot_keys": p["hot_keys"]
            }
            for p in result["partitions"] if p["is_skewed"]
        ]
