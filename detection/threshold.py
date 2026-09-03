from statistics import median


class AdaptiveThreshold:
    """
    Calculates a workload-adaptive skew threshold using
    Median + MAD (Median Absolute Deviation).

    This is more robust than a fixed skew factor (e.g. 3x median)
    because MAD is not distorted by a few extreme outliers the way
    mean/std-dev would be. This lets the detector correctly flag
    BOTH a single extreme hot partition AND several moderately hot
    partitions, which a fixed threshold tends to miss.
    """

    def __init__(self, z=3.5):
        self.z = z

    def calculate(self, partition_sizes):
        if not partition_sizes:
            return 0

        med = median(partition_sizes)

        deviations = [
            abs(x - med)
            for x in partition_sizes
        ]

        mad = median(deviations)

        # Avoid division by zero when all sizes are identical
        if mad == 0:
            return med * 3

        threshold = med + self.z * 1.4826 * mad

        return threshold
