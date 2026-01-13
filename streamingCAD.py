import numpy as np

from streamingNCM import StreamingNCM
from collections import deque


class StreamingICAD:
    def __init__(self, streaming_ncm, anomaly_threshold=0.05, min_train=100):
        """
        Initialize a Streaming ICAD (Incremental Contour Anomaly Detector).

        Parameters
        ----------
        streaming_ncm : StreamingNCM
            The underlying streaming nearest‑neighbour classifier that
            provides LOF scores and maintains a sliding buffer.
        anomaly_threshold : float, optional
            Initial target false‑positive rate (default 0.05).  The algorithm
            will try to keep the empirical FPR close to this value by
            adaptively adjusting :pyattr:`epsilon`.
        min_train : int, optional
            Minimum number of training points required before the detector
            starts producing decisions.  Defaults to 100.

        Attributes
        ----------
        ncm : StreamingNCM
            Reference to the underlying model.
        epsilon : float
            Current anomaly threshold used for decision making.
        target_fpr : float
            Desired false‑positive rate that drives the epsilon update.
        min_train : int
            Minimum training sample count.
        recent_anomaly_flags : collections.deque
            Sliding window of the most recent anomaly decisions (``True``/``False``).
        """
        self.ncm = streaming_ncm
        self.epsilon = anomaly_threshold
        self.target_epsilon = anomaly_threshold
        self.min_train = min_train
        self.recent_anomaly_flags = deque(maxlen=self.ncm.calibration_size)

    def _adjust_epsilon(self):
        """
        Adjust the anomaly threshold (``epsilon``) toward the target FPR.

        The function computes the empirical false‑positive rate from
        :pyattr:`recent_anomaly_flags` and moves ``epsilon`` 10 % of the
        way toward the maximum of the observed FPR and the desired
        ``target_fpr``.  This smoothing prevents abrupt threshold
        changes when the stream contains bursts of anomalies.
        """

        # proportion of recent points that were flagged as anomalies
        observed_anomalies = np.mean(self.recent_anomaly_flags)
        # Push ε toward the target (self.target_fpr) but smooth it
        self.epsilon = 0.9 * self.epsilon + 0.1 * max(observed_fpr, self.target_epsilon)

    def _update_quantiles(self):
        """
        Re‑compute the calibration quantiles every time the calibration
        set changes.  These quantiles are used to construct a confidence
        interval for the LOF score of a new point.

        The upper bound corresponds to the (1‑ε)‑th percentile of the
        calibration scores, while the lower bound is the ε‑th percentile.
        """
        # `self.ncm.calibration_scores` is a list of LOF scores.
        self.calibration_quantiles = {
            # 1‑ε quantile – the upper bound of the normal region
            "upper": np.quantile(self.ncm.calibration_scores, 1 - self.epsilon),
            # ε quantile – the lower bound (rarely used for LOF, but kept for symmetry)
            "lower": np.quantile(self.ncm.calibration_scores, self.epsilon),
        }

    def process_stream(self, new_point, return_score=False, return_interval=False):
        """
        Process a new data point from the stream and optionally return
        diagnostic information.

        Parameters
        ----------
        new_point : array‑like
            The incoming observation (e.g., ``[x, y]``).
        return_score : bool, optional
            If ``True``, the raw LOF score is returned as the first
            element of the output tuple instead of the boolean anomaly
            flag.  Defaults to ``False``.
        return_interval : bool, optional
            If ``True``, the method also returns a tuple
            ``(lower, upper)`` describing the confidence interval for
            the LOF score derived from the current calibration set.

        Returns
        -------
        tuple or None
            ``(is_anomalous, p_value)`` if a decision is made,
            ``(is_anomalous, p_value, (low, high))`` if
            ``return_interval`` is ``True``, and ``None`` while the
            model is still in the calibration phase.
        """
        # assert self.min_train < self.ncm.buffer_size , "The buffer size is smaller than or equal to the minimum number of training points: {self.ncm.buffer_size}<={self.min_train}."

        n_train = len(self.ncm.training_subsequences)
        if n_train < self.min_train or (n_train < self.ncm.w or n_train < self.ncm.k):
            self.ncm.update(new_point, updating_densities=True)
            return None  # Not enough data yet

        self.ncm.update(new_point)

        test_subsequence = np.array(self.ncm.subsequence_buffer[-1])
        test_score = self.ncm.calculate_lof(test_subsequence)

        # Continue calibration until np.inf is removed and calibration set is filled.
        if len(self.ncm.calibration_scores) < self.ncm.calibration_size or any(np.isinf(score) for score in self.ncm.calibration_scores):
            self.ncm.update_calibration(test_score, test_subsequence)
            return None  # Keep calibrating

        self._update_quantiles()

        p_value = self.ncm.compute_p_value(test_score)
        if p_value > self.epsilon:
            self.ncm.update_calibration(test_score, test_subsequence)

        # Store anomaly flag for recent points
        anomaly_flag = p_value < self.epsilon
        self.recent_anomaly_flags.append(anomaly_flag)

        #self._adjust_epsilon()

        if return_score:
            is_anomalous = test_score
        else:
            is_anomalous = anomaly_flag

        if return_interval:
            low = self.calibration_quantiles["lower"]
            high = self.calibration_quantiles["upper"]
            return is_anomalous, p_value, (low, high)

        return is_anomalous, p_value


# Example Usage:
if __name__ == "__main__":
    streaming_ncm = StreamingNCM(subsequence_length=10, neighborhood_size=5, buffer_size=30, calibration_size=10)
    streaming_icad = StreamingICAD(streaming_ncm, min_train=10, anomaly_threshold=0.05)

    # Simulated streaming data
    stream_data = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [2000, 2500],
        [20, 20],
        [6, 6],
        [1, 1],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]

    # stream_data = np.array(stream_data).flat

    for point in stream_data:
        result = streaming_icad.process_stream(point, return_score=True, return_interval=True)

        if result is None:
            pass
            # print(f"Point: {point} → Still calibrating...")
        else:
            is_anomalous, p_value, (low, high) = result
            print(f"Point: {point}, Anomalous? {is_anomalous}, P-value: {p_value}, confidence‑region LOF∈[{low:.2f}, {high:.2f}]")

    # First phase: Calibration (assuming normal data)
    # for point in stream_data[:20]:
    #    streaming_icad.process_stream(point, is_calibration=True)

    # print(streaming_ncm.calibration_scores)
    # Second phase: Anomaly detection
    # for point in stream_data[20:]:
    #    result = streaming_icad.process_stream(point)
    #    if result:
    #        is_anomalous, p_value = result
    #        print(f"Point: {point}, Anomalous? {is_anomalous}, P-value: {p_value}")
