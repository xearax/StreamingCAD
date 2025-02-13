from streamingNCM import StreamingNCM
import numpy as np

class StreamingICAD:
    def __init__(self, streaming_ncm, anomaly_threshold=0.05, min_train=100):
        """
        Initialize Streaming ICAD.

        :param streaming_ncm: Instance of StreamingNCM.
        :param anomaly_threshold: Threshold for anomaly detection.
        """
        self.ncm = streaming_ncm
        self.epsilon = anomaly_threshold

        #assert min_train < self.ncm.buffer_size, f"The buffer size is smaller than or equal to the minimum number of training points: {self.ncm.buffer_size}<={min_train}."

        self.min_train = min_train

    def process_stream(self, new_point, return_score=True):
        """
        Process a new data point from the stream.

        :param new_point: A new data point (e.g., [x, y]).
        :param is_calibration: Whether the point should be used for calibration.
        :return: (is_anomalous, p_value) tuple if not calibration, else None.
        """
        #assert self.min_train < self.ncm.buffer_size , "The buffer size is smaller than or equal to the minimum number of training points: {self.ncm.buffer_size}<={self.min_train}."

        n_train = len(self.ncm.training_subsequences)
        if n_train<self.min_train or (n_train < self.ncm.w or n_train < self.ncm.k):
            self.ncm.update(new_point, updating_densities=True)
            return None  # Not enough data yet
        else:
            self.ncm.update(new_point)

        test_subsequence = np.array(self.ncm.subsequence_buffer[-1])
        test_score = self.ncm.calculate_lof(test_subsequence)

        # Continue calibration until np.inf is removed and calibration set is filled.
        if len(self.ncm.calibration_scores) < self.ncm.calibration_size or any(np.isinf(score) for score in self.ncm.calibration_scores):
            self.ncm.update_calibration(test_score, test_subsequence)
            #print("🔄 Calibrating... (still contains np.inf)")
            return None  # Keep calibrating

        p_value = self.ncm.compute_p_value(test_score)
        if p_value>self.epsilon:
            self.ncm.update_calibration(test_score, test_subsequence)

        if return_score is True:
            is_anomalous = test_score
        else:
            is_anomalous = p_value < self.epsilon
        return is_anomalous, p_value

# Example Usage:
if __name__ == "__main__":

    streaming_ncm = StreamingNCM(subsequence_length=10, neighborhood_size=5, buffer_size=30, calibration_size=10)
    streaming_icad = StreamingICAD(streaming_ncm, min_train=10, anomaly_threshold=0.05)

    # Simulated streaming data
    stream_data = [
       [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [2000, 2500], [20, 20], [6, 6], [1, 1], [1, 1], [2, 2], [3, 3], [4, 4]
    ]

    #stream_data = np.array(stream_data).flat

    for point in stream_data:
        result = streaming_icad.process_stream(point)

        if result is None:
            pass
            #print(f"Point: {point} → Still calibrating...")
        else:
            is_anomalous, p_value = result
            print(f"Point: {point}, Anomalous? {is_anomalous}, P-value: {p_value}")

    # First phase: Calibration (assuming normal data)
    #for point in stream_data[:20]:
    #    streaming_icad.process_stream(point, is_calibration=True)

    #print(streaming_ncm.calibration_scores)
    # Second phase: Anomaly detection
    #for point in stream_data[20:]:
    #    result = streaming_icad.process_stream(point)
    #    if result:
    #        is_anomalous, p_value = result
    #        print(f"Point: {point}, Anomalous? {is_anomalous}, P-value: {p_value}")
