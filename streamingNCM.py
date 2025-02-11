from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors

import copy

class StreamingNCM:
    def __init__(self, subsequence_length, neighborhood_size, distance_metric='euclidean', buffer_size=1000, calibration_size=100):
        """
        Initialize the Streaming NCM.

        :param subsequence_length: Length of subsequences (w).
        :param neighborhood_size: Neighborhood size (k).
        :param distance_metric: Metric for calculating distance (default is Euclidean).
        :param buffer_size: Maximum number of subsequences to store for local density estimation.
        :param calibration_size: Number of recent nonconformity scores to store for calibration.
        """
        self.w = subsequence_length
        self.k = neighborhood_size
        self.distance_metric = distance_metric
        self.buffer_size = buffer_size
        self.calibration_size = calibration_size

        # Rolling buffers
        self.subsequence_buffer = deque(maxlen=buffer_size)  # Stores recent subsequences
        self.training_subsequences = deque(maxlen=buffer_size)  # Stores recent subsequences
        self.calibration_buffer = deque(maxlen=calibration_size)  # Stores only normal subsequences
        self.calibration_scores = deque(maxlen=calibration_size)  # Stores nonconformity scores for calibration
        self.neighbor_densities = None  # Store precomputed neighbor densities

    def update(self, new_point, update_densities=False):
        """
        Update the subsequence buffer with a new data point.

        :param new_point: A new data point (e.g., 2D coordinate [x, y]).
        """
        new_point = np.atleast_1d(new_point)
        
        if not update_densities and (self.neighbor_densities is None):
            self.update_neighbor_densities()

        if len(self.subsequence_buffer) > 0:
            last_subsequence = self.subsequence_buffer[-1]
            new_subsequence = np.append(last_subsequence[1:], new_point)
        else:
            new_subsequence = np.full(self.w, new_point)  # Fill with duplicates initially

        self.subsequence_buffer.append(new_subsequence)
        if update_densities:
            self.training_subsequences.append(new_subsequence)

    def fit(self, points):
        for point in points:
            self.update(point, True)
        self.update_neighbor_densities()

    def retrain(self):
        self.training_subsequences = copy.deepcopy(self.subsequence_buffer)
        self.update_neighbor_densities()

    def update_neighbor_densities(self):
        """
        Recompute neighbor densities using KDTree, instead of cdist which is O(N^2) memory-wise
        """
        if len(self.training_subsequences) < self.k:
            self.neighbor_densities = None
            return
    
        subsequences = np.array(self.training_subsequences)
        #subsequences = np.array(self.training_subsequences, dtype=np.float32) #32bit precision rather than 64 to half the used memory
    
        # Use KDTree since only one feature and euclidian distances (if non-euclidian metric is used the switch to KDTree, also if >20 features.
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree', metric=self.distance_metric).fit(subsequences)
        distances, indices = nbrs.kneighbors(subsequences)
    
        neighbor_densities = []
        for i in range(len(subsequences)):
            threshold_distance = distances[i, -1]
            reachability_distances = np.maximum(distances[i], threshold_distance)
            neighbor_density = self.k / np.sum(reachability_distances)
            neighbor_densities.append(neighbor_density)
    
        self.neighbor_densities = np.array(neighbor_densities)

    def calculate_lof(self, test_subsequence, epsilon=1e-8):
        """
        Compute LOF dynamically for a test subsequence.

        :param test_subsequence: The subsequence to evaluate.
        :return: Nonconformity score (LOF).
        """
        if self.neighbor_densities is None or len(self.training_subsequences) < self.k:
            return np.inf  # Not enough data

        subsequences = np.array(self.training_subsequences)
        distances = cdist([test_subsequence.flatten()], subsequences.reshape(len(subsequences), -1), metric=self.distance_metric).flatten()

        threshold_distance = np.partition(distances, self.k - 1)[self.k - 1]
        neighbors = np.where(distances <= threshold_distance)[0]

        if len(neighbors) == 0:
            return np.inf  # No valid neighbors, should never happen

        reachability_distances = np.maximum(distances[neighbors], threshold_distance)
        test_density = len(neighbors) / (np.sum(reachability_distances) + epsilon)

        # Use precomputed neighbor densities
        lof = np.mean(self.neighbor_densities[neighbors] / test_density)
        return lof

    def update_calibration(self, score, subsequence):
        """
        Update the calibration buffer with a new nonconformity score and corresponding normal subsequence.

        :param score: Nonconformity score from a normal (assumed non-anomalous) subsequence.
        :param subsequence: The corresponding subsequence.
        """

        #if len(self.calibration_buffer) >= self.calibration_size:
        #    # Move the oldest calibration sequence into the subsequence buffer
        #    self.subsequence_buffer.append(self.calibration_buffer.popleft())

        self.calibration_scores.append(score)
        self.calibration_buffer.append(subsequence)  # Store the normal subsequence

    def compute_p_value(self, test_score, smoothed=True):
        """
        Compute the p-value of a new subsequence based on the stored calibration scores.

        :param test_score: Nonconformity score of the test subsequence.
        :return: The p-value (proportion of calibration scores greater than test_score).
        """

        if len(self.calibration_scores) == 0:
            return 1.0  # Default high p-value if no calibration data

        if smoothed:
            n = len(np.array(self.calibration_scores))+1
            tau = np.random.uniform(0, 1)
            return (np.sum(np.array(self.calibration_scores) > test_score) + tau*np.sum(np.array(self.calibration_scores) == test_score))/n
        else:
            return np.mean(np.array(self.calibration_scores) >= test_score)



if __name__ == "__main__":

    streaming_ncm = StreamingNCM(subsequence_length=3, neighborhood_size=5, buffer_size=500, calibration_size=10)

    # Simulated streaming data
    stream_data = [
       [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [2000, 2500], [20, 20], [6, 6], [1, 1]
    ]

    for point in stream_data:
        test_subsequence = streaming_ncm.update(point)
        test_score = streaming_ncm.calculate_lof(test_subsequence)

        if test_score is not None:
            streaming_ncm.update_calibration(test_score, test_subsequence)
            p_value = streaming_ncm.compute_p_value(test_score)
            is_anomalous = p_value < 0.05  # Anomaly threshold

            print(f"Point: {point}, Anomalous? {is_anomalous}, P-value: {p_value}")
