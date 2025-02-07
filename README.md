# StreamingCAD
repository for streaming Conformal Anomaly Detection. Adapted from Laxhammars research [https://link.springer.com/article/10.1007/s10472-013-9381-7)].

# How to use

```python
from streamingNCM import StreamingNCM
from streamingCAD import StreamingICAD
import numpy as np

# create our non-conformity measure. This implementation is based on Laxhammars sslo-ncm.
# We need to set the subsequence length, the number of neighbors, and the buffer sizes. In this case, they're set for the following example.
streaming_ncm = StreamingNCM(subsequence_length=4, neighborhood_size=5, buffer_size=500, calibration_size=15)

# create our ICAD instance. We need to pass along the NCM, the minimum number of training instances we want, and the threshold. In this case the threshold indicates that we want a 95% certainty (1-0.95).
streaming_icad = StreamingICAD(streaming_ncm, min_train=10, anomaly_threshold=0.05)

# Declare our data for the testing purpose.
stream_data = [
       [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
       [5, 5], [6, 6], [7, 7], [8, 8], [2000, 2500], [20, 20], [6, 6], [1, 1], [1, 1], [2, 2], [3, 3], [4, 4]
    ]

# We then iterate over the data points one by one, i.e. we're streaming the data.

for point in stream_data:
        result = streaming_icad.process_stream(point)
        # As long as the model is training or calibrating, this function returns none.
        if result is None:
            pass
            #print(f"Point: {point} → Still calibrating...")
        else:
            is_anomalous, p_value = result
            print(f"Point: {point}, Test score? {is_anomalous}, P-value: {p_value}")
```

Process_stream has an argument `return_score=True` that, by default, is set to true. This returns the test score for the point. If set to false, it will return a bool indicating whether the test point is an anomaly.

A p-value below the threshold indicates an anomaly.
