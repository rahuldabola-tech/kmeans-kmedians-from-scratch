import numpy as np

class KMeans:
    def __init__(self, k=3, num_iter=1000, order=2):
        np.random.seed(42)
        self.k = k
        self.num_iter = num_iter
        self.centers = None
        self.cluster_idx = None

        if order in (1, 2):
            self.order = order
        else:
            raise ValueError("Order must be 1 (L1) or 2 (L2)")

    def fit(self, X):
        m, n = X.shape
        self.centers = np.zeros((self.k, n))
        self.cluster_idx = np.zeros(m)

        # Initialize centers using percentiles
        for i in range(n):
            self.centers[:, i] = np.random.uniform(
                np.percentile(X[:, i], 10),
                np.percentile(X[:, i], 90),
                self.k
            )

        for i in range(self.num_iter):
            distances = np.linalg.norm(
                X[:, np.newaxis] - self.centers,
                axis=2,
                ord=self.order
            )
            cluster_idx = np.argmin(distances, axis=1)

            new_centers = np.zeros((self.k, n))
            for idx in range(self.k):
                cluster_points = X[cluster_idx == idx]
                if len(cluster_points) == 0:
                    new_centers[idx] = self.centers[idx]
                else:
                    if self.order == 2:
                        new_centers[idx] = np.mean(cluster_points, axis=0)
                    else:
                        new_centers[idx] = np.median(cluster_points, axis=0)

            if np.array_equal(cluster_idx, self.cluster_idx):
                print(f"Early stopped at iteration {i}")
                break

            self.centers = new_centers
            self.cluster_idx = cluster_idx

        return self

    def predict(self, X):
        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centers,
            axis=2,
            ord=self.order
        )
        return np.argmin(distances, axis=1)
