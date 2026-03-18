import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(
        self,
        n_clusters=8,
        random_state=None,
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init if isinstance(n_init, int) else 10
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_centroids(self, X):
        n_samples = X.shape[0]
        centroids = []

        idx = np.random.randint(n_samples)
        centroids.append(X[idx].copy())

        for _ in range(1, self.n_clusters):
            distances = cdist(X, centroids, metric="euclidean")
            min_distances = np.min(distances, axis=1)

            probabilities = min_distances**2
            probabilities /= probabilities.sum()

            idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[idx].copy())

        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        distances = cdist(X, centroids, metric="euclidean")
        return np.argmin(distances, axis=1), np.min(distances, axis=1)

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                random_idx = np.random.randint(X.shape[0])
                centroids[k] = X[random_idx].copy()
        return centroids

    def _compute_inertia(self, min_distances):
        return np.sum(min_distances**2)

    def _fit_single_run(self, X):
        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            labels, min_distances = self._assign_clusters(X, centroids)
            inertia = self._compute_inertia(min_distances)

            new_centroids = self._update_centroids(X, labels)

            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if centroid_shift < self.tol:
                if self.verbose > 0:
                    print(f"Converged at iteration {iteration}")
                break

        labels, min_distances = self._assign_clusters(X, centroids)
        inertia = self._compute_inertia(min_distances)

        return centroids, labels, inertia, iteration + 1

    def fit(self, X, y=None):
        X = np.asarray(X)

        best_centroids = None
        best_labels = None
        best_inertia = np.inf
        best_n_iter = None

        for run in range(self.n_init):
            if self.verbose > 0:
                print(f"Run {run + 1}/{self.n_init}")

            centroids, labels, inertia, n_iter = self._fit_single_run(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X, self.cluster_centers_)
