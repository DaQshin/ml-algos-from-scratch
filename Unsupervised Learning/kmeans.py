import numpy as np
class KMeans:
    def __init__(self, n_clusters = 2, init = "random", max_iter = 100, random_state = None, algorithm = 'lloyd'):
        self.k = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.inertia_ = None
        self.clusters = {}
        if init in {'k-means++', 'random'}:
            self.init = init
        if algorithm in {'lloyd', 'elkan'}:
            self.algorithm = algorithm 
        if random_state == None:
            self.random_state = 42
    
    def _initialize_centroids(self,data):
        if self.init == "k-means++":
            centroids = [data[np.random.choice(data.shape[0])]]
            cluster = {
                    'center' : centroids[0],
                    'points' : []
                }
            self.clusters[0] = cluster
            for idx in range(1, self.k):
                distances = [
                    min(np.linalg.norm(point - centroid) ** 2 for centroid in centroids)
                    for point in data
                ]

                probabilities = distances / np.sum(distances)

                next_centroid_idx = np.random.choice(data.shape[0], p = probabilities)
                cluster = {
                    'center' : data[next_centroid_idx],
                    'points' : []
                }
                self.clusters[idx] = cluster

        elif self.init == "random":
            for idx in range(self.k):
                center = np.random.random((data.shape[1],))
                cluster = {
                    'center' : center,
                    'points' : []
                }
                self.clusters[idx] = cluster
        return self.clusters
    
    def _distance(self,p1, p2):
        return np.linalg.norm(p1 - p2) ** 2

    def _assign_clusters(self, data):
        for idx in range(data.shape[0]):
            dist = []
            curr_x = data[idx]

            for _ in range(self.k):
                center = self.clusters[_]['center']
                dis = self._distance(curr_x, center)
                dist.append(dis)
            current_cluster = np.argmin(dist)
            self.clusters[current_cluster]['points'].append(curr_x)
        return self.clusters

    def _update_clusters(self):
        for idx in range(self.k):
            points = self.clusters[idx]['points']
            if len(points) > 0:
                center = np.mean(points, axis = 0)
                self.clusters[idx]['center'] = center
                self.clusters[idx]['points'] = []
        return self.clusters

    def fit(self, data):
        self.clusters = self._initialize_centroids(data)
        for epoch in range(self.max_iter):
            self.clusters = self._assign_clusters(data)
            self.clusters = self._update_clusters()

        pred = []
        for _ in range(data.shape[0]):
            point = data[_]
            dist = []
            for i in range(self.k):
                center = self.clusters[i]['center']
                dis = self._distance(point, center)
                dist.append(dis)
            pred.append(np.argmin(dist))
            dist = []

        self.labels_ = pred
        wcss = 0
        for idx in range(self.k):
            center = self.clusters[idx]['center']
            points = self.clusters[idx]['points']
            wcss += np.sum([(point - center) ** 2 for point in points])
        self.inertia_ = wcss
    
#main
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples= 1000, n_features= 2, centers = 3)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
scaler.fit_transform(X_train)
scaler.transform(X_test)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
print(kmeans.labels_)
print(kmeans.inertia_)
# plt.scatter(range(100), loss)
# plt.xlabel("epochs")
# plt.ylabel('loss')
# plt.show()