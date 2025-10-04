import numpy as np

class DBSCAN:

    def __init__(self, eps, min_points = 4, metric = 'euclidean'):
        self.epsilon = eps
        self.min_points = min_points
        self.metric = metric
        self.labels = []

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def region_query(self, data, query_point_idx):
        neighbours = []
        for idx in range(len(data)):
            if self.distance(data[query_point_idx], data[idx]) <= self.epsilon:
                neighbours.append(idx)
        return neighbours
    
    def expand_cluster(self, data, point_idx, cluster_id, neighbours):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]
            if self.labels[neighbour_idx] == -1:
                self.labels[neighbour_idx] = cluster_id
            elif self.labels[neighbour_idx] == 0:
                self.labels[neighbour_idx] = cluster_id
                new_neighbours = self.region_query(data, neighbour_idx)
                if len(new_neighbours) >= self.min_points:
                    neighbours.extend(new_neighbours)
            i += 1 
    
    def fit(self, data):
        n_points = len(data)
        self.labels = np.zeros(n_points)
        cluster_id = 0

        for idx in range(n_points):
            if self.labels[idx] != 0:
                continue
        
            neighbours = self.region_query(data, idx)
            if len(neighbours) < self.min_points:
                self.labels[idx] = -1

            else:
                cluster_id += 1
                self.expand_cluster(data, idx, cluster_id, neighbours)
                
        return self.labels
#main

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=250, noise = 0.05)

# plt.scatter(X[:,0], X[:, 1])
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DBSCAN(0.3)
model.fit(X_train)
print(model.labels)


from sklearn.cluster import DBSCAN
model = DBSCAN(eps = 0.3, )
model.fit(X_train)

plt.scatter(X_train[:,0], X_train[:, 1], c = model.labels_)
plt.show()


    