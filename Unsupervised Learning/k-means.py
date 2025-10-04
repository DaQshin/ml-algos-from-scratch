import numpy as np

class KMeans:

    def __init__(self, n_classes):
        self.n_classes=n_classes
        self.clusters={}

    def distance(self, x, y):
        return np.linalg.norm(x - y) ** 2
    
    def initilize_clusters(self, X):
        for idx in range(self.n_classes):
            center = np.random.random((X.shape[1], ))
            cluster = {
                "center": center,
                "points": []
            }

            self.clusters[idx] = cluster

    def assign_clusters(self, X):
        for idx in range(X.shape[0]):
            dist = []
            current_data = X[idx]

            for j in range(self.n_classes):
                center = self.clusters[j]['center']
                dist_x = self.distance(center, current_data)
                dist.append(dist_x)
            current_cluster = np.argmin(dist)
            self.clusters[current_cluster]['points'].append(current_data)

    def update_clusters(self):
        for idx in range(self.n_classes):
            cluster = self.clusters[idx]
            cluster_points = cluster['points']
            if len(cluster_points) > 0:
                new_center = np.mean(cluster_points, axis=0)
                cluster['center'] = new_center
                cluster['points'] = []

    def distortion_func(self, X):
        loss = 0
        for cluster in self.clusters.values():
            center = cluster['center']
            for point in cluster['points']:
                loss += self.distance(point, center)

        return loss
            
    def fit(self, X, epochs=100):
        self.initilize_clusters(X)

        loss = []

        for _ in range(epochs):
            self.assign_clusters(X)
            self.update_clusters()
            loss.append(self.distortion_func(X))

        return loss

    def predict(self, X):
        preds = []
        for idx in range(X.shape[0]):
            dist = []
            current_data = X[idx]
            for c in range(self.n_classes):
                cluster_center = self.clusters[c]['center']
                dist.append(self.distance(cluster_center, current_data))
            preds.append(np.argmin(dist))

        return np.array(preds)
    
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    import matplotlib.pyplot as plt
    X, y = make_blobs(n_samples= 100, n_features= 2, centers = 3, random_state=42)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KMeans(n_classes=4)
    loss = model.fit(X_train)

    y_pred = model.predict(X_test)
    print(adjusted_rand_score(y_test, y_pred))
    print(silhouette_score(X_test, y_pred))

    plt.scatter(range(100), loss)
    plt.xlabel("epochs")
    plt.ylabel('loss')
    plt.show()


