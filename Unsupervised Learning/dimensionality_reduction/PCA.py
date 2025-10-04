import numpy as np
class PCA:
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.explained_variance_ = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def _center_data(self, data):
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        centered_data = (data - mean) / std
        return centered_data
    
    def get_covariance_matrix(self, data):
        cov = (data.T @ data) / (data.shape[0] - 1)
        return cov
    
    def get_components(self, covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx[:self.n_components]]
        self.eigenvectors = eigenvectors[:, idx[:self.n_components]]
    
    def fit(self, data):
        c_data = self._center_data(data)
        cov_matrix = self.get_covariance_matrix(c_data)
        self.get_components(cov_matrix)
        self.explained_variance_ = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)

    def fit_transform(self, data):
        self.fit(data)
        data_proj = data @ self.eigenvectors
        return data_proj

from sklearn.datasets import load_breast_cancer
import pandas as pd
data_raw = load_breast_cancer()
target  = np.array(data_raw.target).reshape(-1, 1)
data = pd.DataFrame(data = data_raw.data, columns= data_raw.feature_names)
pca = PCA(n_components=3)
data = pca.fit_transform(data)
data = np.concatenate([data, target], axis=1)
print(data.shape)
print(pca.explained_variance_)
print(pca.eigenvalues)