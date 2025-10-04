import numpy as np 


class LogisticRegression:

    
    def __init__(self, epochs=1000, solver='newton'):
        self.epochs = epochs
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))
    
    def _binary_cross_entropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        cost = []

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            cost.append(self._binary_cross_entropy(y, y_pred))
            m = len(y)
            loss_gradient = np.dot(X.T, (y_pred - y)) / m
            hessian = -X.T @ np.diag(y_pred * (1 - y_pred)) @ X / m
            hessian_inv = np.linalg.inv(hessian)
            self.weights -= hessian_inv @ loss_gradient
        
        return cost
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights))
    
#main
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset for binary classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(epochs=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
