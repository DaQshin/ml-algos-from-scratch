import numpy as np

class MultinomialClassifier:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _cross_entropy_loss__(self, y_true, y_pred):
        m = len(y_true)
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    
    def _update_weights(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        dw = np.dot(X.T, (y_pred - y)) / m
        db = np.sum((y_pred - y), axis=0) / m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        num_features = X.shape[1]
        num_classes = y.shape[1]
        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        cost = []
        for _ in range(self.epochs):
            epoch_cost = 0
            for i in range(0, len(y), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                self._update_weights(X_batch, y_batch)
                epoch_cost += self._cross_entropy_loss__(y_batch, self.predict(X_batch))
                epoch_cost /= len(y) / self.batch_size
            cost.append(epoch_cost)
        return cost
    
    def predict(self, X):
        return self._softmax(np.dot(X, self.weights) + self.bias)
    
    def predict_classes(self, X):
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)
    


#main
if __name__ == '__main__':
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    # Generate a synthetic dataset for multinomial classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=3, random_state=42)
   

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialClassifier(learning_rate=0.01, epochs=32, batch_size=32)
    model.fit(X_train, y_train)
    y_pred = model.predict_classes(X_test)
    y_test_classes = np.argmax(y_test, axis=1)
    print('Confusion Matrix:\n', confusion_matrix(y_test_classes, y_pred))
    print('Accuracy:', accuracy_score(y_test_classes, y_pred))
    print('Classification Report:\n', classification_report(y_test_classes, y_pred))

    
    # Plotting the loss curve
    plt.plot(model.fit(X_train, y_train))
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # # Plotting the decision boundary (for 2D data)
    # if X.shape[1] == 2:
    #     plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolor='k')
    #     plt.title('Decision Boundary')
    #     plt.xlabel('Feature 1')
    #     plt.ylabel('Feature 2')
    #     plt.show()
    
    # # Using sklearn's LogisticRegression for comparison
    # model2 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # model2.fit(X_train, y_train.argmax(axis=1))
    # y_pred2 = model2.predict(X_test)
    # print('\n\nUsing sklearn Logistic Regression')
    # print('Confusion Matrix:\n', confusion_matrix(y_test_classes, y_pred2))
    # print('Accuracy:', accuracy_score(y_test_classes, y_pred2))
    # print('Classification Report:\n', classification_report(y_test_classes, y_pred2))

   
