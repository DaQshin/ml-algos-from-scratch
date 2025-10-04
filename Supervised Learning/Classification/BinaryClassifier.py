import numpy as np

class BinaryClassifier:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def __binary_cross_entropy_loss__(self, y_true, y_pred):
        m = len(y_true)
        return np.mean((-y_true * np.log(y_pred + 1e-8)) - ((1 - y_true) * np.log(1 - y_pred + 1e-8))) / m
    
    def __update_weights(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        dw = np.dot(X.T, (y_pred - y)) / m
        db = np.sum((y_pred - y)) / m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0
        cost = []
        for _ in range(self.epochs):
            epoch_cost = 0
            for i in range(0, len(y), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                self.__update_weights(X_batch, y_batch)
                epoch_cost += self.__binary_cross_entropy_loss__(y_batch, self.predict(X_batch))
            cost.append(epoch_cost)
        
        return cost
    
#main
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset for binary classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    model1 = BinaryClassifier(learning_rate=0.01, epochs=100, batch_size=32)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)

    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    # Plotting the cost function
    plt.plot(model1.fit(X_train, y_train))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.show()

    print('Using sklearn Logistic Regression')
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred2))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred2))
    print('Classification Report:\n', classification_report(y_test, y_pred2))
   