import numpy as np

class LinearRegressiond:
    def __init__(self, learning_rate = 0.01, epochs = 100, batch_size = 1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def __compute_cost__(self, X, y):
        m = len(y)
        return np.mean((self.predict(X) - y) ** 2) / (2*m)
    
    def __update_weights__(self, X, y):
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
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                self.__update_weights__(X_batch, y_batch)
                epoch_cost += self.__compute_cost__(X_batch, y_batch)
            cost.append(epoch_cost)
        return cost
#main
# if __name__ == '__main__':
#     import pandas as pd
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import mean_squared_error, r2_score

#     data = pd.read_csv('./height-weight.csv')
#     X = data[['Height(Inches)']].values
#     y = data['Weight(Pounds)'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)


#     from sklearn.linear_model import LinearRegression
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     print('Mean Squared Error:', mean_squared_error(y_test, model.predict(X_test)))
#     print('R2 Score:', r2_score(y_test, model.predict(X_test)))
#     import matplotlib.pyplot as plt
#     plt.scatter(X, y)
#     plt.plot(X, model.predict(X), color = 'red')
#     plt.show()

#     model = LinearRegressiond(learning_rate = 0.01, epochs = 10, batch_size = 1)
#     cost = model.fit(X_train, y_train)
#     print('Mean Squared Error:', mean_squared_error(y_test, model.predict(X_test)))
#     print('R2 Score:', r2_score(y_test, model.predict(X_test)))
#     import matplotlib.pyplot as plt
#     plt.scatter(X, y)
#     plt.plot(X, model.predict(X), color = 'red')
#     plt.show()
