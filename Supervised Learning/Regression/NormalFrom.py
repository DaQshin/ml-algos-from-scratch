import numpy as np

class NormalFrom:

    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):

        X = np.c_[np.ones((X.shape[0], 1)), X]

        if np.linalg.cond(X.T @ X) > 1e10:
            raise Exception('Matrix is singular')
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return np.dot(X, self.weights) 
    

#main
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression

    data = pd.read_csv('./height-weight.csv')
    X = data[['Height(Inches)']].values
    y = data['Weight(Pounds)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NormalFrom()
    model.fit(X_train, y_train)
    print('Mean Squared Error:', mean_squared_error(y_test, model.predict(X_test)))
    print('R2 Score:', r2_score(y_test, model.predict(X_test)))
    
    import matplotlib.pyplot as plt
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, model.predict(X_test), color = 'blue')
    plt.title('Height vs Weight')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()

    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Mean Squared Error:', mean_squared_error(y_test, model.predict(X_test)))
    print('R2 Score:', r2_score(y_test, model.predict(X_test)))
    