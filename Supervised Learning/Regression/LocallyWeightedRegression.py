import numpy as np

class LWR:

    def __init__(self, tau = 0.1):
        self.tau = tau

    def __gaussian_kernel__(self, x, query):
        return np.exp(-np.sum((x - query) ** 2) / (2 * self.tau ** 2))
    
    def __compute_weights__(self, X, y, query):
        m = len(X)
        W = np.zeros((m, m))

        X = np.c_[np.ones((m, 1)), X]

        for i in range(m):
            W[i, i] = self.__gaussian_kernel__(X[i], query)
        
        XTWX = X.T @ W @ X
        XTWy = X.T @ W @ y

        return np.linalg.solve(XTWX, XTWy)      
    
    def predict(self, X, y, query):
        weights = self.__compute_weights__(X, y, query)
        query = np.r_[1, query]
        return np.dot(query, weights)
    
#main
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv('./height-weight.csv')
    X = data[['Height(Inches)']].values
    y = data['Weight(Pounds)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LWR(tau = 0.5)
    y_pred = []
    for i in range(32):
        y_pred.append(model.predict(X_train[:32], y_train[:32], X_test[i]))
    print('Mean Squared Error:', mean_squared_error(y_test[:32], y_pred))
    print('R2 Score:', r2_score(y_test[:32], y_pred))
    