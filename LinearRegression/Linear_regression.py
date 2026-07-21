import numpy as np

class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0

        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            predictions = X @ self.weights + self.bias

            error = predictions - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.weights + self.bias

    def score(self, X, y):
        predictions = self.predict(X)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegressionGD(
    learning_rate=0.01,
    epochs=5000
)

model.fit(X, y)

print(model.weights)
print(model.bias)
print(model.predict([[6]]))
print(model.score(X, y))