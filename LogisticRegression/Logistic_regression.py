import numpy as np
import pandas as pd

data = pd.DataFrame({
    "Hours_Studied": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "Attendance":    [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Pass":          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})
X = data.drop(columns=["Pass"]).values
y = data["Pass"].values


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Train the model
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            # Linear model
            linear = np.dot(X, self.weights) + self.bias

            # Predicted probabilities
            y_pred = self.sigmoid(linear)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Return probabilities
    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)

    # Return class labels
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    # Accuracy
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

model = LogisticRegression(
    learning_rate=0.1,
    epochs=1000
)

# Train
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print(predictions)

# Probabilities
probabilities = model.predict_proba(X)
print(probabilities)

# Accuracy
accuracy = model.score(X, y)
print("Accuracy:", accuracy)