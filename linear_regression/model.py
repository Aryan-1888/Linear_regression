import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            # gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - Y))
            db = (2/n_samples) * np.sum(y_pred - Y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                loss = np.mean((Y - y_pred) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")