import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split  # added

from model import LinearRegression
from metrics import mean_squared_error, r2_score

# 🔹 Load dataset
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 🔹 Features & Target
X = df.drop('target', axis=1).values
Y = df['target'].values

# 🔥 1. Data will be shuffled during train-test split

# 🔥 2. Add LOG features
X_log = np.log1p(np.abs(X))
X = np.concatenate((X, X_log), axis=1)

# 🔥 3. Standardization
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# 🔥 4. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 🔥 5. Train model
model = LinearRegression(lr=0.01, epochs=3000)
model.fit(X_train, Y_train)

# 🔹 Predictions
y_pred = model.predict(X_test)

# 🔹 Metrics
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("\n===== FINAL OPTIMIZED RESULT =====")
print("MSE:", mse)
print("R² Score:", r2)
print("Accuracy (%):", r2 * 100)

# 🔥 6. Save graph
plt.figure()
plt.scatter(Y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Optimized Linear Regression")

plt.savefig("final_output.png")
print("Graph saved as final_output.png")
