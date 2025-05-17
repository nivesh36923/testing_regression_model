
# regression_model.py

from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Generate synthetic data
X = np.array([i for i in range(1000)]).reshape(-1, 1)
Y = np.array([3 * i + 2 for i in range(1000)])

# Train the regression model
model = LinearRegression()
model.fit(X, Y)

# Save the model
with open("regression_model.pkl", 'wb') as f:
    pickle.dump(model, f)

print("Model saved as regression_model.pkl")
