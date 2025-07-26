from joblib import load
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load model and data
model = load('linear_regression.joblib')
data = fetch_california_housing()
X, y = data.data, data.target

# Make predictions
preds = model.predict(X[:5])  # Test on first 5 samples
print("Sample predictions:", preds)
print("R2 score:", r2_score(y[:5], preds))