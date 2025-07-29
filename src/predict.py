from joblib import load
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def predict_quantized(X, params):
    """Robust prediction from quantized parameters"""
    weights = params['coef'] * params['coef_scale'] + params['coef_scale'] * params['coef_zero']
    bias = params['intercept'] * params['intercept_scale']
    return X @ weights + bias

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Original model
    orig_model = load('linear_regression.joblib')
    orig_preds = orig_model.predict(X)
    
    # Quantized predictions
    params = load('quant_params.joblib')
    quant_preds = predict_quantized(X, params)
    
    # Ensure 1D arrays
    orig_preds = orig_preds.flatten()
    quant_preds = quant_preds.flatten()
    
    # Metrics
    print("\nQUANTIZATION METRICS")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Original':<15} | {'Quantized':<15}")
    print("-" * 60)
    print(f"{'R² Score':<20} | {r2_score(y, orig_preds):<15.6f} | {r2_score(y, quant_preds):<15.6f}")
    print(f"{'RMSE':<20} | {np.sqrt(mean_squared_error(y, orig_preds)):<15.6f} | {np.sqrt(mean_squared_error(y, quant_preds)):<15.6f}")
    print(f"{'First Pred':<20} | {orig_preds[0]:<15.6f} | {quant_preds[0]:<15.6f}")
    print("-" * 60)
    print(f"Accuracy Preservation: {r2_score(y, quant_preds)/r2_score(y, orig_preds):.2%}")

if __name__ == "__main__":
    main()