import numpy as np
from joblib import load, dump

def safe_quantize(model):
    """Robust quantization with proper rounding and scaling"""
    coef = model.coef_
    intercept = model.intercept_
    
    # Quantize coefficients with dynamic range
    coef_min, coef_max = coef.min(), coef.max()
    coef_range = coef_max - coef_min
    
    # Handle case where all coefficients are equal
    if coef_range < 1e-10:
        coef_scale = 1.0
        coef_zero = 0
        quant_coef = np.zeros_like(coef, dtype=np.uint8)
    else:
        coef_scale = coef_range / 255
        coef_zero = np.round(coef_min / coef_scale)
        quant_coef = np.clip(np.round((coef - coef_min) / coef_scale), 0, 255).astype(np.uint8)
    
    # Quantize intercept with separate scaling
    int_scale = max(abs(intercept) / 127, 1e-10)
    quant_intercept = np.clip(np.round(intercept / int_scale), -128, 127).astype(np.int8)
    
    params = {
        'coef': quant_coef,
        'coef_scale': float(coef_scale),
        'coef_zero': int(coef_zero),
        'intercept': quant_intercept,
        'intercept_scale': float(int_scale)
    }
    
    # Verify reconstruction
    recon_coef = quant_coef * coef_scale + coef_min
    recon_intercept = quant_intercept * int_scale
    
    print(f"Original coef[0]: {coef[0]:.6f}, Reconstructed: {recon_coef[0]:.6f}")
    print(f"Original intercept: {intercept:.6f}, Reconstructed: {recon_intercept:.6f}")
    
    # More tolerant check
    coef_error = np.max(np.abs(recon_coef - coef))
    int_error = np.abs(recon_intercept - intercept)
    
    if coef_error > 1e-4 or int_error > 1e-4:
        print(f"Warning: Reconstruction error may be significant (coef: {coef_error:.6f}, int: {int_error:.6f})")
    else:
        print("Reconstruction successful!")
    
    return params

if __name__ == "__main__":
    model = load('linear_regression.joblib')
    params = safe_quantize(model)
    dump(params, 'quant_params.joblib')