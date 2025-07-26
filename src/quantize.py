import numpy as np
import torch
import torch.nn as nn
from joblib import load, dump

# Load sklearn model
sklearn_model = load('linear_regression.joblib')
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_

# Save unquantized parameters
unquant_params = {
    'coef': coef,
    'intercept': intercept
}
dump(unquant_params, 'unquant_params.joblib')

# Quantization function
def quantize(x, scale, zero_point, dtype=np.uint8):
    return np.clip(np.round(x/scale + zero_point), np.iinfo(dtype).min, np.iinfo(dtype).max)

# Calculate scale and zero point
rmin, rmax = min(coef.min(), intercept.min()), max(coef.max(), intercept.max())
qmin, qmax = 0, 255  # uint8 range
scale = (rmax - rmin) / (qmax - qmin)
zero_point = qmin - rmin / scale

# Quantize parameters
quant_coef = quantize(coef, scale, zero_point)
quant_intercept = quantize(intercept, scale, zero_point)

# Save quantized parameters
quant_params = {
    'coef': quant_coef,
    'intercept': quant_intercept,
    'scale': scale,
    'zero_point': zero_point
}
dump(quant_params, 'quant_params.joblib')

# Create PyTorch model with dequantized weights
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        
        # Dequantize and set weights
        dequant_coef = (quant_coef.astype(np.float32) - zero_point) * scale
        dequant_intercept = (quant_intercept.astype(np.float32) - zero_point) * scale
        
        self.linear.weight.data = torch.from_numpy(dequant_coef).unsqueeze(0)
        self.linear.bias.data = torch.from_numpy(dequant_intercept)

    def forward(self, x):
        return self.linear(x)

# Test inference
model = LinearModel()
test_input = torch.randn(1, 8)
print("Test output:", model(test_input))