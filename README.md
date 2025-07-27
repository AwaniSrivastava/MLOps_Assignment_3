# MLOps_Assignment_3
# MLOps Assignment 3: End-to-End MLOps Pipeline

## Project Overview
This repository implements a complete MLOps pipeline for the California Housing price prediction model, including model training, Docker containerization, CI/CD with GitHub Actions, and manual quantization.

## Quantization Results

| Metric               | Original Sklearn Model | Quantized Model | Difference |
|----------------------|------------------------|-----------------|------------|
| **R² Score**         | 0.605329               | 0.603617        | -0.002712  |
| **RMSE**             | 0.724931               | 0.726501        | +0.001570  |
| **Model Size (KB)**  | 0.399                  | 0.462           | +0.063     |
| **First Prediction** | 4.151943               | 4.196165        | +0.044222  |

### Key Observations:
- Quantized model maintains **99.72%** of original accuracy
- Minimal performance degradation (-0.28% R² score difference)
- Quantized parameters file is slightly larger (0.462 KB vs 0.399 KB)
- Reconstruction error in quantization: 0.000704 for coefficients

## File Size Comparison
| File Name               | Size (KB) | Description                     |
|-------------------------|-----------|---------------------------------|
| `linear_regression.joblib` | 0.681     | Original sklearn model          |
| `unquant_params.joblib`   | 0.399     | Original model parameters       |
| `quant_params.joblib`     | 0.462     | Quantized model parameters      |

## How to Reproduce Results

1. **Train the model**:
   ```bash
   python src/train.py
2. **Quantize the model:**:
   python src/quantize.py
   **Output:**
   Original coef[0]: 0.448675, Reconstructed: 0.449107
   Original intercept: -37.023278, Reconstructed: -37.023278
   Warning: Reconstruction error may be significant (coef: 0.000704, int: 0.000000)
3. **Evaluate models:**:
    python src/predict.py
    **Output:**
    QUANTIZATION METRICS
------------------------------------------------------------
Metric               | Original        | Quantized
------------------------------------------------------------
R² Score             | 0.605329        | 0.603617
RMSE                 | 0.724931        | 0.726501
First Pred           | 4.151943        | 4.196165
------------------------------------------------------------
Accuracy Preservation: 99.72%
**Docker Integration**
docker build -t awani86/mlops-model:latest .
docker run awani86/mlops-model:latest
