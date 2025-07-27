FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/predict.py .
COPY src/quantize.py .
COPY linear_regression.joblib .

CMD ["python", "predict.py"]