FROM python:3.12.4-slim-bullseye

WORKDIR /app

# Copy only requirements.txt and install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY linear_regression.joblib .
COPY src/predict.py .

CMD ["python", "predict.py"]