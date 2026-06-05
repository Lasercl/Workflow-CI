FROM python:3.12-slim

# Install system dependencies if any, and Python packages
RUN pip install --no-cache-dir mlflow==3.12.0 pandas scikit-learn==1.8.0

# Set working directory
WORKDIR /app

# Copy model artifacts into the container
COPY model_dir /opt/ml/model

# Expose port 8080 for the MLflow model serving
EXPOSE 8080

# Run MLflow models serve on port 8080 host 0.0.0.0
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]
