FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || pip install scikit-learn pandas numpy xgboost flask gunicorn joblib

# Copy application files
COPY pipeline.py .
COPY model.pkl .
COPY feature_names.json .
COPY predict.py .

# Expose port
ENV PORT=9696
EXPOSE 9696

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
