FROM python:3.10-slim

WORKDIR /app

# PyCaret + LightGBM + XGBoost need these system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Render internal port
EXPOSE 10000

# Start Flask via Gunicorn
CMD ["gunicorn","--bind","0.0.0.0:10000","app:app"]