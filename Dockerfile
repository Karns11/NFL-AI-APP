FROM python:3.11-slim

WORKDIR /app

# System deps required by lightgbm and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and required static files
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create data dirs (will be overridden by volume mount in production)
RUN mkdir -p data/db data/reports data/raw

EXPOSE 8080

CMD ["python", "app.py", "--no-browser", "--port", "8080"]
