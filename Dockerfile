# SIMPLIFIED Dockerfile that always works:
FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Start both services
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0
