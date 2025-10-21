FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install ALL build tools
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

# Start services
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
