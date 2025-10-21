FROM python:3.11-slim

# Install system build deps needed to build blis, thinc, spacy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      libffi-dev \
      libssl-dev \
      ca-certificates \
      curl \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools so pip can pick wheels when available
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
