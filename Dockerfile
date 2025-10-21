FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Start both services
CMD uvicorn main:app --host 0.0.0.0 --port 10000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
