# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports
EXPOSE 8000 8501

# Start both services
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
