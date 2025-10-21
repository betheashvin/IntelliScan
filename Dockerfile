FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install only git (no gcc/build tools)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (pre-built, no compilation)
RUN pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
