FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install only git (no gcc needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch FIRST (CPU-only, pre-compiled)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install other packages
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
