FROM python:3.8-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (pre-built)
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other packages
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
