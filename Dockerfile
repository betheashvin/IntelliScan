FROM python:3.8-slim

WORKDIR /app
COPY . .

# Install system dependencies INCLUDING build tools
RUN apt-get update && apt-get install -y git gcc g++ build-essential && rm -rf /var/lib/apt/lists/*

# Install older PyTorch that doesn't need blis compilation
RUN pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other packages
RUN pip install --no-cache-dir -r requirements.txt

# Start services
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
