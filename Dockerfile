FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install PyTorch with its own index
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other packages from PyPI
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} & streamlit run app.py --server.port 8501 --server.address 0.0.0.0
