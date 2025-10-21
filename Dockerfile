FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install git AND build tools (gcc, g++, etc.)
RUN apt-get update && apt-get install -y git gcc g++ build-essential

# Install packages
RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
