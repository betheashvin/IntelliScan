FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
