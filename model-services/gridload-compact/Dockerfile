FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY weights/ ./weights/

# The nets are tiny; one intra-op thread is all they need (and what the latency numbers were measured with).
ENV OMP_NUM_THREADS=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
