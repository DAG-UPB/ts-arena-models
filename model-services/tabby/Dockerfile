FROM python:3.11-slim

WORKDIR /app

# torch MUST come from a CUDA 12.x index: the default PyPI wheel targets a newer
# CUDA than common datacenter drivers support, so the model would silently run on CPU.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
