FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# torch MUST come from the CUDA 12.6 index: the default PyPI wheel is built against
# CUDA 13, which proteus' driver 535 cannot init, so the model would silently run on
# CPU (models-24). requirements.txt's torch==2.13.0 is satisfied by 2.13.0+cu126.
RUN pip install --no-cache-dir --upgrade "pip>=25"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
