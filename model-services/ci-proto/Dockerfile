FROM python:3.13-slim

WORKDIR /app

# On Ampere or newer GPUs, the package compiles its fused kernels on first use, which
# needs a C and a C++ compiler. The slim image has neither.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch MUST come from the CUDA 12.6 index: the default PyPI wheel is built against
# CUDA 13, which proteus' driver 535 cannot init, so the model would silently run on
# CPU (models-24). continuousintelligence pins the CUDA 13 compiler for its fused
# kernels. From 2.11 on, torch's CUDA 12 builds pin cuda-toolkit 12, so 2.10 is the
# newest torch that installs alongside the package. requirements.txt's torch==2.10.0
# is satisfied by 2.10.0+cu126.
RUN pip install --no-cache-dir --upgrade "pip>=25"
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.10.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
