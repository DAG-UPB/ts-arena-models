# Devel image: the flashrnn sLSTM CUDA kernels used by TiRex-2 are JIT-compiled
# at runtime and need nvcc. Torch 2.9.1+cu128 matches the tirex-2 tested setup.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Drop the CUDA forward-compatibility libs: they only support datacenter GPUs
# (error 804 on TITAN RTX). Without them the host driver is used directly via
# CUDA minor-version compatibility.
RUN rm -rf /usr/local/cuda/compat && ldconfig

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
