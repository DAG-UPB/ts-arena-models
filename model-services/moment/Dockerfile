FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# torch 2.13 on CUDA routes some degenerate matmul shapes (e.g. a context of a single patch,
# as in the /health probe) to a Triton kernel. Triton builds a small C helper on first use,
# so the image needs a C compiler; without one those calls fail and /health returns 503.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# torch MUST come from the CUDA 12.6 index. The default PyPI wheel for 2.13.0 is +cu130,
# built against CUDA 13, which needs driver >= 580. proteus runs 535 (CUDA 12.2), so the
# cu130 build fails CUDA init with only a UserWarning, torch.cuda.is_available() returns
# False and the model silently runs on CPU. Same version, different CUDA build:
# requirements.txt's torch==2.13.0 is satisfied by 2.13.0+cu126, so the pinned set is unchanged.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]