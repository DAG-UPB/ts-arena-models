FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# torch MUST come from the CUDA 12.6 index (models #7). The default PyPI wheel for
# 2.13.0 is +cu130, built against CUDA 13, which needs driver >= 580. proteus runs
# 535.183.01 (CUDA 12.2), so the cu130 build fails CUDA init with nothing but a
# UserWarning, torch.cuda.is_available() returns False, TabPFN's device='auto'
# resolves to CPU, and inference takes 173-297s instead of seconds.
# CUDA 12.x minor-version compatibility makes the cu126 build run on a 12.2 driver;
# toto (2.7.0+cu126) and timesfm-3.0 (2.7.1+cu126) already prove that on this host.
# Same version, different CUDA build -- requirements.txt's torch==2.13.0 is satisfied
# by 2.13.0+cu126, so the pinned set below is unchanged.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV TABPFN_DISABLE_TELEMETRY=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
