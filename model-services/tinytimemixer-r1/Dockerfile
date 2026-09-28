FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# torch MUST come from the CUDA 12.6 index. The default PyPI wheel for 2.13.0 is +cu130,
# built against CUDA 13, which needs driver >= 580. proteus runs 535 (CUDA 12.2), so the
# cu130 build fails CUDA init with only a UserWarning, torch.cuda.is_available() returns
# False and the model silently runs on CPU. Same version, different CUDA build:
# requirements.txt's torch==2.13.0 is satisfied by 2.13.0+cu126, so the pinned set is unchanged.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV TTM_R1_MODEL_PATH=ibm-granite/granite-timeseries-ttm-r1
ENV TTM_R1_REVISION=main
ENV TTM_R1_CONTEXT_LENGTH=512
ENV TTM_R1_PREDICTION_LENGTH=96

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]