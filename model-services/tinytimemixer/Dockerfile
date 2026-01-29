FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV TTM_MODEL_PATH=ibm-granite/granite-timeseries-ttm-r2
ENV TTM_CONTEXT_LENGTH=512
ENV TTM_PREDICTION_LENGTH=96

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]