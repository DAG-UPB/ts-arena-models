FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV TTM_R1_MODEL_PATH=ibm-granite/granite-timeseries-ttm-r1
ENV TTM_R1_REVISION=main
ENV TTM_R1_CONTEXT_LENGTH=512
ENV TTM_R1_PREDICTION_LENGTH=96

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]