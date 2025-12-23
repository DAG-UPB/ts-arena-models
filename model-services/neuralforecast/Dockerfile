FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	MODELS_DIR=/models

WORKDIR /app

# Minimal system dependencies (helps with some wheels)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Prepare persistence directory (mounted via volume)
RUN mkdir -p ${MODELS_DIR}

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]