FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libdb5.3-dev \
        libbz2-dev \
        libexpat1-dev \
        liblzma-dev \
        tk-dev \
        libffi-dev \
        curl \
        && \
    wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar -xf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.11.9*

RUN python3.11 -m ensurepip && python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Install TimesFM 3.0 from source. TimesFM 3.0 lives on the repo's master branch
# (`src/timesfm3`); the PyPI release still ships the 2.x tree, so we build from git.
# Pinned to a commit on purpose -- an unpinned clone is how a cold rebuild silently
# changes model behaviour (models #21).
ARG TIMESFM_REF=331c6d33cb1ac2611de3056d0ac7164aab6301eb
RUN git clone https://github.com/google-research/timesfm.git && \
    cd timesfm && \
    git checkout ${TIMESFM_REF} && \
    python3.11 -m pip install .[torch]

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
