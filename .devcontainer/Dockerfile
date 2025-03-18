FROM python:3.11-slim
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip

# requirements.txtがある場合、そのファイルをコンテナ内にコピーしてからインストール
COPY requirements.txt /workdir/requirements.txt

RUN sed -i '/cudf/d' /workdir/requirements.txt && \
    pip install -r /workdir/requirements.txt

RUN pip install ipykernel jupyter

# Keep container running
CMD ["sleep", "infinity"]