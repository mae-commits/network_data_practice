FROM python:3.11-slim
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common build-essential && \
    pip install --upgrade pip

# requirements.txtがある場合、そのファイルをコンテナ内にコピーしてからインストール
# COPY requirements.txt /workdir/requirements.txt

RUN pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html && \
    pip install torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-2.0.0+cpu.html && \
    pip install ipykernel jupyter

# Keep container running
CMD ["sleep", "infinity"]