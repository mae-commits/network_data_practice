FROM python:3.11-slim
USER root

# Set working directory
WORKDIR /workdir

# Install system dependencies
RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get install software-properties-common && \
    pip install --upgrade pip

RUN sed -i '/cudf/d' /workdir/requirements.txt && \
    pip install -r /workdir/requirements.txt

RUN pip install ipykernel jupyter

# Keep container running
CMD ["sleep", "infinity"]