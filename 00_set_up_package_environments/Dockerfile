# Start from Python base image for ARM64
FROM --platform=linux/arm64 python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment_esm.yml /tmp/environment_esm.yml

# Create conda environment and install dependencies
RUN conda env create -f /tmp/environment_esm.yml && \
    conda clean -afy

# Activate conda environment
SHELL ["conda", "run", "-n", "esmfold", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Create a test script to verify the installation
RUN echo 'import torch; print("MPS available:", torch.backends.mps.is_available())' > test_gpu.py

# Command to run Jupyter notebook
CMD ["conda", "run", "--no-capture-output", "-n", "esmfold", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 