FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    && \
    # Make python3.10 the default python
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    # Upgrade pip
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .

# Install specific PyTorch version compatible with CUDA 12.1
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install compatible CTranslate2 backend with CUDA 12.x support
RUN pip install ctranslate2==4.3.1 --extra-index-url https://pip.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121

# Install faster-whisper (will use the already installed torch and ctranslate2)
RUN pip install faster-whisper

# Install remaining packages from requirements.txt
ARG TORCH_CUDA_ARCH_LIST="Pascal;Turing;Ampere;Ada;Hopper"
RUN pip install --no-cache-dir -r requirements.txt

# Sensible defaults; override via docker-compose or environment
ENV TRANSCRIBE_PORT=8123
ENV TRANSCRIBE_WORKERS=4

# Copy the application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8123

# Command to run multi-worker uvicorn with uvloop/httptools
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${TRANSCRIBE_PORT:-8123} --workers ${TRANSCRIBE_WORKERS:-4} --loop uvloop --http httptools"]