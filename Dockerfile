FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (like ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
# Ensure faster-whisper uses the correct CUDA version if available
ARG TORCH_CUDA_ARCH_LIST="Pascal;Turing;Ampere;Ada;Hopper"
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8123

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8123"] 