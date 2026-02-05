FROM python:3.11-slim

# System dependencies required by unstructured/cv2 at import time
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep CPU thread usage bounded in container runtime
ENV HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    HUGGINGFACE_HUB_CACHE=/app/cache/hf \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Copy the rest of the project
COPY . .

EXPOSE 8002

# Run as HTTP service; pipeline logic remains unchanged, only wrapped
ENTRYPOINT ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8002"]
