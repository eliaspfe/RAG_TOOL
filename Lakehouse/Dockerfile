FROM python:3.11-slim

# System dependencies for hi_res PDF processing (poppler) and optional OCR (tesseract)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set a stable, in-image cache path for Hugging Face models
ENV HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    HUGGINGFACE_HUB_CACHE=/app/cache/hf \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Pre-fetch model artifacts so the runtime container can start offline
RUN mkdir -p /app/cache/hf && python - <<'PY'
from transformers import AutoProcessor, AutoModelForObjectDetection
from huggingface_hub import hf_hub_download

# Table Transformer used for table structure recognition
model_id = "microsoft/table-transformer-structure-recognition"
AutoProcessor.from_pretrained(model_id)
AutoModelForObjectDetection.from_pretrained(model_id)

# Timm ResNet18 weights (used by unstructured/table models)
hf_hub_download(repo_id="timm/resnet18.a1_in1k", filename="model.safetensors")

# YOLO layout model used by unstructured (layout detection)
hf_hub_download(repo_id="unstructuredio/yolo_x_layout", filename="yolox_l0.05.onnx")
PY

# Copy the rest of the project
COPY . .

EXPOSE 8002

# Run as HTTP service; pipeline logic remains unchanged, only wrapped
ENTRYPOINT ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8002"]
