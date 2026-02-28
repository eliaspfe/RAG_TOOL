FROM python:3.12-slim

WORKDIR /app

# Installiere System-Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiere Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Service
COPY llm_service.py .

# Umgebungsvariablen
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

EXPOSE 8002

CMD ["python", "llm_service.py"]