FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt


COPY backend.py .
COPY ragpipeline.py .
COPY .env .

COPY dataLake.py .
COPY initialize_db.sql .

EXPOSE 8000

CMD [ "python", "backend.py" ]