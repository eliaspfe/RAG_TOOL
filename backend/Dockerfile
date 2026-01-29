FROM python:3.12-slim

WORKDIR /app

COPY backend.py .
COPY ragpipeline.py .
COPY .env .
COPY requirements.txt .
COPY sample.pdf .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "python", "backend.py" ]