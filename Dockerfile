FROM python:3.12-slim

WORKDIR /app

COPY backend.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "python", "backend.py" ]