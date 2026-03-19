# LRAG – Local Retrieval-Augmented Generation Tool

LRAG ist ein Tool zum Aufbau und Betrieb einer modularen **RAG-Pipeline (Retrieval-Augmented Generation)**.  
Die Anwendung besteht aus mehreren Services und wird vollständig über **Docker Compose** orchestriert.

Ziel ist es, Dokumente zu verarbeiten, Vektoreinbettungen zu erzeugen und diese für Anfragen über ein Frontend nutzbar zu machen.

Alle Services werden über die Datei `docker-compose.yaml` gestartet und miteinander vernetzt.

# Voraussetzungen

- Docker
- Docker Compose

---

# Starten der Anwendung

1. .env-Datei im backend Ordner erstellen und API-KEY einfügen (Dafür kann `env_template.txt` genutzt werden)
2. Im Terminal:

```bash
docker compose up --build
```

---
