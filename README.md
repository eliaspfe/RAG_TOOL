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

1. .env-Datei erstellen und API-KEY einfügen (Dafür kann `env_template.txt` genutzt werden)
2. Im Terminal:

```bash
docker compose up --build
```

---

# Architektur

Die Anwendung ist in mehrere Services aufgeteilt:

---

## Frontend

Das Frontend basiert auf **Vite** mit **TypeScript/JavaScript** und dient als Benutzeroberfläche für die Interaktion mit der RAG-Pipeline.

Es stellt ein Chatinterface bereit, über das Benutzer mit dem angebundenen Large Language Model (LLM) kommunizieren können. Zusätzlich besteht die Möglichkeit, Dateien hochzuladen, die in den Retrieval-Prozess eingebunden werden.

### Funktionen

- **Chatoberfläche**
  - Texteingabe für Benutzeranfragen
  - Anzeige der vom LLM generierten Antworten
  - Darstellung des Konversationsverlaufs

- **Datei-Upload**
  - Upload von Dokumenten über das Webinterface
  - Übergabe der Dateien an das Backend zur weiteren Verarbeitung

- **Kommunikation mit dem Backend**
  - Nutzung von HTTP-Requests (REST) zur:
    - Übermittlung von Chatnachrichten
    - Übertragung hochgeladener Dateien
    - Abfrage generierter Antworten
  - Trennung von UI-Logik und Backend-Verarbeitung

### Technische Basis

- Build-Tool: **Vite**
- Programmiersprache: **TypeScript / JavaScript**
- Webtechnologien: HTML, CSS
- Modularer Aufbau der UI-Komponenten

### Ablauf im Frontend (vereinfacht)

1. Der Benutzer gibt eine Nachricht in das Chatfeld ein.
2. Optional werden Dateien über das Upload-Feld ausgewählt.
3. Die Anfrage und die Dateien werden an das Backend gesendet.
4. Die Antwort des Backends wird empfangen.
5. Die Antwort wird im Chatfenster dargestellt.

### Ziel des Frontends

- Bereitstellung einer einfachen und intuitiven Benutzeroberfläche
- Abstraktion der internen RAG-Logik vom Benutzer
- Ermöglichung der Kombination aus:
  - natürlicher Sprache (Chat)
  - dokumentenbasierter Wissensabfrage (Upload)

---

## Backend

Das Backend ist in **Python** mit **FastAPI** implementiert und stellt die zentrale Steuerung der RAG-Pipeline bereit.  
Es verarbeitet Datei-Uploads, nimmt LLM-Anfragen entgegen und koordiniert die einzelnen Schritte der Retrieval- und Generierungslogik.

### API-Routen

Das Backend stellt zwei Haupt-Endpunkte bereit:

- **Upload-Route**
  - Entgegennahme hochgeladener Dateien vom Frontend
  - Weiterleitung der Dateien an die RAG-Pipeline zur Verarbeitung
  - Initiale Speicherung und Aufbereitung der Dokumente

- **LLM-Request-Route**
  - Entgegennahme von Benutzeranfragen aus dem Chat
  - Durchführung von Retrieval (Ähnlichkeitssuche)
  - Zusammenstellung eines finalen Prompts
  - Übergabe an das LLM und Rückgabe der Antwort an das Frontend

### RagPipeline-Objekt

Die zentrale Logik des Backends ist im Objekt **`RagPipeline`** gekapselt.  
Dieses Objekt verwaltet Datenhaltung, Vorverarbeitung, Retrieval und Prompt-Erstellung.

#### Datenhaltung (DuckDB)

Beim Initialisieren der `RagPipeline` wird eine **DuckDB-Datenbank** erzeugt bzw. geladen.  
Diese ist in mehrere logische Layer unterteilt:

- **Bronze Layer**  
  Rohdaten der hochgeladenen Dateien

- **Silber Layer**  
  Bereinigte und vorverarbeitete Inhalte

- **Gold Layer**  
  Weiterverarbeitete, für Retrieval geeignete Daten

Zusätzlich existiert ein weiteres Schema mit einer Tabelle für die **Embedded Chunks** der Dokumente.

#### Chunk- und Embedding-Tabelle

- Enthält:
  - Text-Chunks der Dokumente
  - Zugehörige Vektoreinbettungen
- Dient als Grundlage für:
  - Ähnlichkeitssuche
  - Kontextzusammenstellung für das LLM

### Zentrale Funktionen der RagPipeline

Die `RagPipeline` stellt mehrere Kernfunktionen bereit:

- **Cleaning durch die Layer (Bronze → Silber → Gold)**
  - Aufbereitung und Normalisierung der Dokumentinhalte
  - Strukturierung für die weitere Verarbeitung

- **Chunking der Dokumente**
  - Zerlegung der Inhalte in kleinere Textsegmente
  - Vorbereitung für das Embedding

- **Embedding der Chunks**
  - Erzeugung von Vektorrepräsentationen
  - Speicherung in der zusätzlichen Chunk-Tabelle

- **Ähnlichkeitssuche**
  - Vergleich von Anfrage-Embeddings mit gespeicherten Chunk-Vektoren
  - Auswahl der relevantesten Textsegmente

- **Prompt-Zusammenstellung**
  - Kombination aus:
    - Benutzerfrage
    - gefundenem Kontext
  - Erzeugung eines finalen Prompts für das LLM

### Chunk-Cleaning

Die Chunk-Bereinigung im Backend nutzt immer `unstructured.cleaners` im Fast-Mode
(z. B. Whitespace/Bullets/Dashes/Unicode-Quotes).

Hinweis:
- Das Backend startet nur, wenn `unstructured` installiert ist.
- Nach Änderungen an Dependencies bitte den Backend-Container neu bauen.

### Ablauf im Backend (vereinfacht)

1. Datei wird über die Upload-Route empfangen.
2. Speicherung im Bronze Layer.
3. Bereinigung und Transformation über Silber- und Gold-Layer.
4. Chunking der Inhalte.
5. Embedding der Chunks und Speicherung in der Chunk-Tabelle.
6. Benutzer stellt eine Anfrage über die LLM-Route.
7. Anfrage wird embedded.
8. Ähnlichkeitssuche in der Vektortabelle.
9. Zusammenstellung des finalen Prompts.
10. Übergabe an das LLM und Rückgabe der Antwort.

### Ziel des Backends

- Trennung von:
  - API-Schicht (FastAPI)
  - Pipeline-Logik (`RagPipeline`)
- Nachvollziehbare Verarbeitung über Layer-Modell
- Erweiterbarkeit:
  - andere Embedding-Modelle
  - andere Vektorspeicher
  - zusätzliche Verarbeitungsschritte

---

## Embedding-Service

Der Embedding-Service ist ein eigenständiger Microservice, implementiert in **Python** mit **FastAPI**.  
Er stellt ein Embedding-Modell bereit, das Texte in Vektorrepräsentationen umwandelt und vom Backend für die RAG-Pipeline genutzt wird.

### Zweck

Der Service dient ausschließlich der:

- Erzeugung von Vektoreinbettungen für Text-Chunks
- Erzeugung von Vektoreinbettungen für Benutzeranfragen

Damit ist er von der eigentlichen RAG-Logik getrennt und kann unabhängig skaliert oder ausgetauscht werden.

---

### Modell

Beim Start des Services wird ein Embedding-Modell geladen:

- Standardmodell:  
  `jinaai/jina-embeddings-v2-base-de`
- Konfigurierbar über die Umgebungsvariable: `EMBEDDING_MODEL` im docker-compose.yaml

---
