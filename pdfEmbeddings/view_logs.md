# Log-Abfrage Script

Mit diesem Script kannst du die Logs aus der Datenbank abfragen:

```python
import duckdb

# Verbinde zur Datenbank
conn = duckdb.connect('/app/db/vector.duckdb')

# Alle Logs anzeigen
print("\n=== ALLE LOGS ===")
logs = conn.execute("""
    SELECT timestamp, level, message, details 
    FROM processing_logs 
    ORDER BY timestamp DESC
""").fetchall()

for log in logs:
    print(f"[{log[0]}] {log[1]}: {log[2]}")
    if log[3]:
        print(f"  Details: {log[3]}")

# Nur Fehler anzeigen
print("\n=== FEHLER ===")
errors = conn.execute("""
    SELECT timestamp, message, details 
    FROM processing_logs 
    WHERE level = 'ERROR'
    ORDER BY timestamp DESC
""").fetchall()

for error in errors:
    print(f"[{error[0]}] {error[1]}")
    if error[2]:
        print(f"  Details: {error[2]}")

# Statistiken
print("\n=== STATISTIKEN ===")
stats = conn.execute("""
    SELECT 
        level,
        COUNT(*) as count
    FROM processing_logs
    GROUP BY level
    ORDER BY count DESC
""").fetchall()

for stat in stats:
    print(f"{stat[0]}: {stat[1]}")

conn.close()
```

## Im Docker Container:

```bash
# Logs anzeigen
docker-compose exec embedding-processor python -c "
import duckdb
conn = duckdb.connect('/app/db/vector.duckdb')
for log in conn.execute('SELECT timestamp, level, message FROM processing_logs ORDER BY timestamp DESC LIMIT 20').fetchall():
    print(f'[{log[0]}] {log[1]}: {log[2]}')
"

# Nur Fehler
docker-compose exec embedding-processor python -c "
import duckdb
conn = duckdb.connect('/app/db/vector.duckdb')
for log in conn.execute(\"SELECT timestamp, level, message, details FROM processing_logs WHERE level='ERROR' ORDER BY timestamp DESC\").fetchall():
    print(f'[{log[0]}] {log[2]}')
    if log[3]: print(f'Details: {log[3]}')
"
```
