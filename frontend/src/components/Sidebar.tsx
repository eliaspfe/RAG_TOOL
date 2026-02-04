import { useState } from "react";

export default function Sidebar() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [status, setStatus] = useState<string>("Keine laufende Verarbeitung");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    setFiles(e.target.files);
  };

  const uploadPdfs = async () => {
    if (!files || files.length === 0) {
      alert("Keine Dateien ausgewählt");
      return;
    }

    setStatus("Lade PDFs hoch...");

    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append("file", file);

      await fetch("http://127.0.0.1:8000/upload_pdf", {
        method: "POST",
        body: formData,
      });
    }

    setStatus("Upload abgeschlossen");
  };

  const buildIndex = async () => {
    setStatus("Index wird aufgebaut...");

    try {
      const res = await fetch("http://127.0.0.1:8000/build_index", {
        method: "POST",
      });

      if (!res.ok) throw new Error("Index-Fehler");

      setStatus("Index erfolgreich aufgebaut");
    } catch {
      setStatus("Fehler beim Index-Aufbau");
    }
  };

  return (
    <aside className="sidebar">
      <h2>Datenquellen</h2>

      <div className="section">
        <label>PDFs hochladen</label>
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={handleFileChange}
        />
        <button onClick={uploadPdfs}>In VektorDB speichern</button>
      </div>

      <div className="section">
        <label>Status</label>
        <div className="status">{status}</div>
        <button className="secondary" onClick={buildIndex}>
          Index neu aufbauen
        </button>
      </div>
    </aside>
  );
}
