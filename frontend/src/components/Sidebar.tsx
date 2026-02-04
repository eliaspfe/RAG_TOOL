import { useEffect, useState } from "react";

export default function Sidebar() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [status, setStatus] = useState<string>("Keine laufende Verarbeitung");
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);

  const fetchUploadedDocs = async () => {
    const res = await fetch("http://127.0.0.1:8000/list_pdfs");
    const data = await res.json();
    setUploadedDocs(data.files);
  };

  useEffect(() => {
    fetchUploadedDocs();
  }, []);

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
    fetchUploadedDocs(); // Liste neu laden
  };

  const buildIndex = async () => {
    setStatus("Index wird aufgebaut...");

    try {
      const res = await fetch("http://127.0.0.1:8000/build_index", {
        method: "POST",
      });

      if (!res.ok) throw new Error();

      setStatus("Index erfolgreich aufgebaut");
    } catch {
      setStatus("Fehler beim Index-Aufbau");
    }
  };

  const deleteIndex = async () => {
  const ok = confirm("Willst du den Index wirklich löschen?");
  if (!ok) return;

  setStatus("Index wird gelöscht...");

  try {
    const res = await fetch("http://127.0.0.1:8000/delete_index", {
      method: "POST",
    });

    if (!res.ok) throw new Error();

    setStatus("Index gelöscht");
    setUploadedDocs([]);
  } catch {
    setStatus("Fehler beim Löschen des Index");
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
        <label>Hochgeladene Dokumente</label>
        <ul>
          {uploadedDocs.length === 0 && <li>Keine Dokumente</li>}
          {uploadedDocs.map((doc) => (
            <li key={doc}>{doc}</li>
          ))}
        </ul>
      </div>

      <div className="section">
        <label>Status</label>
        <div className="status">{status}</div>
        <button className="secondary" onClick={buildIndex}>
          Index neu aufbauen
        </button>
        <button className="secondary" onClick={deleteIndex}>
          Index und pdfs löschen
        </button>
      </div>

      
    </aside>

  );
}
